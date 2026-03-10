//! Core frame types: [`FrameEnvelope`], [`PixelFormat`], and pixel data.

use std::any::Any;
use std::sync::Arc;

use nv_core::{FeedId, MonotonicTs, TypedMetadata, WallTs};

/// Pixel format of the decoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit RGB, 3 bytes per pixel.
    Rgb8,
    /// 8-bit BGR, 3 bytes per pixel.
    Bgr8,
    /// 8-bit RGBA, 4 bytes per pixel.
    Rgba8,
    /// NV12 (Y plane + interleaved UV), 12 bits per pixel.
    Nv12,
    /// I420 (Y + U + V planes), 12 bits per pixel.
    I420,
    /// 8-bit grayscale, 1 byte per pixel.
    Gray8,
}

impl PixelFormat {
    /// Bytes per pixel for packed formats.
    ///
    /// Returns `None` for planar formats (NV12, I420) where the concept
    /// of "bytes per pixel" is ambiguous.
    #[must_use]
    pub fn bytes_per_pixel(&self) -> Option<u32> {
        match self {
            Self::Rgb8 | Self::Bgr8 => Some(3),
            Self::Rgba8 => Some(4),
            Self::Gray8 => Some(1),
            Self::Nv12 | Self::I420 => None,
        }
    }
}

/// Opaque guard that keeps a zero-copy memory mapping alive.
///
/// When this guard drops, the underlying buffer (e.g., a GStreamer mapped buffer)
/// is released. The concrete type is set by `nv-media` and erased here to avoid
/// exposing GStreamer types.
pub(crate) type PinGuard = Box<dyn Any + Send + Sync>;

/// How pixel data is stored.
pub(crate) enum PixelData {
    /// Zero-copy: pointer into a memory-mapped buffer (e.g., GStreamer).
    /// The `_guard` holds the mapping alive; dropping it releases the buffer.
    Mapped {
        ptr: *const u8,
        len: usize,
        _guard: PinGuard,
    },
    /// Owned copy — used for synthetic/test frames or when zero-copy isn't available.
    Owned(Vec<u8>),
}

// SAFETY: `Mapped` variant holds an immutable pointer (`*const u8`) into a
// buffer whose lifetime is managed by `_guard` (a trait-object box that is
// `Send + Sync`). The contract on `new_mapped()` requires:
//   1. The pointer remains valid as long as the guard lives.
//   2. The data at the pointer is never mutated while any `FrameEnvelope`
//      clone referencing it exists.
// These invariants ensure the pointee is `Sync`-safe (immutable shared data)
// and `Send`-safe (the guard is Send).
unsafe impl Send for PixelData {}
unsafe impl Sync for PixelData {}

/// Internal frame data behind the `Arc`.
pub(crate) struct FrameInner {
    pub feed_id: FeedId,
    pub seq: u64,
    pub ts: MonotonicTs,
    pub wall_ts: WallTs,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub stride: u32,
    pub data: PixelData,
    pub metadata: TypedMetadata,
}

/// An immutable, ref-counted video frame.
///
/// `Clone` is cheap — it bumps the `Arc` reference count. Pixel data is
/// never copied between stages after construction.
///
/// Construct via [`FrameEnvelope::new_owned`] for test/synthetic frames,
/// or via the zero-copy bridge in `nv-media` for live streams.
#[derive(Clone)]
pub struct FrameEnvelope {
    pub(crate) inner: Arc<FrameInner>,
}

impl FrameEnvelope {
    /// Create a frame with owned pixel data.
    ///
    /// This copies the data into the frame. Use the zero-copy bridge
    /// in `nv-media` for production paths.
    #[must_use]
    pub fn new_owned(
        feed_id: FeedId,
        seq: u64,
        ts: MonotonicTs,
        wall_ts: WallTs,
        width: u32,
        height: u32,
        format: PixelFormat,
        stride: u32,
        data: Vec<u8>,
        metadata: TypedMetadata,
    ) -> Self {
        Self {
            inner: Arc::new(FrameInner {
                feed_id,
                seq,
                ts,
                wall_ts,
                width,
                height,
                format,
                stride,
                data: PixelData::Owned(data),
                metadata,
            }),
        }
    }

    /// Create a frame with a zero-copy mapped buffer.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to `len` readable bytes.
    /// - The `guard` must keep the underlying buffer alive for the lifetime
    ///   of this frame (and all its clones).
    /// - The data at `ptr` must not be mutated while any clone of this frame exists.
    #[must_use]
    pub unsafe fn new_mapped(
        feed_id: FeedId,
        seq: u64,
        ts: MonotonicTs,
        wall_ts: WallTs,
        width: u32,
        height: u32,
        format: PixelFormat,
        stride: u32,
        ptr: *const u8,
        len: usize,
        guard: PinGuard,
        metadata: TypedMetadata,
    ) -> Self {
        Self {
            inner: Arc::new(FrameInner {
                feed_id,
                seq,
                ts,
                wall_ts,
                width,
                height,
                format,
                stride,
                data: PixelData::Mapped {
                    ptr,
                    len,
                    _guard: guard,
                },
                metadata,
            }),
        }
    }

    /// The feed this frame originated from.
    #[must_use]
    pub fn feed_id(&self) -> FeedId {
        self.inner.feed_id
    }

    /// Monotonic frame counter within this feed session.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.inner.seq
    }

    /// Monotonic timestamp (nanoseconds since feed start).
    #[must_use]
    pub fn ts(&self) -> MonotonicTs {
        self.inner.ts
    }

    /// Wall-clock timestamp (for output/provenance only).
    #[must_use]
    pub fn wall_ts(&self) -> WallTs {
        self.inner.wall_ts
    }

    /// Frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.inner.width
    }

    /// Frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.inner.height
    }

    /// Pixel format of the decoded frame.
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.inner.format
    }

    /// Row stride in bytes.
    #[must_use]
    pub fn stride(&self) -> u32 {
        self.inner.stride
    }

    /// Borrow the raw pixel data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        match &self.inner.data {
            PixelData::Owned(v) => v.as_slice(),
            // SAFETY: the PinGuard keeps the buffer alive and immutable.
            PixelData::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }

    /// Reference to per-frame metadata.
    #[must_use]
    pub fn metadata(&self) -> &TypedMetadata {
        &self.inner.metadata
    }
}

impl std::fmt::Debug for FrameEnvelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameEnvelope")
            .field("feed_id", &self.inner.feed_id)
            .field("seq", &self.inner.seq)
            .field("ts", &self.inner.ts)
            .field("dims", &(self.inner.width, self.inner.height))
            .field("format", &self.inner.format)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame() -> FrameEnvelope {
        FrameEnvelope::new_owned(
            FeedId::new(1),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            320,
            240,
            PixelFormat::Rgb8,
            960,
            vec![0u8; 320 * 240 * 3],
            TypedMetadata::new(),
        )
    }

    #[test]
    fn clone_is_cheap() {
        let f1 = test_frame();
        let f2 = f1.clone();
        assert!(std::ptr::eq(f1.data().as_ptr(), f2.data().as_ptr()));
    }

    #[test]
    fn accessors() {
        let f = test_frame();
        assert_eq!(f.width(), 320);
        assert_eq!(f.height(), 240);
        assert_eq!(f.format(), PixelFormat::Rgb8);
        assert_eq!(f.data().len(), 320 * 240 * 3);
    }

    // -- Send/Sync invariant tests --

    /// Compile-time assertion: `PixelData` is `Send`.
    const _: () = {
        const fn assert_send<T: Send>() {}
        assert_send::<PixelData>();
    };

    /// Compile-time assertion: `PixelData` is `Sync`.
    const _: () = {
        const fn assert_sync<T: Sync>() {}
        assert_sync::<PixelData>();
    };

    /// Compile-time assertion: `FrameEnvelope` is `Send + Sync`.
    const _: () = {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FrameEnvelope>();
    };

    /// Runtime test: owned frames can be sent across threads.
    #[test]
    fn frame_is_send_across_threads() {
        let f = test_frame();
        let handle = std::thread::spawn(move || {
            assert_eq!(f.width(), 320);
            f
        });
        let f = handle.join().unwrap();
        assert_eq!(f.height(), 240);
    }

    /// Runtime test: cloned frames can be shared across threads (Sync).
    #[test]
    fn frame_is_sync_across_threads() {
        let f = Arc::new(test_frame());
        let f2 = Arc::clone(&f);
        let handle = std::thread::spawn(move || {
            assert_eq!(f2.data().len(), 320 * 240 * 3);
        });
        assert_eq!(f.width(), 320);
        handle.join().unwrap();
    }

    /// Runtime test: mapped frames maintain invariants across Send boundary.
    #[test]
    fn mapped_frame_send_invariant() {
        let data = vec![1u8, 2, 3, 4, 5, 6];
        let ptr = data.as_ptr();
        let len = data.len();
        let guard: PinGuard = Box::new(data);

        let f = unsafe {
            FrameEnvelope::new_mapped(
                FeedId::new(1),
                0,
                MonotonicTs::ZERO,
                WallTs::from_micros(0),
                2,
                1,
                PixelFormat::Rgb8,
                6,
                ptr,
                len,
                guard,
                TypedMetadata::new(),
            )
        };
        let handle = std::thread::spawn(move || {
            assert_eq!(f.data(), &[1, 2, 3, 4, 5, 6]);
        });
        handle.join().unwrap();
    }
}
