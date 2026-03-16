//! Core frame types: [`FrameEnvelope`], [`PixelFormat`], [`Residency`], [`DataAccess`],
//! and pixel data.

use std::any::Any;
use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

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

/// Where the frame's pixel data physically resides.
///
/// `Residency` describes **storage location**, not access semantics.
/// A host-resident frame stores pixel bytes in CPU-accessible memory.
/// A device-resident frame stores data on an accelerator (GPU, NPU,
/// DMA‑buf, etc.) where host-readable bytes are not directly available.
///
/// # Access pattern
///
/// The canonical way to handle frames in CPU/GPU-mixed pipelines:
///
/// ```ignore
/// match frame.data_access() {
///     DataAccess::HostReadable => {
///         let bytes = frame.host_data().unwrap();
///     }
///     DataAccess::MappableToHost => {
///         let pixels = frame.require_host_data()?;
///     }
///     DataAccess::Opaque => {
///         let handle = frame.accelerated_handle::<MyBuffer>().unwrap();
///     }
///     _ => {}
/// }
/// ```
///
/// For pure CPU stages, [`FrameEnvelope::require_host_data()`] handles
/// all three cases with a single call.
///
/// # Relationship to [`DataAccess`]
///
/// `Residency` describes **where** the data lives. [`DataAccess`] describes
/// **what host-access is available**:
///
/// | `Residency` | `DataAccess` | Meaning |
/// |---|---|---|
/// | `Host` | `HostReadable` | CPU bytes directly available |
/// | `Device` | `MappableToHost` | Device buffer, host-downloadable |
/// | `Device` | `Opaque` | Device buffer, no host path |
///
/// Use [`FrameEnvelope::data_access()`] when you need the finer-grained
/// access classification.
///
/// # Adapter crates
///
/// This enum deliberately does not name any vendor or backend. Concrete
/// buffer types (CUDA buffers, OpenCL images, DMA‑buf fds, …) are defined
/// by adapter crates and stored behind a type‑erased handle accessible via
/// [`FrameEnvelope::accelerated_handle()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Residency {
    /// Pixel data is in CPU-accessible memory.
    ///
    /// [`FrameEnvelope::host_data()`] returns `Some(&[u8])`.
    Host,
    /// Pixel data resides on an accelerator device.
    ///
    /// Host-readable bytes may or may not be obtainable —
    /// check [`FrameEnvelope::data_access()`] for details.
    /// Use [`FrameEnvelope::accelerated_handle()`] to obtain the
    /// type-erased handle.
    Device,
}

/// What host-access is available for a frame's pixel data.
///
/// This is the finer-grained companion to [`Residency`]. Where
/// `Residency` says **where** the data lives, `DataAccess` says
/// **how** a CPU consumer can (or cannot) obtain host bytes.
///
/// Retrieve via [`FrameEnvelope::data_access()`].
///
/// # Canonical usage
///
/// ```ignore
/// match frame.data_access() {
///     DataAccess::HostReadable => {
///         let bytes = frame.host_data().unwrap();
///     }
///     DataAccess::MappableToHost => {
///         let pixels = frame.require_host_data()?;
///     }
///     DataAccess::Opaque => {
///         let handle = frame.accelerated_handle::<MyAccelBuf>();
///     }
///     _ => { /* forward-compatible: handle future access classes */ }
/// }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataAccess {
    /// Host-readable bytes are directly available (zero-copy borrow).
    HostReadable,
    /// Device-resident, but host materialization is available.
    ///
    /// The first call to [`FrameEnvelope::require_host_data()`] invokes
    /// the backend materializer and caches the result in the frame's
    /// `Arc`-shared inner state. Subsequent calls return a zero-copy
    /// borrow of the cached bytes.
    MappableToHost,
    /// Opaque accelerated data; no host view is guaranteed.
    ///
    /// [`FrameEnvelope::require_host_data()`] returns
    /// [`FrameAccessError::NotHostAccessible`].
    Opaque,
}

/// Errors from frame data access operations.
///
/// Returned by [`FrameEnvelope::require_host_data()`] when host bytes
/// cannot be obtained.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FrameAccessError {
    /// The frame is device-resident with no host materialization path.
    #[error("frame is not host-accessible (opaque device-resident data)")]
    NotHostAccessible,

    /// A host-materialization path exists but the transfer failed.
    #[error("host materialization failed: {detail}")]
    MaterializationFailed {
        /// Backend-provided failure description.
        detail: String,
    },
}

/// A function that materializes device-resident pixel data to host memory.
///
/// Adapter crates provide this when constructing device frames that support
/// host fallback (e.g., GPU buffers that can be mapped or downloaded).
///
/// Called by [`FrameEnvelope::require_host_data()`] on
/// [`DataAccess::MappableToHost`] frames. The result is automatically
/// cached per frame — only the first call invokes the closure.
pub type HostMaterializeFn = Box<dyn Fn() -> Result<HostBytes, FrameAccessError> + Send + Sync>;

/// Host-readable bytes produced by materializing device-resident data.
///
/// Adapter crates return this from their [`HostMaterializeFn`] closures.
/// Two construction paths are available:
///
/// - [`HostBytes::from_vec`] — owned copy (e.g., device-to-host download).
/// - [`HostBytes::from_mapped`] — zero-copy mapped view with a lifetime
///   guard (e.g., a memory-mapped device buffer).
pub struct HostBytes {
    repr: HostBytesRepr,
}

enum HostBytesRepr {
    Owned(Vec<u8>),
    Mapped {
        ptr: *const u8,
        len: usize,
        _guard: Box<dyn Any + Send + Sync>,
    },
}

// SAFETY: The `Mapped` variant holds an immutable `*const u8` into a buffer
// whose lifetime is managed by `_guard` (a `Send + Sync` trait-object box).
// The [`HostBytes::from_mapped`] contract requires the pointer to remain
// valid and immutable while the guard is alive — the same invariants as
// `PixelData::Mapped`.
unsafe impl Send for HostBytes {}
unsafe impl Sync for HostBytes {}

impl HostBytes {
    /// Create host bytes from an owned allocation.
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self {
            repr: HostBytesRepr::Owned(data),
        }
    }

    /// Create host bytes from a zero-copy mapped view.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to at least `len` readable bytes.
    /// - `guard` must keep the underlying buffer alive.
    /// - The data at `ptr` must not be mutated while `guard` exists.
    #[must_use]
    pub unsafe fn from_mapped(
        ptr: *const u8,
        len: usize,
        guard: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            repr: HostBytesRepr::Mapped {
                ptr,
                len,
                _guard: guard,
            },
        }
    }
}

impl AsRef<[u8]> for HostBytes {
    fn as_ref(&self) -> &[u8] {
        match &self.repr {
            HostBytesRepr::Owned(v) => v.as_slice(),
            // SAFETY: constructor contract guarantees validity while guard lives.
            HostBytesRepr::Mapped { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

impl std::ops::Deref for HostBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_ref()
    }
}

impl std::fmt::Debug for HostBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.repr {
            HostBytesRepr::Owned(v) => write!(f, "HostBytes::Owned({} bytes)", v.len()),
            HostBytesRepr::Mapped { len, .. } => write!(f, "HostBytes::Mapped({len} bytes)"),
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
    /// Device-resident data (GPU, NPU, DMA-buf, etc.).
    ///
    /// The handle is type-erased; backend adapter crates define the
    /// concrete type and downcast via [`FrameEnvelope::accelerated_handle()`].
    Device {
        handle: Arc<dyn Any + Send + Sync>,
        /// Optional host-materialization callback.
        ///
        /// When present, [`FrameEnvelope::require_host_data()`] can
        /// download device data to a host `Vec<u8>`.
        materialize: Option<HostMaterializeFn>,
    },
}

// SAFETY: `Mapped` variant holds an immutable pointer (`*const u8`) into a
// buffer whose lifetime is managed by `_guard` (a trait-object box that is
// `Send + Sync`). The contract on `new_mapped()` requires:
//   1. The pointer remains valid as long as the guard lives.
//   2. The data at the pointer is never mutated while any `FrameEnvelope`
//      clone referencing it exists.
// These invariants ensure the pointee is `Sync`-safe (immutable shared data)
// and `Send`-safe (the guard is Send).
//
// `Device` variant holds an `Arc<dyn Any + Send + Sync>` which is
// inherently `Send + Sync`.
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
    /// Lazily cached host bytes for [`DataAccess::MappableToHost`] frames.
    /// Never touched by host-resident frames (the fast path bypasses it).
    ///
    /// # Memory note
    ///
    /// The cache lives inside the `Arc`-shared inner state, so it persists
    /// as long as **any** clone of this [`FrameEnvelope`] is alive. In
    /// fan-out topologies (e.g., multiple subscribers receiving clones of
    /// the same frame) the materialized bytes remain allocated until the
    /// last clone drops. Operators should account for this when sizing
    /// queue depths and subscriber counts on feeds with large device frames.
    pub host_cache: OnceLock<Result<HostBytes, FrameAccessError>>,
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
    #[allow(clippy::too_many_arguments)]
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
                host_cache: OnceLock::new(),
            }),
        }
    }

    /// Create a frame backed by a device-resident accelerated buffer.
    ///
    /// The handle is type-erased and stored behind an `Arc`. Backend
    /// adapter crates create the concrete handle type; stages that
    /// understand it recover it via
    /// [`accelerated_handle()`](Self::accelerated_handle).
    ///
    /// ## Host materialization
    ///
    /// If the backend can download device data to host memory, pass a
    /// `Some(materialize)` closure. This promotes the frame's
    /// [`DataAccess`] from [`Opaque`](DataAccess::Opaque) to
    /// [`MappableToHost`](DataAccess::MappableToHost), enabling CPU
    /// consumers to call [`require_host_data()`](Self::require_host_data).
    ///
    /// Pass `None` when host fallback is unavailable or undesirable.
    ///
    /// ## Geometry fields
    ///
    /// `width`, `height`, `format`, and `stride` still describe the logical
    /// image layout — they are meaningful metadata even when the raw bytes
    /// are not host-accessible.
    ///
    /// ## Intended uses for the opaque handle
    ///
    /// - Accelerated decode buffers (hardware-decoded video surfaces)
    /// - GPU tensors destined for inference
    /// - Accelerator-native frame storage (DMA-buf, OpenCL images, …)
    ///
    /// The handle must **not** be used for general stage metadata or
    /// cross-stage messaging. Use [`TypedMetadata`] for those purposes.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new_device(
        feed_id: FeedId,
        seq: u64,
        ts: MonotonicTs,
        wall_ts: WallTs,
        width: u32,
        height: u32,
        format: PixelFormat,
        stride: u32,
        handle: Arc<dyn Any + Send + Sync>,
        materialize: Option<HostMaterializeFn>,
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
                data: PixelData::Device {
                    handle,
                    materialize,
                },
                metadata,
                host_cache: OnceLock::new(),
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
    #[allow(clippy::too_many_arguments)]
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
                host_cache: OnceLock::new(),
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

    /// Where this frame's pixel data physically resides.
    ///
    /// For a finer-grained classification that distinguishes mappable
    /// device frames from opaque ones, see [`data_access()`](Self::data_access).
    ///
    /// ```ignore
    /// match frame.residency() {
    ///     Residency::Host   => { /* use host_data() */ }
    ///     Residency::Device => { /* use accelerated_handle::<T>() */ }
    /// }
    /// ```
    #[must_use]
    pub fn residency(&self) -> Residency {
        match &self.inner.data {
            PixelData::Owned(_) | PixelData::Mapped { .. } => Residency::Host,
            PixelData::Device { .. } => Residency::Device,
        }
    }

    /// What host-access is available for this frame's pixel data.
    ///
    /// This is the finer-grained companion to [`residency()`](Self::residency):
    ///
    /// | Return value | Meaning |
    /// |---|---|
    /// | [`HostReadable`](DataAccess::HostReadable) | [`host_data()`](Self::host_data) returns `Some`. |
    /// | [`MappableToHost`](DataAccess::MappableToHost) | [`require_host_data()`](Self::require_host_data) will materialize. |
    /// | [`Opaque`](DataAccess::Opaque) | No host path available. |
    #[must_use]
    pub fn data_access(&self) -> DataAccess {
        match &self.inner.data {
            PixelData::Owned(_) | PixelData::Mapped { .. } => DataAccess::HostReadable,
            PixelData::Device {
                materialize: Some(_),
                ..
            } => DataAccess::MappableToHost,
            PixelData::Device {
                materialize: None, ..
            } => DataAccess::Opaque,
        }
    }

    /// Whether host-readable pixel bytes are directly available.
    ///
    /// Equivalent to `self.data_access() == DataAccess::HostReadable`.
    /// Note: returns `false` for [`DataAccess::MappableToHost`] frames —
    /// use [`require_host_data()`](Self::require_host_data) to obtain
    /// host bytes from those frames.
    #[must_use]
    pub fn is_host_readable(&self) -> bool {
        !matches!(&self.inner.data, PixelData::Device { .. })
    }

    /// Host-readable pixel bytes, if available.
    ///
    /// Returns `Some(&[u8])` for host-resident frames ([`Residency::Host`]),
    /// `None` for device-resident frames ([`Residency::Device`]).
    ///
    /// This is the **zero-cost** accessor for the hot path when frames are
    /// known to be host-resident. For a fallback-aware accessor that can
    /// materialize device data, see [`require_host_data()`](Self::require_host_data).
    #[must_use]
    pub fn host_data(&self) -> Option<&[u8]> {
        match &self.inner.data {
            PixelData::Owned(v) => Some(v.as_slice()),
            // SAFETY: the PinGuard keeps the buffer alive and immutable.
            PixelData::Mapped { ptr, len, .. } => {
                Some(unsafe { std::slice::from_raw_parts(*ptr, *len) })
            }
            PixelData::Device { .. } => None,
        }
    }

    /// Obtain host-readable bytes, materializing from device if needed.
    ///
    /// This is the **primary ergonomic accessor** for CPU consumers that
    /// may receive either host-resident or device-resident frames.
    ///
    /// | Frame kind | First call | Subsequent calls |
    /// |---|---|---|
    /// | Host-resident | `Cow::Borrowed` (zero-copy) | Same |
    /// | Device + materializer | `Cow::Borrowed` (materialize → cache) | `Cow::Borrowed` (cached) |
    /// | Device, opaque | `Err(NotHostAccessible)` | Same |
    ///
    /// # Memoization
    ///
    /// For [`DataAccess::MappableToHost`] frames, the first call invokes
    /// the backend's [`HostMaterializeFn`] and caches the result in the
    /// frame's `Arc`-shared inner state. All subsequent calls (including
    /// from clones of this frame) return a zero-copy borrow of the cache.
    ///
    /// **Failures are also cached**: if the materializer returns an error,
    /// that error is retained and cloned on subsequent calls. Frame data
    /// is immutable, so a transfer that fails for a given frame will not
    /// succeed on retry.
    ///
    /// # Performance
    ///
    /// - **Host-resident frames**: the `OnceLock` cache is never touched;
    ///   this returns a direct borrow with zero overhead.
    /// - **First materialization**: may allocate and/or block (backend-dependent).
    /// - **Cached access**: zero-copy borrow from the cached [`HostBytes`].
    ///
    /// # Example: CPU-only stage
    ///
    /// ```ignore
    /// let pixels = frame.require_host_data()
    ///     .map_err(|e| StageError::ProcessingFailed {
    ///         stage_id: MY_STAGE,
    ///         detail: e.to_string(),
    ///     })?;
    /// process_cpu(&pixels);
    /// ```
    pub fn require_host_data(&self) -> Result<Cow<'_, [u8]>, FrameAccessError> {
        match &self.inner.data {
            PixelData::Owned(v) => Ok(Cow::Borrowed(v.as_slice())),
            // SAFETY: the PinGuard keeps the buffer alive and immutable.
            PixelData::Mapped { ptr, len, .. } => Ok(Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            })),
            PixelData::Device {
                materialize: Some(f),
                ..
            } => {
                let cached = self.inner.host_cache.get_or_init(|| f());
                match cached {
                    Ok(bytes) => Ok(Cow::Borrowed(bytes.as_ref())),
                    Err(e) => Err(e.clone()),
                }
            }
            PixelData::Device {
                materialize: None, ..
            } => Err(FrameAccessError::NotHostAccessible),
        }
    }

    /// Downcast the opaque accelerated handle to a concrete type `T`.
    ///
    /// Returns `Some(&T)` if the frame is device-resident and the handle
    /// is of type `T`. Returns `None` for host-resident frames or if the
    /// concrete type does not match.
    ///
    /// # Intended uses
    ///
    /// The accelerated handle is intended **only** for:
    ///
    /// - Backend adapter crates bridging accelerated decode buffers
    /// - GPU tensors destined for inference
    /// - Accelerator-native frame storage (DMA-buf, OpenCL images, …)
    ///
    /// It must **not** be used for general stage metadata, arbitrary
    /// payload storage, or cross-stage messaging. Use
    /// [`TypedMetadata`] for those purposes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// match frame.data_access() {
    ///     DataAccess::HostReadable => {
    ///         let bytes = frame.host_data().unwrap();
    ///         run_on_cpu(bytes);
    ///     }
    ///     DataAccess::MappableToHost | DataAccess::Opaque => {
    ///         let buf = frame.accelerated_handle::<MyAccelBuffer>()
    ///             .expect("expected MyAccelBuffer");
    ///         run_on_device(buf);
    ///     }
    ///     _ => {}
    /// }
    /// ```
    #[must_use]
    pub fn accelerated_handle<T: Send + Sync + 'static>(&self) -> Option<&T> {
        match &self.inner.data {
            PixelData::Device { handle, .. } => handle.downcast_ref::<T>(),
            _ => None,
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
            .field("residency", &self.residency())
            .field("data_access", &self.data_access())
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
        let d1 = f1.host_data().unwrap();
        let d2 = f2.host_data().unwrap();
        assert!(std::ptr::eq(d1.as_ptr(), d2.as_ptr()));
    }

    #[test]
    fn accessors() {
        let f = test_frame();
        assert_eq!(f.width(), 320);
        assert_eq!(f.height(), 240);
        assert_eq!(f.format(), PixelFormat::Rgb8);
        assert_eq!(f.host_data().unwrap().len(), 320 * 240 * 3);
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
            assert_eq!(f2.host_data().unwrap().len(), 320 * 240 * 3);
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
            assert_eq!(f.host_data().unwrap(), &[1, 2, 3, 4, 5, 6]);
        });
        handle.join().unwrap();
    }

    // -- Residency / accelerated-handle tests --

    /// A mock accelerated buffer used only in tests.
    #[derive(Debug, Clone, PartialEq)]
    struct MockGpuBuffer {
        device_id: u32,
        mem_handle: u64,
    }

    fn device_frame(surface: MockGpuBuffer) -> FrameEnvelope {
        FrameEnvelope::new_device(
            FeedId::new(2),
            10,
            MonotonicTs::from_nanos(5_000_000),
            WallTs::from_micros(100),
            1920,
            1080,
            PixelFormat::Nv12,
            1920,
            Arc::new(surface),
            None,
            TypedMetadata::new(),
        )
    }

    #[test]
    fn owned_frame_is_host_resident() {
        let f = test_frame();
        assert_eq!(f.residency(), Residency::Host);
        assert!(f.is_host_readable());
        assert!(f.host_data().is_some());
        assert_eq!(f.host_data().unwrap().len(), 320 * 240 * 3);
    }

    #[test]
    fn mapped_frame_is_host_resident() {
        let data = vec![42u8; 6];
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
        assert_eq!(f.residency(), Residency::Host);
        assert!(f.is_host_readable());
        assert_eq!(f.host_data(), Some([42u8; 6].as_slice()));
    }

    #[test]
    fn device_frame_residency() {
        let f = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0xDEAD,
        });
        assert_eq!(f.residency(), Residency::Device);
        assert!(!f.is_host_readable());
        assert!(f.host_data().is_none());
    }

    #[test]
    fn accelerated_handle_downcast() {
        let surface = MockGpuBuffer {
            device_id: 3,
            mem_handle: 0xBEEF,
        };
        let f = device_frame(surface.clone());
        let recovered = f.accelerated_handle::<MockGpuBuffer>().unwrap();
        assert_eq!(recovered, &surface);
    }

    #[test]
    fn accelerated_handle_wrong_type_returns_none() {
        let f = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0,
        });
        assert!(f.accelerated_handle::<String>().is_none());
    }

    #[test]
    fn host_frame_accelerated_handle_returns_none() {
        let f = test_frame();
        assert!(f.accelerated_handle::<MockGpuBuffer>().is_none());
    }

    #[test]
    fn device_frame_host_data_returns_none() {
        let f = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0,
        });
        assert!(f.host_data().is_none());
    }

    #[test]
    fn device_frame_preserves_metadata_and_geometry() {
        #[derive(Clone, Debug, PartialEq)]
        struct Tag(u32);

        let mut meta = TypedMetadata::new();
        meta.insert(Tag(99));

        let f = FrameEnvelope::new_device(
            FeedId::new(5),
            42,
            MonotonicTs::from_nanos(1_000),
            WallTs::from_micros(500),
            3840,
            2160,
            PixelFormat::Rgb8,
            3840 * 3,
            Arc::new(MockGpuBuffer {
                device_id: 1,
                mem_handle: 0xCAFE,
            }),
            None,
            meta,
        );

        assert_eq!(f.feed_id(), FeedId::new(5));
        assert_eq!(f.seq(), 42);
        assert_eq!(f.width(), 3840);
        assert_eq!(f.height(), 2160);
        assert_eq!(f.format(), PixelFormat::Rgb8);
        assert_eq!(f.stride(), 3840 * 3);
        assert_eq!(f.metadata().get::<Tag>(), Some(&Tag(99)));
    }

    #[test]
    fn device_frame_clone_shares_handle() {
        let f1 = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0xAA,
        });
        let f2 = f1.clone();
        // Both clones see the same handle.
        let s1 = f1.accelerated_handle::<MockGpuBuffer>().unwrap();
        let s2 = f2.accelerated_handle::<MockGpuBuffer>().unwrap();
        assert!(std::ptr::eq(s1, s2));
    }

    #[test]
    fn device_frame_is_send_sync() {
        let f = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0,
        });
        let handle = std::thread::spawn(move || {
            assert_eq!(f.residency(), Residency::Device);
            f.accelerated_handle::<MockGpuBuffer>().unwrap().device_id
        });
        assert_eq!(handle.join().unwrap(), 0);
    }

    /// Verify the canonical residency-branching pattern compiles and works.
    #[test]
    fn residency_branch_pattern() {
        let host = test_frame();
        let device = device_frame(MockGpuBuffer {
            device_id: 1,
            mem_handle: 0xFF,
        });

        for f in [host, device] {
            match f.residency() {
                Residency::Host => {
                    let bytes = f.host_data().expect("host-resident");
                    assert!(!bytes.is_empty());
                }
                Residency::Device => {
                    let buf = f
                        .accelerated_handle::<MockGpuBuffer>()
                        .expect("expected MockGpuBuffer");
                    assert_eq!(buf.device_id, 1);
                }
            }
        }
    }

    #[test]
    fn debug_includes_residency() {
        let f = test_frame();
        let dbg = format!("{f:?}");
        assert!(dbg.contains("Host"));

        let f2 = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0,
        });
        let dbg2 = format!("{f2:?}");
        assert!(dbg2.contains("Device"));
    }

    // -- DataAccess tests --

    #[test]
    fn host_frame_data_access() {
        let f = test_frame();
        assert_eq!(f.data_access(), DataAccess::HostReadable);
    }

    #[test]
    fn opaque_device_frame_data_access() {
        let f = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0,
        });
        assert_eq!(f.data_access(), DataAccess::Opaque);
    }

    #[test]
    fn mappable_device_frame_data_access() {
        let data = vec![10u8, 20, 30];
        let f = FrameEnvelope::new_device(
            FeedId::new(3),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            1,
            1,
            PixelFormat::Rgb8,
            3,
            Arc::new(MockGpuBuffer {
                device_id: 1,
                mem_handle: 0xAA,
            }),
            Some(Box::new(move || Ok(HostBytes::from_vec(data.clone())))),
            TypedMetadata::new(),
        );
        assert_eq!(f.residency(), Residency::Device);
        assert_eq!(f.data_access(), DataAccess::MappableToHost);
    }

    // -- require_host_data tests --

    #[test]
    fn require_host_data_host_frame_is_borrowed() {
        let f = test_frame();
        let cow = f.require_host_data().unwrap();
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.len(), 320 * 240 * 3);
    }

    #[test]
    fn require_host_data_mappable_device_materializes() {
        let expected = vec![1u8, 2, 3, 4, 5, 6];
        let expected_clone = expected.clone();
        let f = FrameEnvelope::new_device(
            FeedId::new(4),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            2,
            1,
            PixelFormat::Rgb8,
            6,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(move || Ok(HostBytes::from_vec(expected_clone.clone())))),
            TypedMetadata::new(),
        );
        let cow = f.require_host_data().unwrap();
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(&*cow, &expected[..]);
    }

    #[test]
    fn require_host_data_opaque_device_returns_error() {
        let f = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0,
        });
        let err = f.require_host_data().unwrap_err();
        assert!(matches!(err, FrameAccessError::NotHostAccessible));
    }

    #[test]
    fn require_host_data_materialization_failure() {
        let f = FrameEnvelope::new_device(
            FeedId::new(5),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            1,
            1,
            PixelFormat::Gray8,
            1,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(|| {
                Err(FrameAccessError::MaterializationFailed {
                    detail: "transfer timeout".into(),
                })
            })),
            TypedMetadata::new(),
        );
        let err = f.require_host_data().unwrap_err();
        assert!(matches!(
            err,
            FrameAccessError::MaterializationFailed { .. }
        ));
        assert!(err.to_string().contains("transfer timeout"));
    }

    /// Verify data_access branching pattern compiles and works.
    #[test]
    fn data_access_branch_pattern() {
        let host = test_frame();
        let data = vec![42u8; 6];
        let mappable = FrameEnvelope::new_device(
            FeedId::new(6),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            2,
            1,
            PixelFormat::Rgb8,
            6,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(move || Ok(HostBytes::from_vec(data.clone())))),
            TypedMetadata::new(),
        );
        let opaque = device_frame(MockGpuBuffer {
            device_id: 0,
            mem_handle: 0,
        });

        for f in [host, mappable, opaque] {
            match f.data_access() {
                DataAccess::HostReadable => {
                    assert!(f.host_data().is_some());
                    assert!(f.require_host_data().is_ok());
                }
                DataAccess::MappableToHost => {
                    assert!(f.host_data().is_none());
                    assert!(f.require_host_data().is_ok());
                }
                DataAccess::Opaque => {
                    assert!(f.host_data().is_none());
                    assert!(f.require_host_data().is_err());
                }
                _ => panic!("unexpected DataAccess variant"),
            }
        }
    }

    // -- HostBytes tests --

    /// Compile-time assertion: `HostBytes` is `Send + Sync`.
    const _: () = {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HostBytes>();
    };

    #[test]
    fn host_bytes_from_vec() {
        let hb = HostBytes::from_vec(vec![1, 2, 3]);
        assert_eq!(hb.as_ref(), &[1, 2, 3]);
        assert_eq!(&*hb, &[1, 2, 3]);
    }

    #[test]
    fn host_bytes_from_mapped_zero_copy() {
        let data = vec![10u8, 20, 30];
        let ptr = data.as_ptr();
        let len = data.len();
        let guard: Box<dyn Any + Send + Sync> = Box::new(data);
        let hb = unsafe { HostBytes::from_mapped(ptr, len, guard) };
        assert_eq!(hb.as_ref(), &[10, 20, 30]);
    }

    // -- Memoization tests --

    #[test]
    fn require_host_data_memoizes_materialization() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let call_count = Arc::new(AtomicU32::new(0));
        let cc = Arc::clone(&call_count);
        let f = FrameEnvelope::new_device(
            FeedId::new(10),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            1,
            1,
            PixelFormat::Gray8,
            1,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(move || {
                cc.fetch_add(1, Ordering::Relaxed);
                Ok(HostBytes::from_vec(vec![42]))
            })),
            TypedMetadata::new(),
        );

        let r1 = f.require_host_data().unwrap();
        let r2 = f.require_host_data().unwrap();
        assert_eq!(&*r1, &[42u8]);
        assert_eq!(&*r2, &[42u8]);
        // Both borrows reference the same cached bytes.
        assert!(std::ptr::eq(r1.as_ptr(), r2.as_ptr()));
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn require_host_data_cache_shared_across_clones() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let call_count = Arc::new(AtomicU32::new(0));
        let cc = Arc::clone(&call_count);
        let f1 = FrameEnvelope::new_device(
            FeedId::new(11),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            1,
            1,
            PixelFormat::Gray8,
            1,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(move || {
                cc.fetch_add(1, Ordering::Relaxed);
                Ok(HostBytes::from_vec(vec![99]))
            })),
            TypedMetadata::new(),
        );
        let f2 = f1.clone();

        let r1 = f1.require_host_data().unwrap();
        let r2 = f2.require_host_data().unwrap();
        assert_eq!(&*r1, &[99u8]);
        assert_eq!(&*r2, &[99u8]);
        // Materializer invoked exactly once across clones.
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn require_host_data_concurrent_access() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Barrier;

        let call_count = Arc::new(AtomicU32::new(0));
        let cc = Arc::clone(&call_count);
        let f = FrameEnvelope::new_device(
            FeedId::new(12),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            1,
            1,
            PixelFormat::Gray8,
            1,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(move || {
                cc.fetch_add(1, Ordering::Relaxed);
                Ok(HostBytes::from_vec(vec![7]))
            })),
            TypedMetadata::new(),
        );

        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let f = f.clone();
                let b = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    b.wait();
                    let r = f.require_host_data().unwrap();
                    assert_eq!(&*r, &[7u8]);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        // OnceLock guarantees at most one init.
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn require_host_data_failure_is_cached() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let call_count = Arc::new(AtomicU32::new(0));
        let cc = Arc::clone(&call_count);
        let f = FrameEnvelope::new_device(
            FeedId::new(13),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            1,
            1,
            PixelFormat::Gray8,
            1,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(move || {
                cc.fetch_add(1, Ordering::Relaxed);
                Err(FrameAccessError::MaterializationFailed {
                    detail: "device busy".into(),
                })
            })),
            TypedMetadata::new(),
        );

        let e1 = f.require_host_data().unwrap_err();
        let e2 = f.require_host_data().unwrap_err();
        assert!(e1.to_string().contains("device busy"));
        assert!(e2.to_string().contains("device busy"));
        // Materializer invoked exactly once; failure is cached.
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn require_host_data_host_frame_skips_cache() {
        let f = test_frame();
        let r1 = f.require_host_data().unwrap();
        let r2 = f.require_host_data().unwrap();
        assert!(matches!(r1, Cow::Borrowed(_)));
        assert!(matches!(r2, Cow::Borrowed(_)));
        // Both borrow the same underlying owned-data; no cache involved.
        assert!(std::ptr::eq(r1.as_ptr(), r2.as_ptr()));
    }

    #[test]
    fn require_host_data_mapped_materializer() {
        let f = FrameEnvelope::new_device(
            FeedId::new(14),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            1,
            3,
            PixelFormat::Gray8,
            1,
            Arc::new(MockGpuBuffer {
                device_id: 0,
                mem_handle: 0,
            }),
            Some(Box::new(|| {
                let data = vec![10u8, 20, 30];
                let ptr = data.as_ptr();
                let len = data.len();
                let guard: Box<dyn Any + Send + Sync> = Box::new(data);
                Ok(unsafe { HostBytes::from_mapped(ptr, len, guard) })
            })),
            TypedMetadata::new(),
        );

        let cow = f.require_host_data().unwrap();
        assert_eq!(&*cow, &[10, 20, 30]);
        // Second call: cached, same pointer.
        let cow2 = f.require_host_data().unwrap();
        assert!(std::ptr::eq(cow.as_ptr(), cow2.as_ptr()));
    }
}
