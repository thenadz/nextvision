//! CUDA-resident frame bridge.
//!
//! When the `cuda` feature is enabled, this module provides:
//!
//! - [`CudaBufferHandle`] — opaque handle to a CUDA device buffer
//!   extracted from a GStreamer CUDA-memory sample.
//! - [`bridge_gst_sample_device()`] — the GPU-path counterpart of
//!   [`bridge_gst_sample()`](crate::bridge::bridge_gst_sample). Wraps
//!   the CUDA device buffer in a [`FrameEnvelope`] with
//!   [`PixelData::Device`](nv_frame::PixelData::Device) and an optional
//!   [`HostMaterializeFn`](nv_frame::HostMaterializeFn) for CPU fallback.
//!
//! # Pipeline topology
//!
//! The CUDA-resident pipeline replaces the standard `videoconvert → appsink`
//! tail with:
//!
//! ```text
//! decode → [hook] → cudaupload → cudaconvert → appsink(video/x-raw(memory:CUDAMemory))
//! ```
//!
//! The resulting `GstSample` carries a `GstCudaMemory` buffer. This module
//! extracts the CUDA device pointer via GStreamer's buffer-mapping FFI,
//! wraps geometry and pointer in a [`CudaBufferHandle`], and stores the
//! handle in a device-resident [`FrameEnvelope`].
//!
//! # Device pointer extraction
//!
//! The CUDA device pointer is obtained at bridge time by mapping the
//! GStreamer buffer with the `GST_MAP_CUDA` flag (`GST_MAP_FLAG_LAST << 1`).
//! When a CUDA-aware GStreamer allocator handles this flag, the returned
//! `GstMapInfo.data` field is the `CUdeviceptr`.  The map is performed
//! and released inside this module — no raw FFI leaks to downstream code.
//! Callers simply read [`CudaBufferHandle::device_ptr`].
//!
//! # Device pointer lifetime
//!
//! The extracted `device_ptr` is expected to remain valid for the lifetime
//! of the [`CudaBufferHandle`] (and therefore the `FrameEnvelope` that
//! owns it).  This relies on GStreamer's CUDA buffer pool keeping the GPU
//! allocation stable while the `GstBuffer` has a non-zero refcount.
//! `CudaBufferHandle._gst_buffer` holds a strong reference to enforce
//! this.  **This is a GStreamer runtime guarantee, not something this
//! library can statically prove** — if a third-party allocator violates
//! the convention, the pointer may become invalid.
//!
//! # CUDA stream and context synchronization
//!
//! **This library does not manage CUDA contexts or streams.**  Doing so
//! would couple the core to CUDA runtime semantics and violate
//! domain-agnostic design.
//!
//! GStreamer's CUDA elements (`cudaupload`, `cudaconvert`, `nvh264dec`,
//! etc.) perform work on GStreamer's internal CUDA stream.  The appsink
//! callback fires after those elements have submitted their work, but
//! **the library does not verify that the GPU has finished executing**.
//! In practice GStreamer synchronizes before handing off the buffer, but
//! this is a GStreamer implementation detail — not a guarantee made by
//! this library.
//!
//! If a downstream stage launches CUDA kernels on a *different* CUDA
//! stream or context, the **caller is responsible for synchronizing**
//! before reading from or writing to `device_ptr`.  Typical patterns:
//!
//! - `cuStreamSynchronize` on GStreamer's stream before launching work
//!   on a different stream.
//! - `cuEventRecord` / `cuStreamWaitEvent` for fine-grained multi-stream
//!   coordination.
//! - For most single-GPU inference pipelines (TensorRT, cuDNN) that use
//!   the default stream, no explicit synchronization is needed — but
//!   verify this for your specific integration.
//!
//! # Host fallback
//!
//! Every device frame carries a [`HostMaterializeFn`](nv_frame::HostMaterializeFn)
//! that downloads the CUDA buffer to a host `Vec<u8>` by mapping the
//! GStreamer CUDA buffer readable — GStreamer's CUDA allocator handles
//! the device→host copy internally. CPU-only stages can call
//! [`FrameEnvelope::require_host_data()`](nv_frame::FrameEnvelope::require_host_data)
//! transparently — the materializer runs once and the result is cached.
//!
//! # GStreamer isolation
//!
//! `CudaBufferHandle` is *not* a GStreamer type publicly. The GStreamer
//! buffer reference is kept alive inside the handle to prevent pool
//! reclamation, but downstream code interacts only with the handle's
//! typed metadata fields (`device_ptr`, `width`, `height`, `stride`,
//! `format`, `len`).
//!
//! # No CUDA SDK required at build time
//!
//! This module does **not** link against the CUDA toolkit or GStreamer
//! CUDA headers. It uses `gst_buffer_map` / `gst_buffer_unmap` from the
//! `gstreamer-sys` crate plus the `GST_MAP_CUDA` flag constant. Accessing
//! `glib` types (e.g., `GFALSE`) goes through `gstreamer::glib` — no
//! direct `glib-rs` dependency is needed.  GStreamer CUDA plugins must
//! be available at *runtime*.

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nv_core::TypedMetadata;
use nv_core::error::MediaError;
use nv_core::id::FeedId;
use nv_core::timestamp::{MonotonicTs, WallTs};
use nv_frame::{FrameEnvelope, HostBytes, HostMaterializeFn, PixelFormat};

use crate::bridge::PtzTelemetry;
use crate::pipeline::OutputFormat;

/// GStreamer CUDA memory type string.
///
/// GStreamer's CUDA allocator sets this as the `mem_type` field when
/// creating `GstMemory` blocks backed by device memory (defined as
/// `GST_CUDA_MEMORY_TYPE_NAME` in gst-plugins-bad).  Checked at bridge
/// time via `gst_memory_is_type()` to verify the buffer actually carries
/// CUDA device memory before we attempt the `GST_MAP_CUDA` mapping.
const GST_CUDA_MEMORY_TYPE_NAME: &str = "cuda-memory";

/// `GST_MAP_CUDA` flag — tells GStreamer's CUDA allocator to return the
/// device pointer in `GstMapInfo.data` instead of staging a host copy.
///
/// Defined in gst-plugins-bad as `GST_MAP_FLAG_LAST << 1`.
/// `GST_MAP_FLAG_LAST` is `1 << 16 = 65536`, so `GST_MAP_CUDA = 1 << 17`.
const GST_MAP_CUDA: u32 = gstreamer::ffi::GST_MAP_FLAG_LAST << 1;

// ---------------------------------------------------------------------------
// CudaBufferHandle
// ---------------------------------------------------------------------------

/// Opaque handle to a CUDA device buffer extracted from a GStreamer sample.
///
/// Stages that understand CUDA recover this via
/// [`FrameEnvelope::accelerated_handle::<CudaBufferHandle>()`](nv_frame::FrameEnvelope::accelerated_handle).
///
/// The device pointer is extracted at bridge time inside `nv-media` —
/// downstream code never needs to call GStreamer FFI.
///
/// # Lifetime
///
/// `device_ptr` is expected to be valid for the lifetime of this handle,
/// assuming GStreamer's CUDA allocator follows the standard pool contract.
/// The underlying `GstBuffer` is pinned by `_gst_buffer` to prevent pool
/// reclamation.  See the module-level docs for caveats.
///
/// # Stream synchronization
///
/// **This library does not guarantee GPU execution has completed.**
/// GStreamer typically synchronizes before the appsink callback, but
/// callers launching work on a different CUDA stream must synchronize
/// explicitly (e.g., `cuStreamSynchronize`).  See the module-level docs.
///
/// # Example
///
/// ```rust,ignore
/// let handle = frame
///     .accelerated_handle::<CudaBufferHandle>()
///     .expect("frame should be CUDA-resident");
///
/// // Pass to TensorRT, cuBLAS, or any CUDA kernel.
/// launch_kernel(handle.device_ptr, handle.width, handle.height);
/// ```
pub struct CudaBufferHandle {
    /// Raw CUDA device pointer (`CUdeviceptr`) to the frame's pixel
    /// data on the GPU.
    pub device_ptr: u64,
    /// Byte length of the device allocation.
    pub len: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Row stride in bytes on the device.
    pub stride: u32,
    /// Pixel format of the device buffer.
    pub format: PixelFormat,
    /// The GStreamer buffer keeping the CUDA allocation alive. Dropping
    /// this releases the buffer back to GStreamer's CUDA buffer pool.
    _gst_buffer: gstreamer::Buffer,
}

// SAFETY: The CUDA device pointer is a numeric address into GPU memory
// that does not alias any host data. The GStreamer buffer handle is
// internally ref-counted and safe to send across threads.
unsafe impl Send for CudaBufferHandle {}
unsafe impl Sync for CudaBufferHandle {}

// ---------------------------------------------------------------------------
// Device-pointer extraction (FFI)
// ---------------------------------------------------------------------------

/// Map the buffer with `GST_MAP_READ | GST_MAP_CUDA` and extract the
/// CUDA device pointer from `GstMapInfo.data`.
///
/// The mapping is released immediately — we only need the pointer value.
/// The pointer remains valid as long as the owning `gstreamer::Buffer`
/// is alive (guaranteed by `CudaBufferHandle._gst_buffer`).  See the
/// module-level "Device pointer lifetime" section for the invariant.
///
/// # Safety contract
///
/// * `buffer` must contain CUDA-resident memory (caller verifies via
///   allocator name check before calling this function).
/// * GStreamer's CUDA buffer pool guarantees the device pointer is
///   stable for the buffer's lifetime (not just the map's lifetime).
fn extract_cuda_device_ptr(buffer: &gstreamer::BufferRef) -> Result<(u64, usize), MediaError> {
    #[allow(unused_imports)] // Used by gst_buffer_map's buffer.as_ptr()
    use gstreamer::glib::translate::ToGlibPtr;

    unsafe {
        let mut map_info = MaybeUninit::<gstreamer::ffi::GstMapInfo>::zeroed();
        let flags = gstreamer::ffi::GST_MAP_READ | GST_MAP_CUDA;

        // gst_buffer_map with GST_MAP_CUDA makes the allocator place
        // the CUdeviceptr into map_info.data.
        let ok =
            gstreamer::ffi::gst_buffer_map(buffer.as_ptr() as *mut _, map_info.as_mut_ptr(), flags);

        if ok == gstreamer::glib::ffi::GFALSE {
            return Err(MediaError::DecodeFailed {
                detail: "gst_buffer_map with GST_MAP_CUDA failed — \
                         CUDA allocator may not be available"
                    .into(),
            });
        }

        let info = map_info.assume_init();
        let device_ptr = info.data as u64;
        let size = info.size;

        gstreamer::ffi::gst_buffer_unmap(buffer.as_ptr() as *mut _, map_info.as_mut_ptr());

        Ok((device_ptr, size))
    }
}

// ---------------------------------------------------------------------------
// Pointer validation
// ---------------------------------------------------------------------------

/// Minimum plausible device-memory size for a frame (e.g. 1×1 single-channel).
const MIN_DEVICE_BUFFER_SIZE: usize = 1;

/// Reject clearly invalid device pointers before they reach downstream code.
///
/// This catches:
/// - null / zero pointer (GStreamer returned a CPU-side failure or
///   the allocator doesn't understand `GST_MAP_CUDA`)
/// - zero-length mapping (buffer is empty)
/// - size too small for the declared geometry (truncated allocation)
fn validate_device_ptr(ptr: u64, len: usize, width: u32, height: u32) -> Result<(), MediaError> {
    if ptr == 0 {
        return Err(MediaError::DecodeFailed {
            detail: "CUDA device pointer is null — the GStreamer CUDA allocator \
                     may not have handled GST_MAP_CUDA correctly, or the buffer \
                     does not contain device memory"
                .into(),
        });
    }
    if len < MIN_DEVICE_BUFFER_SIZE {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "CUDA device buffer size is {len} bytes — expected at least \
                 {MIN_DEVICE_BUFFER_SIZE} byte(s) for a mapped frame"
            ),
        });
    }
    // Minimum plausible: 1 byte per pixel × width × height.
    let min_plausible = (width as usize).saturating_mul(height as usize);
    if min_plausible > 0 && len < min_plausible {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "CUDA device buffer too small for declared geometry: \
                 buffer is {len} bytes but {width}×{height} requires at \
                 least {min_plausible} bytes (1 byte/pixel minimum)"
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Bridge function
// ---------------------------------------------------------------------------

/// Bridge a GStreamer CUDA-memory sample into a device-resident [`FrameEnvelope`].
///
/// 1. Parses caps to obtain frame geometry.
/// 2. Verifies the buffer carries `GstCudaMemory` (allocator name check).
/// 3. Extracts the CUDA device pointer via [`extract_cuda_device_ptr`].
/// 4. Wraps everything in a [`CudaBufferHandle`] and returns a
///    [`FrameEnvelope`] with a [`HostMaterializeFn`] for transparent
///    device→host fallback.
pub(crate) fn bridge_gst_sample_device(
    feed_id: FeedId,
    seq: &Arc<AtomicU64>,
    output_format: OutputFormat,
    sample: &gstreamer::Sample,
    ptz: Option<PtzTelemetry>,
) -> Result<FrameEnvelope, MediaError> {
    let caps = sample.caps().ok_or_else(|| MediaError::DecodeFailed {
        detail: "sample has no caps".into(),
    })?;

    let video_info =
        gstreamer_video::VideoInfo::from_caps(caps).map_err(|e| MediaError::DecodeFailed {
            detail: format!("failed to parse VideoInfo from caps: {e}"),
        })?;

    let width = video_info.width();
    let height = video_info.height();
    let stride = video_info.stride()[0] as u32;
    let format = output_format.to_pixel_format();

    let buffer = sample
        .buffer_owned()
        .ok_or_else(|| MediaError::DecodeFailed {
            detail: "sample has no buffer".into(),
        })?;

    let pts_ns = buffer.pts().map(|pts| pts.nseconds()).unwrap_or(0);
    let ts = MonotonicTs::from_nanos(pts_ns);
    let wall_ts = WallTs::now();
    let frame_seq = seq.fetch_add(1, Ordering::Relaxed);

    // ── Verify CUDA memory ───────────────────────────────────────────
    // Use gst_memory_is_type() — the GStreamer-canonical way to check
    // what kind of memory a buffer block carries.  This checks the
    // allocator's `mem_type` field, which is set at allocation time and
    // is reliable regardless of the allocator's GObject instance name.
    let memory = buffer.memory(0).ok_or_else(|| MediaError::DecodeFailed {
        detail: "buffer has no memory block".into(),
    })?;
    if !memory.is_type(GST_CUDA_MEMORY_TYPE_NAME) {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "buffer memory is not CUDA device memory (expected type \
                 \"{GST_CUDA_MEMORY_TYPE_NAME}\") — is the cudaupload \
                 element present and working?"
            ),
        });
    }

    // ── Extract device pointer ───────────────────────────────────────
    let (device_ptr, len) = extract_cuda_device_ptr(&buffer)?;
    validate_device_ptr(device_ptr, len, width, height)?;

    let handle = Arc::new(CudaBufferHandle {
        device_ptr,
        len,
        width,
        height,
        stride,
        format,
        _gst_buffer: buffer.clone(),
    });

    // ── Host materializer ────────────────────────────────────────────
    // Mapping a GStreamer CUDA buffer in plain GST_MAP_READ mode
    // triggers the CUDA allocator's device→host copy transparently.
    //
    // `HostMaterializeFn` is `Box<dyn Fn()>` — it must be callable
    // multiple times. We use `map_readable()` on the `BufferRef`
    // (borrow-based, does not consume the buffer) so the same buffer
    // can be re-mapped on repeated calls.  In practice the result is
    // cached by `FrameEnvelope`'s `OnceLock`, so this runs at most once.
    let mat_buffer = buffer.clone();
    let materialize: HostMaterializeFn = Box::new(move || {
        let map = mat_buffer.map_readable().map_err(|_| {
            nv_frame::FrameAccessError::MaterializationFailed {
                detail: "failed to map CUDA buffer to host".into(),
            }
        })?;
        Ok(HostBytes::from_vec(map.as_slice().to_vec()))
    });

    let mut metadata = TypedMetadata::new();
    if let Some(telemetry) = ptz {
        metadata.insert(telemetry);
    }

    Ok(FrameEnvelope::new_device(
        feed_id,
        frame_seq,
        ts,
        wall_ts,
        width,
        height,
        format,
        stride,
        handle,
        Some(materialize),
        metadata,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gst_map_cuda_flag_value() {
        // GST_MAP_FLAG_LAST = 1 << 16 = 65536.
        // GST_MAP_CUDA = GST_MAP_FLAG_LAST << 1 = 1 << 17 = 131072.
        assert_eq!(gstreamer::ffi::GST_MAP_FLAG_LAST, 1 << 16);
        assert_eq!(GST_MAP_CUDA, 1 << 17);
        assert_eq!(GST_MAP_CUDA, 131_072);
    }

    #[test]
    fn cuda_buffer_handle_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaBufferHandle>();
    }

    #[test]
    fn cuda_memory_type_name_matches_gstreamer_convention() {
        // gst-plugins-bad defines GST_CUDA_MEMORY_TYPE_NAME = "cuda-memory".
        // This test guards against accidental constant drift.
        assert_eq!(GST_CUDA_MEMORY_TYPE_NAME, "cuda-memory");
    }

    // -- validate_device_ptr tests ----------------------------------------

    #[test]
    fn validate_rejects_null_pointer() {
        let err = validate_device_ptr(0, 1024, 32, 32).unwrap_err();
        match err {
            MediaError::DecodeFailed { detail } => {
                assert!(detail.contains("null"), "expected 'null' in: {detail}");
            }
            other => panic!("expected DecodeFailed, got {other}"),
        }
    }

    #[test]
    fn validate_rejects_zero_length() {
        let err = validate_device_ptr(0xDEAD_BEEF, 0, 32, 32).unwrap_err();
        match err {
            MediaError::DecodeFailed { detail } => {
                assert!(
                    detail.contains("0 bytes"),
                    "expected size mention in: {detail}"
                );
            }
            other => panic!("expected DecodeFailed, got {other}"),
        }
    }

    #[test]
    fn validate_rejects_undersized_buffer() {
        // 640×480 = 307_200 minimum bytes, but we supply only 1000.
        let err = validate_device_ptr(0xDEAD_BEEF, 1000, 640, 480).unwrap_err();
        match err {
            MediaError::DecodeFailed { detail } => {
                assert!(
                    detail.contains("too small"),
                    "expected 'too small' in: {detail}"
                );
                assert!(detail.contains("640"), "should mention width: {detail}");
                assert!(detail.contains("480"), "should mention height: {detail}");
            }
            other => panic!("expected DecodeFailed, got {other}"),
        }
    }

    #[test]
    fn validate_accepts_valid_pointer() {
        // 640×480 RGB = 921600 bytes; supply exactly that.
        validate_device_ptr(0x1000_0000, 921_600, 640, 480).unwrap();
    }

    #[test]
    fn validate_accepts_larger_than_minimum() {
        // stride padding makes buffer larger than w×h — should pass.
        validate_device_ptr(0x1000_0000, 1_000_000, 640, 480).unwrap();
    }

    #[test]
    fn validate_accepts_1x1_frame() {
        validate_device_ptr(0x1000_0000, 1, 1, 1).unwrap();
    }
}
