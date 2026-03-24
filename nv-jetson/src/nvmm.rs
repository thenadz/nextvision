//! NVMM-based GPU pipeline provider for JetPack 5.x.
//!
//! [`NvmmProvider`] implements
//! [`GpuPipelineProvider`](nv_media::GpuPipelineProvider) using NVIDIA's
//! NVMM (NvBufSurface) memory model, which is the native GPU memory
//! path on JetPack 5.x where upstream GStreamer CUDA elements are not
//! available.
//!
//! # Pipeline topology
//!
//! ```text
//! decode (nvv4l2decoder, NV12/NVMM) → nvvidconv (NVMM→NVMM:RGBA) → appsink(memory:NVMM,RGBA)
//! ```
//!
//! `nvvidconv` performs GPU-accelerated colour-space conversion while
//! keeping frames in NVMM.  The appsink receives RGBA (or RGB) frames
//! in NVMM, which the bridge function converts to a device-resident
//! [`FrameEnvelope`] via the `NvBufSurface` FFI.
//!
//! # Host fallback
//!
//! Every device frame carries a [`HostMaterializeFn`] that maps the
//! NVMM buffer to host memory via `NvBufSurfaceMap`, copies the data,
//! and unmaps.  CPU-only stages transparently fall back without extra
//! configuration.

use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use nv_core::error::MediaError;
use nv_core::id::FeedId;
use nv_frame::{FrameEnvelope, HostBytes, HostMaterializeFn, PixelFormat};

use nv_media::PtzTelemetry;
use nv_media::gpu_provider::{GpuPipelineProvider, GpuPipelineTail};

use crate::ffi;

// ---------------------------------------------------------------------------
// NvmmBufferHandle
// ---------------------------------------------------------------------------

/// Opaque handle to an NVMM device buffer extracted from a GStreamer sample.
///
/// Downstream CUDA/TensorRT stages recover this via
/// [`FrameEnvelope::accelerated_handle::<NvmmBufferHandle>()`](nv_frame::FrameEnvelope::accelerated_handle).
///
/// # Lifetime
///
/// `data_ptr` is valid for the lifetime of this handle.  The underlying
/// GStreamer buffer (and therefore the NVMM allocation) is pinned by
/// `_gst_buffer`.
///
/// # Unified memory
///
/// On Xavier AGX (JetPack 5.x), NVMM allocations use unified memory —
/// the same pointer is accessible from both CPU and GPU contexts.
/// However, explicit mapping via `NvBufSurfaceMap` is still required
/// for portable correctness.
pub struct NvmmBufferHandle {
    /// GPU-accessible pointer to the frame's pixel data.
    ///
    /// On Xavier's unified memory, this pointer is also CPU-accessible
    /// after `NvBufSurfaceMap`, but should be treated as a device
    /// pointer for downstream CUDA kernel launches.
    pub data_ptr: u64,
    /// Byte length of the pixel data.
    pub len: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Row stride in bytes.
    pub stride: u32,
    /// Pixel format of the buffer (matches the appsink caps negotiation).
    pub format: PixelFormat,
    /// DMA-buf file descriptor for the NVMM allocation.
    ///
    /// Exposed for advanced integrations that need to import the buffer
    /// into EGL/Vulkan without going through NvBufSurface.
    pub dmabuf_fd: i32,
    /// Pin the GStreamer buffer to keep the NVMM allocation alive.
    _gst_buffer: gstreamer::Buffer,
}

// SAFETY: The data pointer is a numeric GPU address (unified memory on
// Xavier) that does not alias host data.  The GStreamer buffer is
// internally ref-counted and safe to send across threads.
unsafe impl Send for NvmmBufferHandle {}
unsafe impl Sync for NvmmBufferHandle {}

// ---------------------------------------------------------------------------
// NvmmProvider
// ---------------------------------------------------------------------------

/// GPU pipeline provider for JetPack 5.x NVMM memory.
///
/// Plug this into
/// [`FeedConfigBuilder::device_residency()`] as
/// `DeviceResidency::Provider(Arc::new(NvmmProvider::new()))` to enable
/// NVMM-based GPU residency on Jetson Xavier (JetPack 5.x).
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use nv_jetson::NvmmProvider;
/// use nv_runtime::{DeviceResidency, FeedConfig};
///
/// let config = FeedConfig::builder()
///     .device_residency(DeviceResidency::Provider(Arc::new(NvmmProvider::new())))
///     // ...
///     .build()?;
/// ```
pub struct NvmmProvider {
    _private: (),
}

impl NvmmProvider {
    /// Create a new NVMM provider.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for NvmmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuPipelineProvider for NvmmProvider {
    fn name(&self) -> &str {
        "nvmm-jetson5"
    }

    fn build_pipeline_tail(
        &self,
        pixel_format: PixelFormat,
    ) -> Result<GpuPipelineTail, MediaError> {
        let effective = nvmm_effective_format(pixel_format);
        if effective != pixel_format {
            tracing::info!(
                requested = ?pixel_format,
                effective = ?effective,
                "NVMM format upgrade: 3-byte format not supported in NVMM, \
                 using 4-byte equivalent",
            );
        }
        let gst_format = pixel_format_to_gst(effective);

        // Create nvvidconv for NVMM→NVMM colour-space conversion.
        let nvvidconv = gstreamer::ElementFactory::make("nvvidconv")
            .build()
            .map_err(|_| MediaError::Unsupported {
                detail: "nvvidconv element not found — is this a JetPack 5.x system \
                         with gstreamer1.0-plugins-bad installed?"
                    .into(),
            })?;

        // Capsfilter to constrain nvvidconv output to NVMM + target format.
        let caps_str = format!("video/x-raw(memory:NVMM),format={gst_format}");
        let caps = gstreamer::Caps::from_str(&caps_str).map_err(|e| {
            MediaError::DecodeFailed {
                detail: format!("invalid NVMM caps string '{caps_str}': {e}"),
            }
        })?;

        let capsfilter =
            gstreamer::ElementFactory::make("capsfilter")
                .property("caps", &caps)
                .build()
                .map_err(|_| MediaError::Unsupported {
                    detail: "capsfilter element not found".into(),
                })?;

        // Appsink with NVMM caps.
        let appsink = gstreamer_app::AppSink::builder()
            .caps(&caps)
            .max_buffers(1)
            .drop(true)
            .sync(false)
            .build();

        tracing::debug!(
            format = gst_format,
            "NVMM pipeline tail: nvvidconv → capsfilter → appsink",
        );

        Ok(GpuPipelineTail {
            elements: vec![nvvidconv, capsfilter],
            appsink,
        })
    }

    fn bridge_sample(
        &self,
        feed_id: FeedId,
        seq: &Arc<AtomicU64>,
        pixel_format: PixelFormat,
        sample: &gstreamer::Sample,
        ptz: Option<PtzTelemetry>,
    ) -> Result<FrameEnvelope, MediaError> {
        // Use the same effective format as build_pipeline_tail — the NVMM
        // buffer actually contains the upgraded format (e.g., RGBA, not RGB).
        let effective = nvmm_effective_format(pixel_format);

        let mut info = nv_media::SampleInfo::extract(sample, seq)?;

        // ── Extract NVMM device pointer via NvBufSurface FFI ─────────
        let (data_ptr, len, dmabuf_fd, nvmm_pitch) =
            extract_nvmm_pointer(&info.buffer, info.width, info.height)?;

        let handle = Arc::new(NvmmBufferHandle {
            data_ptr,
            len,
            width: info.width,
            height: info.height,
            stride: nvmm_pitch,
            format: effective,
            dmabuf_fd,
            _gst_buffer: info.buffer.clone(),
        });

        // ── Host materializer ────────────────────────────────────────
        // Maps the NVMM surface to CPU-accessible memory on demand.
        // Uses the actual NVMM pitch (from NvBufSurface) as source stride,
        // NOT GStreamer's VideoInfo stride which ignores GPU alignment.
        //
        // The dmabuf FD is cached in the closure to avoid repeating the
        // tier-1/2/3 FD extraction dance on every materialisation call.
        let mat_width = info.width;
        let mat_height = info.height;
        let mat_bpp = pixel_format_bpp(effective);
        let materialize: HostMaterializeFn = Box::new(move || {
            materialize_nvmm_to_host(
                dmabuf_fd,
                mat_width,
                mat_height,
                mat_bpp,
            )
        });

        // Frame stride = tightly-packed row width, since the materializer
        // strips GPU alignment padding when copying to host memory.
        let host_stride = info.width * pixel_format_bpp(effective);
        info.stride = host_stride;

        Ok(info.into_device_envelope(feed_id, effective, handle, Some(materialize), ptz))
    }


}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the effective pixel format for NVMM memory.
///
/// `nvvidconv` on JetPack 5.x only supports 4-byte-per-pixel formats
/// (RGBA, BGRx, NV12, I420 …) in NVMM memory.  3-byte formats (RGB,
/// BGR) are not supported because NVMM allocations rely on GPU-aligned
/// pitch that assumes power-of-two bytes-per-pixel.  When the caller
/// requests a 3-byte format, we transparently upgrade to RGBA — the
/// [`FrameEnvelope`] metadata will reflect the actual buffer format so
/// downstream stages can handle it correctly.
fn nvmm_effective_format(fmt: PixelFormat) -> PixelFormat {
    match fmt {
        // 3-byte formats are not representable in NVMM — promote to RGBA.
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => PixelFormat::Rgba8,
        other => other,
    }
}

/// Map a [`PixelFormat`] to the GStreamer caps format string.
fn pixel_format_to_gst(fmt: PixelFormat) -> &'static str {
    match fmt {
        PixelFormat::Rgb8 => "RGB",
        PixelFormat::Bgr8 => "BGR",
        PixelFormat::Rgba8 => "RGBA",
        _ => "RGBA", // Conservative fallback for uncommon formats.
    }
}

/// Bytes per pixel for a given format.
fn pixel_format_bpp(fmt: PixelFormat) -> u32 {
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => 4,
    }
}

/// Extract the DMA-buf FD from a GStreamer buffer using a tiered strategy.
///
/// # Strategy
///
/// 1. **GStreamer DmaBuf allocator** — try
///    [`downcast_memory_ref::<DmaBufMemory>()`] which calls
///    `gst_is_dmabuf_memory()` under the hood.  Works when the allocator
///    subclasses `GstDmaBufAllocator`.
///
/// 2. **GStreamer Fd allocator** — try
///    [`downcast_memory_ref::<FdMemory>()`].  Works when the allocator
///    subclasses `GstFdAllocator`.
///
/// 3. **NVMM NvBufSurface fallback** — on JetPack 5.x (GStreamer 1.16,
///    L4T R35), NVIDIA's custom NVMM allocator does **not** subclass
///    either standard GStreamer allocator type.  Mapping the GstBuffer
///    returns the `NvBufSurface` structure itself (the standard
///    NVIDIA/DeepStream convention).  The DMA-BUF file descriptor is
///    read from `surfaceList[0].bufferDesc`.
///
///    **Guards:**
///    - Allocator name must contain "nv" (case-insensitive), matching
///      known NVIDIA allocator names (`nvvideo`, `NvBuffer`, etc.).
///      Buffers from other allocators are rejected.
///    - The mapped data must be large enough for a `NvBufSurface` struct.
///    - The `surfaceList` pointer must be non-null and `batchSize > 0`.
///    - The extracted FD must be a plausible Linux file descriptor:
///      in range `[0, MAX_REASONABLE_FD]`.
///
///    This fallback is expected to be unnecessary on JetPack 6+ where
///    NVIDIA's allocator subclasses `GstDmaBufAllocator` directly.
fn extract_dmabuf_fd(buffer: &gstreamer::Buffer) -> Result<i32, MediaError> {
    use std::os::unix::io::RawFd;

    // Linux FD table size is typically limited to 1048576 (ulimit -n hard max).
    // Any FD value above this is almost certainly corrupted data, not a real FD.
    const MAX_REASONABLE_FD: i32 = 1_048_576;

    let n_memory = buffer.n_memory();
    if n_memory == 0 {
        return Err(MediaError::DecodeFailed {
            detail: "NVMM buffer has no GstMemory blocks".into(),
        });
    }

    let mem = buffer.peek_memory(0);

    // ── Tier 1: GStreamer DmaBuf allocator ────────────────────────────
    if let Some(dmabuf) =
        mem.downcast_memory_ref::<gstreamer_allocators::DmaBufMemory>()
    {
        let fd: RawFd = dmabuf.fd();
        if fd < 0 {
            return Err(MediaError::DecodeFailed {
                detail: format!(
                    "DmaBuf memory returned negative FD ({fd})"
                ),
            });
        }
        tracing::trace!(fd, "extracted DMA-buf FD via GstDmaBufMemory");
        return Ok(fd);
    }

    // ── Tier 2: GStreamer Fd allocator ────────────────────────────────
    if let Some(fdmem) =
        mem.downcast_memory_ref::<gstreamer_allocators::FdMemory>()
    {
        let fd: RawFd = fdmem.fd();
        if fd < 0 {
            return Err(MediaError::DecodeFailed {
                detail: format!(
                    "FdMemory returned negative FD ({fd})"
                ),
            });
        }
        tracing::trace!(fd, "extracted DMA-buf FD via GstFdMemory");
        return Ok(fd);
    }

    // ── Tier 3: NVMM NvBufSurface fallback ─────────────────────────
    //
    // Required on JetPack 5.x where nvv4l2decoder's allocator does not
    // subclass GstDmaBufAllocator or GstFdAllocator.  Gated on the
    // allocator name containing "nv" (case-insensitive) to reject
    // non-NVIDIA allocators that accidentally reach this path.
    //
    // On JetPack 5.x, mapping an NVMM GstBuffer returns a pointer to
    // the `NvBufSurface` structure itself (the standard NVIDIA/DeepStream
    // convention).  The DMA-BUF FD lives in `surfaceList[0].bufferDesc`.
    let allocator_name = mem
        .allocator()
        .map(|a| {
            use gstreamer::prelude::*;
            a.name().to_string()
        })
        .unwrap_or_default();

    let alloc_lower = allocator_name.to_lowercase();
    if !alloc_lower.contains("nv") {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "DMA-buf FD not extractable via GStreamer allocator APIs, and \
                 the allocator '{allocator_name}' is not an NVIDIA NVMM \
                 allocator (name does not contain 'nv') — cannot extract FD"
            ),
        });
    }

    let map = buffer.map_readable().map_err(|_| MediaError::DecodeFailed {
        detail: format!(
            "failed to map NVMM buffer readable (allocator='{allocator_name}') — \
             is this a valid NVMM allocation?"
        ),
    })?;

    // The mapped data IS the NvBufSurface struct on JetPack 5.x.
    if map.len() < std::mem::size_of::<ffi::NvBufSurface>() {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "NVMM buffer mapping too small ({} bytes) for NvBufSurface struct \
                 (need {} bytes); allocator='{allocator_name}'",
                map.len(),
                std::mem::size_of::<ffi::NvBufSurface>(),
            ),
        });
    }

    // SAFETY: On JetPack 5.x, the mapped data of an NVMM GstBuffer is
    // the NvBufSurface struct.  This is the standard access pattern used
    // by NVIDIA's DeepStream SDK.
    let surface_ptr = map.as_slice().as_ptr() as *const ffi::NvBufSurface;
    let surface = unsafe { &*surface_ptr };

    if surface.surface_list.is_null() {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "NvBufSurface.surface_list is null; allocator='{allocator_name}'"
            ),
        });
    }
    if surface.batch_size == 0 {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "NvBufSurface.batch_size is 0; allocator='{allocator_name}'"
            ),
        });
    }

    let dmabuf_fd = unsafe { (*surface.surface_list).buffer_desc } as i32;

    if dmabuf_fd < 0 || dmabuf_fd > MAX_REASONABLE_FD {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "NvBufSurface.bufferDesc produced implausible FD value \
                 ({dmabuf_fd}); expected range [0, {MAX_REASONABLE_FD}]. \
                 allocator='{allocator_name}', gpu_id={}, mem_type={}, \
                 batch_size={}",
                surface.gpu_id, surface.mem_type, surface.batch_size,
            ),
        });
    }

    tracing::debug!(
        fd = dmabuf_fd,
        allocator = %allocator_name,
        gpu_id = surface.gpu_id,
        mem_type = surface.mem_type,
        batch_size = surface.batch_size,
        "extracted DMA-buf FD from NvBufSurface.bufferDesc \
         (JetPack 5.x path — allocator does not subclass GstDmaBufAllocator)",
    );

    drop(map);
    Ok(dmabuf_fd)
}

/// Extract the NVMM device pointer from a GStreamer buffer.
///
/// Uses [`extract_dmabuf_fd`] to obtain the DMA-buf FD, then
/// `NvBufSurfaceFromFd` + `NvBufSurfaceMap` to read the device pointer.
///
/// # Pointer lifetime contract
///
/// After map → read `data_ptr` → unmap, `data_ptr` remains valid because:
///
/// 1. **Unified memory**: On Jetson Xavier (JetPack 5.x), NVMM
///    allocations use NVIDIA unified memory. The physical allocation
///    persists regardless of the CPU mapping state; `NvBufSurfaceMap`
///    only populates the `data_ptr` field — it does not make the memory
///    accessible (unified memory is always addressable).
///
/// 2. **DMA-buf pinning**: The underlying DMA-buf allocation lives as
///    long as the GStreamer buffer.  The [`NvmmBufferHandle::_gst_buffer`]
///    field pins the buffer for the lifetime of the handle, preventing
///    deallocation.
///
/// 3. **Runtime validation**: [`validate_nvmm_mem_type`] checks the
///    surface's `mem_type` and rejects memory types where the
///    post-unmap pointer validity invariant does not hold (e.g., pure
///    CUDA device memory that is not CPU-addressable).
///
/// # Safety
///
/// This function calls FFI into `libnvbufsurface.so`. The GStreamer
/// buffer must carry a valid NVMM allocation (as produced by
/// `nvv4l2decoder` or `nvvidconv`).
fn extract_nvmm_pointer(
    buffer: &gstreamer::Buffer,
    width: u32,
    height: u32,
) -> Result<(u64, usize, i32, u32), MediaError> {
    let dmabuf_fd = extract_dmabuf_fd(buffer)?;

    // ── NvBufSurface FFI ─────────────────────────────────────────────
    let mut surface_ptr: *mut ffi::NvBufSurface = std::ptr::null_mut();

    // SAFETY: `dmabuf_fd` was extracted from a valid NVMM GStreamer
    // buffer.  `NvBufSurfaceFromFd` populates `surface_ptr` with a
    // valid `NvBufSurface*` on success.
    let ret =
        unsafe { ffi::NvBufSurfaceFromFd(dmabuf_fd, &mut surface_ptr as *mut _) };

    if ret != 0 || surface_ptr.is_null() {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "NvBufSurfaceFromFd failed (fd={dmabuf_fd}, ret={ret}) — \
                 is this a valid NVMM allocation?"
            ),
        });
    }

    // Validate the memory type before relying on the unified-memory
    // post-unmap pointer validity invariant.
    let mem_type = unsafe { (*surface_ptr).mem_type };
    validate_nvmm_mem_type(mem_type)?;

    tracing::debug!(
        fd = dmabuf_fd,
        mem_type,
        "NvBufSurfaceFromFd succeeded — mapping surface for data pointer",
    );

    // Map the surface for read access to obtain the data pointer.
    //
    // Pass plane = -1 to map all planes at once.
    // SAFETY: `surface_ptr` is a valid NvBufSurface from the call above.
    let map_ret = unsafe {
        ffi::NvBufSurfaceMap(surface_ptr, 0, -1, 0 /* NVBUF_MAP_READ */)
    };

    if map_ret != 0 {
        return Err(MediaError::DecodeFailed {
            detail: "NvBufSurfaceMap failed — cannot access NVMM pixel data".into(),
        });
    }

    // Read the mapped address, surface parameters, and actual pitch.
    //
    // For NVBUF_MEM_SURFACE_ARRAY / NVBUF_MEM_HANDLE (the Jetson case),
    // `data_ptr` is NOT valid (per nvbufsurface.h).  After NvBufSurfaceMap,
    // the CPU-accessible address is in `mapped_addr.addr[0]`.  On Xavier
    // unified memory, this is also a valid GPU address.
    //
    // `plane_params.pitch[0]` is the actual row stride in the GPU buffer,
    // which may be larger than `width * bpp` due to GPU alignment padding.
    // GStreamer's `VideoInfo::stride()` does NOT reflect this padding —
    // using it to iterate rows produces corrupted output.
    //
    // SAFETY: After successful NvBufSurfaceMap, surface_list[0] is valid
    // and mapped_addr.addr[0] contains the mapped plane-0 pointer.
    let (data_ptr, data_size, nvmm_pitch) = unsafe {
        let params = &*(*surface_ptr).surface_list;
        let ptr = params.mapped_addr.addr[0] as u64;
        let size = params.data_size as usize;
        let pitch = params.plane_params.pitch[0];
        (ptr, size, pitch)
    };

    // Unmap. See doc comment above ("Pointer lifetime contract") for
    // why data_ptr remains valid after this call on unified memory.
    unsafe {
        ffi::NvBufSurfaceUnMap(surface_ptr, 0, -1);
    }

    // Validate the extracted pointer.
    if data_ptr == 0 {
        return Err(MediaError::DecodeFailed {
            detail: "NVMM data pointer is null after NvBufSurfaceMap".into(),
        });
    }

    tracing::debug!(
        data_ptr = format_args!("0x{data_ptr:x}"),
        data_size,
        dmabuf_fd,
        nvmm_pitch,
        width,
        height,
        "NVMM pointer extraction succeeded",
    );

    // Rough sanity check — at minimum, 1 byte per pixel expected.
    let min_size = (width as usize).saturating_mul(height as usize);
    if data_size > 0 && data_size < min_size {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "NVMM buffer too small: {data_size} bytes for {width}×{height} \
                 (minimum {min_size} bytes at 1 byte/pixel)"
            ),
        });
    }

    Ok((data_ptr, data_size, dmabuf_fd, nvmm_pitch))
}

// ---------------------------------------------------------------------------
// NVMM memory type validation
// ---------------------------------------------------------------------------

// NvBufSurface memory type constants from nvbufsurface.h.
const NVBUF_MEM_DEFAULT: u32 = 0;
const NVBUF_MEM_CUDA_PINNED: u32 = 1;
const NVBUF_MEM_CUDA_DEVICE: u32 = 2;
const NVBUF_MEM_CUDA_UNIFIED: u32 = 3;
const NVBUF_MEM_SURFACE_ARRAY: u32 = 4;

/// Validate that the NVMM surface's memory type supports the
/// map → read → unmap → use-pointer invariant.
///
/// On Jetson Xavier (JetPack 5.x), NVMM buffers from `nvv4l2decoder`
/// use `NVBUF_MEM_SURFACE_ARRAY` (4) or `NVBUF_MEM_DEFAULT` (0, which
/// resolves to surface-array on Jetson).  Both use unified memory where
/// the CPU-accessible address persists after `NvBufSurfaceUnMap`.
///
/// `NVBUF_MEM_CUDA_DEVICE` (2) is pure device memory — **not** CPU-
/// addressable after unmap — and is rejected to prevent silent data
/// corruption.
fn validate_nvmm_mem_type(mem_type: u32) -> Result<(), MediaError> {
    match mem_type {
        NVBUF_MEM_DEFAULT
        | NVBUF_MEM_CUDA_PINNED
        | NVBUF_MEM_CUDA_UNIFIED
        | NVBUF_MEM_SURFACE_ARRAY => Ok(()),
        NVBUF_MEM_CUDA_DEVICE => Err(MediaError::DecodeFailed {
            detail: format!(
                "NVMM surface uses CUDA device memory (mem_type={mem_type}) — \
                 address is not CPU-accessible after unmap. This memory type \
                 is not supported by the map-read-unmap extraction path."
            ),
        }),
        other => Err(MediaError::DecodeFailed {
            detail: format!(
                "NVMM surface reports unknown mem_type ({other}) — cannot \
                 verify unified-memory pointer validity invariant"
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// RAII guard for NvBufSurface map lifecycle
// ---------------------------------------------------------------------------

/// Drop guard that ensures `NvBufSurfaceUnMap` is called when the guard
/// goes out of scope — even on early returns or panics.
struct MappedSurface {
    surface_ptr: *mut ffi::NvBufSurface,
}

impl Drop for MappedSurface {
    fn drop(&mut self) {
        // SAFETY: surface_ptr was successfully mapped before this guard
        // was created. NvBufSurfaceUnMap is safe to call on a mapped
        // surface and is idempotent.  plane = -1 unmaps all planes.
        unsafe {
            ffi::NvBufSurfaceUnMap(self.surface_ptr, 0, -1);
        }
    }
}

/// Host materializer: maps NVMM buffer to CPU memory and copies the data.
///
/// Uses an RAII [`MappedSurface`] guard to ensure `NvBufSurfaceUnMap`
/// is called on all exit paths (including early validation failures).
///
/// Takes a pre-extracted DMA-buf FD to skip the per-frame
/// [`extract_dmabuf_fd`] call (which on JetPack 5.x requires mapping the
/// `GstBuffer`, reading the `NvBufSurface` struct, and extracting the FD
/// from `bufferDesc`).
///
/// When `src_stride == row_bytes` (no GPU alignment padding to strip),
/// uses a single bulk copy instead of per-row iteration.
fn materialize_nvmm_to_host(
    dmabuf_fd: i32,
    width: u32,
    height: u32,
    bpp: u32,
) -> Result<HostBytes, nv_frame::FrameAccessError> {
    let mut surface_ptr: *mut ffi::NvBufSurface = std::ptr::null_mut();
    let ret = unsafe { ffi::NvBufSurfaceFromFd(dmabuf_fd, &mut surface_ptr as *mut _) };
    if ret != 0 || surface_ptr.is_null() {
        return Err(nv_frame::FrameAccessError::MaterializationFailed {
            detail: format!("NvBufSurfaceFromFd failed during materialization (fd={dmabuf_fd})"),
        });
    }

    let mem_type = unsafe { (*surface_ptr).mem_type };
    validate_nvmm_mem_type(mem_type).map_err(|e| {
        nv_frame::FrameAccessError::MaterializationFailed {
            detail: format!("mem_type validation failed during materialization: {e}"),
        }
    })?;

    let map_ret = unsafe { ffi::NvBufSurfaceMap(surface_ptr, 0, -1, 0) };
    if map_ret != 0 {
        return Err(nv_frame::FrameAccessError::MaterializationFailed {
            detail: "NvBufSurfaceMap failed during host materialization".into(),
        });
    }

    let _guard = MappedSurface { surface_ptr };

    let result = unsafe {
        let params = &*(*surface_ptr).surface_list;
        let src_ptr = params.mapped_addr.addr[0] as *const u8;
        let data_size = params.data_size as usize;
        let nvmm_pitch = params.plane_params.pitch[0] as usize;

        if src_ptr.is_null() {
            Err(nv_frame::FrameAccessError::MaterializationFailed {
                detail: "NvBufSurface mapped_addr.addr[0] is null after Map".into(),
            })
        } else {
            let row_bytes = (width as usize).saturating_mul(bpp as usize);
            let src_stride = if nvmm_pitch > 0 { nvmm_pitch } else { row_bytes };

            if src_stride < row_bytes {
                Err(nv_frame::FrameAccessError::MaterializationFailed {
                    detail: format!(
                        "NVMM stride ({src_stride}) < row_bytes ({row_bytes})"
                    ),
                })
            } else {
                let total_read = (height as usize)
                    .saturating_sub(1)
                    .saturating_mul(src_stride)
                    .saturating_add(row_bytes);

                if data_size > 0 && total_read > data_size {
                    Err(nv_frame::FrameAccessError::MaterializationFailed {
                        detail: format!(
                            "NVMM buffer too small for row copy: need {total_read} \
                             bytes, surface reports {data_size}"
                        ),
                    })
                } else if src_stride == row_bytes {
                    // Fast path: no GPU padding — single bulk copy.
                    let total = row_bytes * height as usize;
                    let slice = std::slice::from_raw_parts(src_ptr, total);
                    Ok(slice.to_vec())
                } else {
                    // Slow path: strip per-row GPU alignment padding.
                    let mut data = Vec::with_capacity(row_bytes * height as usize);
                    for row in 0..height as usize {
                        let row_start = src_ptr.add(row * src_stride);
                        let slice = std::slice::from_raw_parts(row_start, row_bytes);
                        data.extend_from_slice(slice);
                    }
                    Ok(data)
                }
            }
        }
    };

    Ok(HostBytes::from_vec(result?))
}
