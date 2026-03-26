//! FFI bindings for `NvBufSurface` (libnvbufsurface).
//!
//! Struct layouts match the canonical header at
//! `/usr/src/jetson_multimedia_api/include/nvbufsurface.h`
//! on JetPack 5.x (L4T R35).
//!
//! Constants from the header:
//!   STRUCTURE_PADDING  = 4
//!   NVBUF_MAX_PLANES   = 4
//!
//! # Safety
//!
//! All functions in this module are `unsafe` FFI calls.  Callers must
//! ensure the file-descriptor (`fd`) passed to [`NvBufSurfaceFromFd`]
//! originates from a valid NVMM GStreamer buffer.

use std::os::raw::{c_int, c_uint, c_void};

/// NVBUF_MAX_PLANES from `nvbufsurface.h`.
const MAX_PLANES: usize = 4;

/// STRUCTURE_PADDING from `nvbufsurface.h`.
const STRUCT_PAD: usize = 4;

// ---------------------------------------------------------------------------
// NvBufSurfacePlaneParams
// ---------------------------------------------------------------------------

/// Plane-wise parameters of a buffer.
///
/// Layout on aarch64 (with 8-byte pointers):
///   num_planes      @ 0    (4)
///   width           @ 4    (16)
///   height          @ 20   (16)
///   pitch           @ 36   (16)
///   offset          @ 52   (16)
///   psize           @ 68   (16)
///   bytes_per_pix   @ 84   (16)
///   _pad            @ 100  (4)   ← repr(C) inserts for `void*` alignment
///   _reserved       @ 104  (128)
///   total           = 232
#[repr(C)]
pub struct NvBufSurfacePlaneParams {
    pub num_planes: c_uint,
    pub width: [c_uint; MAX_PLANES],
    pub height: [c_uint; MAX_PLANES],
    pub pitch: [c_uint; MAX_PLANES],
    pub offset: [c_uint; MAX_PLANES],
    pub psize: [c_uint; MAX_PLANES],
    pub bytes_per_pix: [c_uint; MAX_PLANES],
    _reserved: [*mut c_void; STRUCT_PAD * MAX_PLANES],
}

// ---------------------------------------------------------------------------
// NvBufSurfaceMappedAddr
// ---------------------------------------------------------------------------

/// Holds pointers to mapped buffer planes.
///
/// After `NvBufSurfaceMap`, `addr[plane]` contains the CPU-accessible
/// pointer for each mapped plane.
///
/// Layout on aarch64:
///   addr        @ 0   (32)
///   egl_image   @ 32  (8)
///   _reserved   @ 40  (32)
///   total       = 72
#[repr(C)]
pub struct NvBufSurfaceMappedAddr {
    /// Plane-wise pointers to CPU-mapped data.
    pub addr: [*mut c_void; MAX_PLANES],
    /// Pointer to a mapped EGLImage (may be null).
    pub egl_image: *mut c_void,
    _reserved: [*mut c_void; STRUCT_PAD],
}

// ---------------------------------------------------------------------------
// NvBufSurfaceParams
// ---------------------------------------------------------------------------

/// Per-surface parameters within an [`NvBufSurface`] batch.
///
/// Layout on aarch64:
///   width         @ 0    (4)
///   height        @ 4    (4)
///   pitch         @ 8    (4)
///   color_format  @ 12   (4)   ← NvBufSurfaceColorFormat enum
///   layout        @ 16   (4)   ← NvBufSurfaceLayout enum
///   _pad          @ 20   (4)   ← repr(C) for u64 alignment
///   buffer_desc   @ 24   (8)   ← DMA-buf FD (SURFACE_ARRAY / HANDLE only)
///   data_size     @ 32   (4)
///   _pad          @ 36   (4)   ← repr(C) for pointer alignment
///   data_ptr      @ 40   (8)   ← **Not valid** for SURFACE_ARRAY/HANDLE
///   plane_params  @ 48   (232)
///   mapped_addr   @ 280  (72)  ← addr[0] is the mapped pointer after Map
///   paramex       @ 352  (8)
///   _reserved     @ 360  (24)
///   total         = 384
#[repr(C)]
pub struct NvBufSurfaceParams {
    pub width: c_uint,
    pub height: c_uint,
    pub pitch: c_uint,
    pub color_format: c_uint,
    pub layout: c_uint,
    // repr(C) inserts 4 bytes padding here (u64 alignment).
    /// DMA-buf file descriptor.  Valid only for `NVBUF_MEM_SURFACE_ARRAY`
    /// and `NVBUF_MEM_HANDLE` memory types.
    pub buffer_desc: u64,
    /// Amount of allocated memory in bytes.
    pub data_size: c_uint,
    // repr(C) inserts 4 bytes padding here (pointer alignment).
    /// Pointer to allocated memory.
    ///
    /// **Not valid for `NVBUF_MEM_SURFACE_ARRAY` or `NVBUF_MEM_HANDLE`.**
    /// For those memory types, use `mapped_addr.addr[plane]` after
    /// `NvBufSurfaceMap` instead.
    pub data_ptr: *mut c_void,
    /// Plane-wise parameters.
    pub plane_params: NvBufSurfacePlaneParams,
    /// Mapped buffer addresses (populated by `NvBufSurfaceMap`).
    pub mapped_addr: NvBufSurfaceMappedAddr,
    /// Extended parameters (may be null).
    pub paramex: *mut c_void,
    _reserved: [*mut c_void; STRUCT_PAD - 1],
}

// ---------------------------------------------------------------------------
// NvBufSurface
// ---------------------------------------------------------------------------

/// Batched buffer descriptor returned by the NVMM allocator.
///
/// Layout on aarch64:
///   gpu_id         @ 0   (4)
///   batch_size     @ 4   (4)
///   num_filled     @ 8   (4)
///   is_contiguous  @ 12  (1)   ← C `bool`
///   _pad           @ 13  (3)   ← repr(C) for u32 alignment
///   mem_type       @ 16  (4)   ← NvBufSurfaceMemType enum
///   _pad           @ 20  (4)   ← repr(C) for pointer alignment
///   surface_list   @ 24  (8)
///   _reserved      @ 32  (32)
///   total          = 64
#[repr(C)]
pub struct NvBufSurface {
    /// GPU device id (typically 0 on single-GPU Jetsons).
    pub gpu_id: c_uint,
    /// Number of surfaces in this batch.
    pub batch_size: c_uint,
    /// Number of filled surfaces.
    pub num_filled: c_uint,
    /// Whether the batch memory is contiguous.
    ///
    /// C type is `bool` (1 byte).  We use `u8` to avoid Rust's
    /// validity constraint (`bool` must be 0 or 1) which FFI data
    /// from C may violate.
    pub is_contiguous: u8,
    /// Memory type (NvBufSurfaceMemType enum, fits in u32).
    pub mem_type: c_uint,
    /// Pointer to the array of per-surface parameters.
    pub surface_list: *mut NvBufSurfaceParams,
    _reserved: [*mut c_void; STRUCT_PAD],
}

// ---------------------------------------------------------------------------
// FFI functions
// ---------------------------------------------------------------------------

// Link against libnvbufsurface.so (available in the L4T BSP).
// This will fail at link-time on non-Jetson hosts, but cargo check
// and cargo test (with no integration tests) will succeed.
#[link(name = "nvbufsurface")]
unsafe extern "C" {
    /// Obtain an `NvBufSurface*` from a DMA-buf file descriptor.
    ///
    /// The returned pointer is valid for the lifetime of the underlying
    /// NVMM allocation (i.e., while the GStreamer buffer is alive).
    ///
    /// Returns 0 on success, -1 on failure.
    pub fn NvBufSurfaceFromFd(dmabuf_fd: c_int, surface: *mut *mut NvBufSurface) -> c_int;

    /// Map the surface to CPU-accessible virtual memory.
    ///
    /// After this call, `surface_list[index].mapped_addr.addr[plane]`
    /// contains the CPU-accessible pointer for the requested plane.
    ///
    /// For `NVBUF_MEM_SURFACE_ARRAY` and `NVBUF_MEM_HANDLE`, `data_ptr`
    /// is **not** valid — use `mapped_addr.addr[0]` instead.
    ///
    /// * `index` — surface index within the batch (usually 0).
    /// * `plane`  — plane index (0 for interleaved formats, -1 for all).
    /// * `type_`  — map type: 0 = read, 1 = write, 2 = read+write.
    ///
    /// Returns 0 on success.
    pub fn NvBufSurfaceMap(
        surface: *mut NvBufSurface,
        index: c_int,
        plane: c_int,
        type_: c_int,
    ) -> c_int;

    /// Unmap a previously mapped surface.
    pub fn NvBufSurfaceUnMap(surface: *mut NvBufSurface, index: c_int, plane: c_int) -> c_int;
}
