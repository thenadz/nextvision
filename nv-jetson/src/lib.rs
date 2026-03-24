//! # nv-jetson — JetPack 5.x GPU residency provider
//!
//! This crate provides [`NvmmProvider`], an implementation of
//! [`GpuPipelineProvider`](nv_media::GpuPipelineProvider) for NVIDIA Jetson
//! platforms running **JetPack 5.x** (GStreamer 1.16, L4T R35).
//!
//! On JetPack 5.x the upstream GStreamer CUDA elements (`cudaupload`,
//! `cudaconvert`) are not available. Instead, the hardware decoder
//! (`nvv4l2decoder`) outputs frames in **NVMM** (NVIDIA Multimedia Memory)
//! — a DMA-capable, GPU-accessible memory type backed by `NvBufSurface`.
//!
//! `NvmmProvider` builds a pipeline tail that keeps frames in NVMM,
//! optionally inserting `nvvidconv` for colour-space conversion, and
//! bridges the resulting `GstSample` into a device-resident
//! [`FrameEnvelope`](nv_frame::FrameEnvelope) via the `NvBufSurface` FFI.
//!
//! # Platform requirement
//!
//! This crate requires **Linux** (uses DMA-buf FDs and Unix I/O) and
//! **NVIDIA Jetson** (links against `libnvbufsurface.so` from the L4T BSP).
//!
//! - On **non-Linux** platforms, the crate emits a compile error.
//! - On **non-Jetson Linux** hosts, `cargo check` succeeds but the
//!   final binary link will fail (missing `libnvbufsurface.so`).
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use nv_jetson::NvmmProvider;
//! use nv_runtime::{DeviceResidency, FeedConfig};
//!
//! let provider = Arc::new(NvmmProvider::new());
//!
//! let config = FeedConfig::builder()
//!     .device_residency(DeviceResidency::Provider(provider))
//!     // ... other config ...
//!     .build()?;
//! ```

#[cfg(not(target_os = "linux"))]
compile_error!(
    "nv-jetson requires Linux — it uses DMA-buf file descriptors and \
     links against libnvbufsurface.so from the NVIDIA L4T BSP."
);

mod ffi;
mod nvmm;

pub use nvmm::{NvmmBufferHandle, NvmmProvider};
