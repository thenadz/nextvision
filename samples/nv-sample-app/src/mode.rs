//! Device residency, post-decode hook, and NVMM preprocessing for the sample app.
//!
//! Extracted into a module so the mode-selection logic is unit-testable
//! without requiring a running GStreamer pipeline or Jetson hardware.
//!
//! When the `jetson-nvmm` feature is active, this module also provides
//! `NvmmBatchPreprocessor` and `NvmmPreprocessor` — device-native
//! preprocessors that read NVMM frames directly from unified memory,
//! bypassing the expensive GPU→CPU host materialization path. This
//! eliminates the primary cause of NVMM buffer pool starvation on
//! Jetson when the hardware decoder's output surface pool is small.

use nv_runtime::DeviceResidency;

/// Determine the [`DeviceResidency`] based on the `--gpu` flag and
/// compile-time feature flags.
///
/// | Feature        | `--gpu=true`                    | `--gpu=false` |
/// |----------------|---------------------------------|---------------|
/// | `jetson-nvmm`  | `Provider(NvmmProvider)`        | `Host`        |
/// | (default)      | `Cuda`                          | `Host`        |
pub fn select_device_residency(gpu: bool) -> DeviceResidency {
    if !gpu {
        return DeviceResidency::Host;
    }

    #[cfg(feature = "jetson-nvmm")]
    {
        let provider = std::sync::Arc::new(nv_jetson::NvmmProvider::new());
        DeviceResidency::Provider(provider)
    }

    #[cfg(not(feature = "jetson-nvmm"))]
    {
        DeviceResidency::Cuda
    }
}

/// Build the NVMM post-decode hook.
///
/// Returns a hook that inserts `nvvidconv` when the decoder outputs
/// `memory:NVMM` buffers.  This is needed on Jetson where hardware
/// decoders output NVMM-mapped buffers that `videoconvert` cannot
/// accept.
///
/// The hook is safe to install unconditionally:
/// - When a Provider is active, the pipeline builder skips all hooks.
/// - On non-Jetson platforms, no decoder produces NVMM output, so the
///   hook never fires.
pub fn nvmm_bridge_hook() -> nv_runtime::PostDecodeHook {
    std::sync::Arc::new(|info: &nv_runtime::DecodedStreamInfo| {
        if info.memory_type.as_deref() == Some("NVMM") {
            Some("nvvidconv".into())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_false_selects_host() {
        let residency = select_device_residency(false);
        assert!(matches!(residency, DeviceResidency::Host));
        assert!(!residency.is_device());
    }

    #[test]
    fn gpu_true_selects_device() {
        let residency = select_device_residency(true);
        // Without jetson-nvmm feature: Cuda.
        // With jetson-nvmm feature: Provider.
        assert!(residency.is_device());
    }

    #[cfg(not(feature = "jetson-nvmm"))]
    #[test]
    fn gpu_true_without_jetson_selects_cuda() {
        let residency = select_device_residency(true);
        assert!(matches!(residency, DeviceResidency::Cuda));
    }

    #[cfg(feature = "jetson-nvmm")]
    #[test]
    fn gpu_true_with_jetson_selects_provider() {
        let residency = select_device_residency(true);
        assert!(matches!(residency, DeviceResidency::Provider(_)));
    }

    #[test]
    fn nvmm_hook_fires_for_nvmm_memory() {
        let hook = nvmm_bridge_hook();
        let info = nv_runtime::DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: Some("NVMM".into()),
            format: Some("NV12".into()),
        };
        assert_eq!(hook(&info), Some("nvvidconv".into()));
    }

    #[test]
    fn nvmm_hook_noop_for_system_memory() {
        let hook = nvmm_bridge_hook();
        let info = nv_runtime::DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: None,
            format: Some("I420".into()),
        };
        assert_eq!(hook(&info), None);
    }

    #[test]
    fn nvmm_hook_noop_for_cuda_memory() {
        let hook = nvmm_bridge_hook();
        let info = nv_runtime::DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: Some("CUDAMemory".into()),
            format: Some("RGBA".into()),
        };
        assert_eq!(hook(&info), None);
    }
}

// ---------------------------------------------------------------------------
// NVMM-native preprocessors (Jetson only)
// ---------------------------------------------------------------------------

#[cfg(feature = "jetson-nvmm")]
pub use nvmm_preprocess::{NvmmBatchPreprocessor, NvmmPreprocessor};

#[cfg(feature = "jetson-nvmm")]
mod nvmm_preprocess {
    use nv_core::error::StageError;
    use nv_core::id::StageId;
    use nv_frame::{FrameEnvelope, PixelFormat};
    use nv_jetson::NvmmBufferHandle;
    use nv_sample_detection::{
        BatchFramePreprocessor, FramePreprocessor, HostBatchPreprocessor, HostFallbackPolicy,
        HostPreprocessor, LetterboxInfo, PreprocessedFrame, letterbox_preprocess_into,
    };

    /// Resolve bytes-per-pixel for NVMM-supported formats.
    ///
    /// `Bgr8` is rejected because `letterbox_preprocess_into` assumes
    /// RGB channel order (channel 0 = R, 1 = G, 2 = B).  Processing
    /// BGR data through this path would silently swap R↔B, degrading
    /// inference quality.  In practice, NVMM buffers arrive as `Rgba8`
    /// (3-byte formats are promoted by `nvmm_effective_format`), so
    /// this guard is defensive.
    fn nvmm_bpp(format: PixelFormat, stage_id: StageId) -> Result<u32, StageError> {
        match format {
            PixelFormat::Rgba8 => Ok(4),
            PixelFormat::Rgb8 => Ok(3),
            PixelFormat::Bgr8 => Err(StageError::ProcessingFailed {
                stage_id,
                detail: "Bgr8 is not supported in the NVMM preprocessing path: \
                         letterbox assumes RGB channel order; BGR would silently \
                         swap R↔B and degrade inference quality"
                    .into(),
            }),
            other => Err(StageError::ProcessingFailed {
                stage_id,
                detail: format!("unsupported NVMM pixel format for preprocessing: {other:?}"),
            }),
        }
    }

    /// Read NVMM pixels directly from unified memory.
    ///
    /// # Safety contract
    ///
    /// `NvmmBufferHandle::data_ptr` is a CPU-accessible address obtained
    /// via `NvBufSurfaceMap` during `bridge_sample()`.  The mapping is
    /// kept alive for the lifetime of the `NvmmBufferHandle` (unmapped
    /// in its `Drop` impl), so `data_ptr` remains valid as long as the
    /// `Arc<NvmmBufferHandle>` inside the `FrameEnvelope` is live.
    ///
    /// The returned slice covers `stride × height` bytes, bounded by the
    /// NVMM allocation size when reported (`handle.len > 0`).
    fn nvmm_pixels<'a>(
        handle: &'a NvmmBufferHandle,
        stage_id: StageId,
    ) -> Result<&'a [u8], StageError> {
        let access_bytes = handle.stride as usize * handle.height as usize;
        if handle.len > 0 && access_bytes > handle.len {
            return Err(StageError::ProcessingFailed {
                stage_id,
                detail: format!(
                    "NVMM buffer too small for direct access: \
                     need {access_bytes} bytes (stride {} × height {}), \
                     allocation reports {} bytes",
                    handle.stride, handle.height, handle.len,
                ),
            });
        }

        // SAFETY: see doc comment above.
        Ok(unsafe { std::slice::from_raw_parts(handle.data_ptr as *const u8, access_bytes) })
    }

    // ---------------------------------------------------------------
    // Batch preprocessor
    // ---------------------------------------------------------------

    /// NVMM-aware batch preprocessor for Jetson (JetPack 5.x).
    ///
    /// For NVMM device-resident frames, reads pixel data directly from
    /// the unified memory pointer in [`NvmmBufferHandle`], performing
    /// letterbox resize without any host materialization.  This avoids
    /// the GPU→CPU DMA copy that causes NVMM buffer pool starvation
    /// when the hardware decoder's output surface pool is small.
    ///
    /// Non-NVMM frames (e.g., host-resident test data) fall through
    /// to the default [`HostBatchPreprocessor`].
    pub struct NvmmBatchPreprocessor {
        stage_id: StageId,
        host_fallback: HostBatchPreprocessor,
    }

    impl NvmmBatchPreprocessor {
        pub fn new(stage_id: StageId, policy: HostFallbackPolicy) -> Self {
            Self {
                stage_id,
                host_fallback: HostBatchPreprocessor::with_policy(stage_id, policy),
            }
        }
    }

    impl BatchFramePreprocessor for NvmmBatchPreprocessor {
        fn preprocess_into(
            &self,
            frame: &FrameEnvelope,
            input_size: u32,
            dest: &mut [f32],
        ) -> Result<LetterboxInfo, StageError> {
            let Some(handle) = frame.accelerated_handle::<NvmmBufferHandle>() else {
                return self.host_fallback.preprocess_into(frame, input_size, dest);
            };

            let bpp = nvmm_bpp(handle.format, self.stage_id)?;
            let pixels = nvmm_pixels(handle, self.stage_id)?;

            Ok(letterbox_preprocess_into(
                pixels,
                handle.width,
                handle.height,
                handle.stride,
                bpp,
                input_size,
                dest,
            ))
        }
    }

    // ---------------------------------------------------------------
    // Single-frame preprocessor
    // ---------------------------------------------------------------

    /// NVMM-aware single-frame preprocessor for Jetson (JetPack 5.x).
    ///
    /// Same zero-copy strategy as [`NvmmBatchPreprocessor`], but returns
    /// a standalone [`PreprocessedFrame`] for per-feed
    /// [`DetectorStage`](nv_sample_detection::DetectorStage) use.
    pub struct NvmmPreprocessor {
        stage_id: StageId,
        host_fallback: HostPreprocessor,
    }

    impl NvmmPreprocessor {
        pub fn new(stage_id: StageId, policy: HostFallbackPolicy) -> Self {
            Self {
                stage_id,
                host_fallback: HostPreprocessor::with_policy(stage_id, policy),
            }
        }
    }

    impl FramePreprocessor for NvmmPreprocessor {
        fn preprocess(
            &self,
            frame: &FrameEnvelope,
            input_size: u32,
        ) -> Result<PreprocessedFrame, StageError> {
            let Some(handle) = frame.accelerated_handle::<NvmmBufferHandle>() else {
                return self.host_fallback.preprocess(frame, input_size);
            };

            let bpp = nvmm_bpp(handle.format, self.stage_id)?;
            let pixels = nvmm_pixels(handle, self.stage_id)?;

            let t = input_size as usize;
            let pixel_count = t * t;
            let mut tensor = vec![114.0_f32 / 255.0; pixel_count * 3];

            let letterbox = letterbox_preprocess_into(
                pixels,
                handle.width,
                handle.height,
                handle.stride,
                bpp,
                input_size,
                &mut tensor,
            );

            Ok(PreprocessedFrame { tensor, letterbox })
        }
    }
}
