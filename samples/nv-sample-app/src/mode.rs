//! Device residency and post-decode hook selection for the sample app.
//!
//! Extracted into a module so the mode-selection logic is unit-testable
//! without requiring a running GStreamer pipeline or Jetson hardware.

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
