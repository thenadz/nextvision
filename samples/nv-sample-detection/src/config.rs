use std::path::PathBuf;

use crate::preprocess::HostFallbackPolicy;

/// Configuration for the sample detector stage.
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Model input size (width and height must match the ONNX graph).
    /// Default: 640.
    pub input_size: u32,
    /// Minimum confidence to keep a detection.
    /// Default: 0.25.
    pub confidence_threshold: f32,
    /// Optional class name table, indexed by `class_id`.
    /// Not stored on detections — provided for downstream consumers
    /// that want human-readable labels.
    pub class_names: Vec<String>,
    /// Run inference on GPU via the CUDA execution provider.
    ///
    /// When enabled, the session is configured with the CUDA EP.
    /// Falls back to CPU if CUDA is unavailable.
    /// Requires the `gpu` crate feature.
    pub gpu: bool,
    /// Use the CUDA EP only, skipping TensorRT.
    ///
    /// Avoids TRT's expensive first-inference JIT compilation at the
    /// cost of slightly lower throughput. Only meaningful when `gpu`
    /// is also `true`.
    pub cuda_only: bool,
    /// Policy for host materialization of device-resident frames during
    /// preprocessing.
    ///
    /// When `gpu = true` but no device-native preprocessor is configured,
    /// every frame pays a GPU→CPU download cost. This policy makes that
    /// cost explicit.
    ///
    /// Default: [`Auto`](HostFallbackPolicy::Auto) — resolves to
    /// [`Warn`](HostFallbackPolicy::Warn) when `gpu = true`,
    /// [`Allow`](HostFallbackPolicy::Allow) otherwise.
    ///
    /// Set to an explicit variant to override the auto-resolution.
    pub host_fallback: HostFallbackPolicy,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("samples/models/yolo26s.onnx"),
            input_size: 640,
            confidence_threshold: 0.25,
            class_names: Vec::new(),
            gpu: false,
            cuda_only: false,
            host_fallback: HostFallbackPolicy::Auto,
        }
    }
}

impl DetectorConfig {
    /// Returns the effective host fallback policy, resolving [`Auto`](HostFallbackPolicy::Auto)
    /// based on the `gpu` flag.
    ///
    /// - `Auto` + `gpu = true`  → `Warn`
    /// - `Auto` + `gpu = false` → `Allow`
    /// - Any explicit variant (`Allow`, `Warn`, `Forbid`) → unchanged.
    pub fn effective_host_fallback(&self) -> HostFallbackPolicy {
        if self.host_fallback == HostFallbackPolicy::Auto {
            if self.gpu {
                HostFallbackPolicy::Warn
            } else {
                HostFallbackPolicy::Allow
            }
        } else {
            self.host_fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Auto resolves to Warn when GPU is enabled.
    #[test]
    fn auto_resolves_to_warn_for_gpu() {
        let cfg = DetectorConfig {
            gpu: true,
            host_fallback: HostFallbackPolicy::Auto,
            ..Default::default()
        };
        assert_eq!(cfg.effective_host_fallback(), HostFallbackPolicy::Warn);
    }

    /// Auto resolves to Allow when GPU is disabled.
    #[test]
    fn auto_resolves_to_allow_for_cpu() {
        let cfg = DetectorConfig {
            gpu: false,
            host_fallback: HostFallbackPolicy::Auto,
            ..Default::default()
        };
        assert_eq!(cfg.effective_host_fallback(), HostFallbackPolicy::Allow);
    }

    /// Explicit Allow is respected even when GPU is enabled.
    #[test]
    fn explicit_allow_stays_allow_in_gpu_mode() {
        let cfg = DetectorConfig {
            gpu: true,
            host_fallback: HostFallbackPolicy::Allow,
            ..Default::default()
        };
        assert_eq!(cfg.effective_host_fallback(), HostFallbackPolicy::Allow);
    }

    /// Explicit Warn is respected even when GPU is disabled.
    #[test]
    fn explicit_warn_stays_warn_in_cpu_mode() {
        let cfg = DetectorConfig {
            gpu: false,
            host_fallback: HostFallbackPolicy::Warn,
            ..Default::default()
        };
        assert_eq!(cfg.effective_host_fallback(), HostFallbackPolicy::Warn);
    }

    /// Explicit Forbid is always respected.
    #[test]
    fn explicit_forbid_is_always_respected() {
        for gpu in [true, false] {
            let cfg = DetectorConfig {
                gpu,
                host_fallback: HostFallbackPolicy::Forbid,
                ..Default::default()
            };
            assert_eq!(cfg.effective_host_fallback(), HostFallbackPolicy::Forbid);
        }
    }

    /// Default config uses Auto.
    #[test]
    fn default_config_is_auto() {
        let cfg = DetectorConfig::default();
        assert_eq!(cfg.host_fallback, HostFallbackPolicy::Auto);
    }
}
