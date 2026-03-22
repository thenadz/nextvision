use std::path::PathBuf;

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
        }
    }
}
