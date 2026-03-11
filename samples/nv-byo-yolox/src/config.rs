use std::path::PathBuf;

/// Configuration for the YOLOX-s detector stage.
#[derive(Debug, Clone)]
pub struct YoloxConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Model input size (width and height must match the ONNX graph).
    /// Default: 640.
    pub input_size: u32,
    /// Minimum confidence to keep a detection before NMS.
    /// Default: 0.25.
    pub confidence_threshold: f32,
    /// IoU threshold for NMS merging.
    /// Default: 0.45.
    pub nms_threshold: f32,
    /// Optional class name table, indexed by `class_id`.
    /// Not stored on detections — provided for downstream consumers
    /// that want human-readable labels.
    pub class_names: Vec<String>,
}

impl Default for YoloxConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("yolox_s.onnx"),
            input_size: 640,
            confidence_threshold: 0.25,
            nms_threshold: 0.45,
            class_names: Vec::new(),
        }
    }
}
