//! # nv-byo-yolox
//!
//! YOLOX-s ONNX detector adapter for the NextVision perception pipeline.
//!
//! This crate provides [`YoloxStage`], an implementation of
//! [`nv_perception::Stage`] that runs YOLOX-s inference via ONNX Runtime
//! and produces [`DetectionSet`](nv_perception::DetectionSet) outputs.
//!
//! ## Preprocessing
//!
//! Input frames are letterbox-resized to the model's expected input size
//! (default 640×640), preserving aspect ratio with gray padding, then
//! normalised to `[0, 1]` float32 in NCHW layout.
//!
//! ## Postprocessing
//!
//! YOLOX decodes grid-based anchor-free predictions, applies confidence
//! thresholding, and runs non-maximum suppression (NMS). Final bounding
//! boxes are remapped from letterbox coordinates back to normalised
//! `[0, 1]` frame coordinates.
//!
//! ## Usage
//!
//! ```no_run
//! use nv_byo_yolox::{YoloxConfig, YoloxStage};
//!
//! let config = YoloxConfig {
//!     model_path: "yolox_s.onnx".into(),
//!     ..Default::default()
//! };
//! let stage = YoloxStage::new(config);
//! // Add to a StagePipeline via pipeline.add(stage)
//! ```

mod config;
mod letterbox;
mod nms;
mod postprocess;
mod stage;

pub use config::YoloxConfig;
pub use stage::YoloxStage;

// Re-export helpers for testing and advanced use.
pub use letterbox::{LetterboxInfo, letterbox_preprocess};
pub use nms::nms;
pub use postprocess::decode_yolox_output;
