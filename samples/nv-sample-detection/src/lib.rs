//! # nv-sample-detection
//!
//! Sample ONNX object-detector adapter for the NextVision perception pipeline.
//!
//! This crate is a **reference implementation** — it demonstrates how to
//! integrate an ONNX detector model with the library's [`Stage`] and
//! [`BatchProcessor`] traits. Library users will typically replace this
//! with their own detection backend.
//!
//! Two execution modes are provided:
//!
//! - [`DetectorStage`] — per-feed [`Stage`](nv_perception::Stage) for
//!   single-feed pipelines.
//! - [`DetectorBatchProcessor`] — cross-feed
//!   [`BatchProcessor`](nv_perception::batch::BatchProcessor) that
//!   batches frames from multiple feeds into one inference call.
//!
//! Both share the same preprocessing (letterbox) and model-loading logic.
//!
//! ## Preprocessing
//!
//! Input frames are letterbox-resized to the model's expected input size
//! (default 640×640), preserving aspect ratio with gray padding, then
//! converted to float32 in NCHW layout. Values are normalised to the
//! 0–1 range.
//!
//! ## Postprocessing
//!
//! The model is expected to produce end-to-end detections in
//! `[x1, y1, x2, y2, confidence, class_id]` format — no NMS needed.
//! We apply confidence thresholding and remap bounding boxes from
//! letterbox coordinates back to normalised `[0, 1]` frame coordinates.
//!
//! ## Usage
//!
//! ```no_run
//! use nv_sample_detection::{DetectorConfig, DetectorStage};
//!
//! let config = DetectorConfig {
//!     model_path: "samples/models/yolo26s.onnx".into(),
//!     ..Default::default()
//! };
//! let stage = DetectorStage::new(config);
//! // Add to a StagePipeline via pipeline.add(stage)
//! ```

mod batch;
mod config;
mod inference;
mod letterbox;
mod postprocess;
pub mod preprocess;
pub mod session;
mod stage;

pub use batch::DetectorBatchProcessor;
pub use config::DetectorConfig;
pub use preprocess::{
    BatchFramePreprocessor, FramePreprocessor, HostBatchPreprocessor, HostFallbackPolicy,
    HostPreprocessor, PreprocessedFrame,
};
pub use stage::DetectorStage;

// Re-export helpers for testing and advanced use.
pub use letterbox::{LetterboxInfo, letterbox_preprocess, letterbox_preprocess_into};
pub use postprocess::decode_end2end_output;
