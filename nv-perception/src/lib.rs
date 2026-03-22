//! # nv-perception
//!
//! Perception stage model and artifact types for the NextVision runtime.
//!
//! This crate defines:
//!
//! - **[`Stage`]** — the core user-implementable trait for per-frame processing.
//! - **[`Detection`]**, **[`DetectionSet`]** — per-frame object detections.
//! - **[`Track`]**, **[`TrackObservation`]**, **[`TrackState`]** — tracked objects.
//! - **[`PerceptionArtifacts`]** — accumulated outputs of all stages for a frame.
//! - **[`DerivedSignal`]**, **[`SignalValue`]** — generic named signals.
//! - **[`SceneFeature`]**, **[`SceneFeatureValue`]** — scene-level feature observations.
//! - **[`StagePipeline`]** — ordered pipeline builder with **[`validate()`](StagePipeline::validate)**.
//! - **[`StageCapabilities`]** — declared stage inputs/outputs for validation.
//!
//! ## Supported model patterns
//!
//! The library supports multiple perception model architectures through
//! the same [`Stage`] trait. A stage may produce *any combination* of
//! output fields — the pipeline does not enforce a fixed detect→track
//! sequence.
//!
//! | Pattern | Description |
//! |---|---|
//! | Classical detector → tracker | Separate detection and tracking stages in sequence. |
//! | Joint detection+tracking | Single stage producing both `detections` and `tracks`. |
//! | Direct track emitter | Model outputs tracks directly, no intermediate detections. |
//! | Richer observations | Per-observation metadata via [`TrackObservation::metadata`]. |
//! | Scene/temporal analysis | Stages producing [`SceneFeature`]s or [`DerivedSignal`]s. |
//!
//! See the [`Stage`] trait documentation for detailed examples and the
//! [`StageOutput`] docs for joint-model and direct-track-emitter patterns.
//!
//! ## Ergonomic entity construction
//!
//! Use [`Detection::builder()`] and [`Track::new()`] to avoid manual struct
//! construction. [`StageOutput::build()`] provides an incremental builder
//! for assembling multi-field outputs, and [`StageOutput::with_artifact()`]
//! is a shorthand for typed artifact-only output.
//!
//! ## Convenience re-exports
//!
//! This crate re-exports types that appear in [`Stage`] trait signatures
//! so that stage authors can import everything from a single crate:
//! [`StageId`], [`StageError`], [`FeedId`], [`ViewEpoch`], [`ViewSnapshot`],
//! [`FrameEnvelope`], [`TypedMetadata`], and [`StageMetrics`].
//!
//! ## Stage execution model
//!
//! Stages execute linearly and synchronously within a feed. For each frame:
//! 1. A `PerceptionArtifacts` accumulator starts empty.
//! 2. Each stage receives a [`StageContext`] with the frame, prior artifacts,
//!    temporal snapshot, and view snapshot.
//! 3. Each stage returns a [`StageOutput`] that is merged into the accumulator.
//! 4. After all stages run, the accumulator feeds the temporal store and output.
//!
//! Stages take `&mut self` — they run on a dedicated thread per feed and are
//! never shared across threads.
//!
//! ## Single `Stage` trait
//!
//! The library uses one trait for all stage types — detection, tracking,
//! classification, scene analysis, etc. Specialization happens by convention
//! (which [`StageOutput`] fields a stage populates), not by type hierarchy.
//! The abstraction stays minimal, avoids taxonomy assumptions, and lets
//! users compose arbitrary pipeline shapes.
//!
//! ## Extension seam: typed artifacts
//!
//! For inter-stage data that does not fit the built-in fields (feature maps,
//! prepared clip/window tensors, calibration data), stages use the
//! [`stage_artifacts`](PerceptionArtifacts::stage_artifacts) type-map.
//! A pre-processing stage can assemble a sliding window and store it as a
//! typed artifact for a downstream inference stage to consume.
//!
//! ## Backend independence
//!
//! This crate has no dependency on GStreamer or any media backend. Stages
//! receive [`FrameEnvelope`](nv_frame::FrameEnvelope) (generic pixel data)
//! and produce domain-agnostic artifacts.

pub mod artifact;
pub mod batch;
pub mod detection;
pub mod pipeline;
pub mod scene;
pub mod signal;
pub mod stage;
pub mod temporal_access;
pub mod track;

pub use artifact::PerceptionArtifacts;
pub use batch::{BatchEntry, BatchProcessor};
pub use detection::{Detection, DetectionBuilder, DetectionSet};
pub use pipeline::{
    StagePipeline, StagePipelineBuilder, ValidationMode, ValidationWarning,
    validate_pipeline_phased, validate_stages,
};
pub use scene::{SceneFeature, SceneFeatureValue};
pub use signal::{DerivedSignal, SignalValue};
pub use stage::{
    Stage, StageCapabilities, StageCategory, StageContext, StageOutput, StageOutputBuilder,
};
pub use temporal_access::TemporalAccess;
pub use track::{Track, TrackObservation, TrackState};

// Re-export types that appear in Stage trait signatures and StageContext,
// so that stage authors can import everything from `nv_perception` without
// needing direct dependencies on nv-core, nv-frame, or nv-view.
pub use nv_core::TypedMetadata;
pub use nv_core::error::StageError;
pub use nv_core::id::{FeedId, StageId};
pub use nv_core::metrics::StageMetrics;
pub use nv_frame::FrameEnvelope;
pub use nv_view::{ViewEpoch, ViewSnapshot};
