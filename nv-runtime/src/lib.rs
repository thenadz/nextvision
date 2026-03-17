//! # nv-runtime
//!
//! Pipeline orchestration and runtime for the NextVision video perception library.
//!
//! ## Conceptual model
//!
//! The runtime manages **feeds** — independent video streams, each running on a
//! dedicated OS thread. For every frame, a linear pipeline of user-defined
//! **[stages](nv_perception::Stage)** produces structured perception output.
//!
//! ```text
//! Media source → FrameQueue → [Stage 1 → Stage 2 → …] → OutputSink
//!                                 │                             │
//!                          TemporalStore               Broadcast channel
//!                          ViewState                   (optional subscribers)
//! ```
//!
//! The media backend (currently GStreamer, via `nv-media`) is an implementation
//! detail. Users interact with backend-agnostic types: [`SourceSpec`](nv_core::SourceSpec),
//! [`FeedConfig`], and [`OutputEnvelope`]. A custom backend can be injected via
//! [`RuntimeBuilder::ingress_factory`].
//!
//! ## Key types
//!
//! - **[`Runtime`]** — manages cross-feed concerns (thread pools, metrics, shutdown).
//! - **[`RuntimeBuilder`]** — builder for configuring and constructing a `Runtime`.
//! - **[`RuntimeHandle`]** — cloneable control handle (add/remove feeds, subscribe, shutdown).
//! - **[`FeedConfig`]** / **[`FeedConfigBuilder`]** — per-feed configuration.
//! - **[`FeedHandle`]** — handle to a running feed (metrics, diagnostics, queue telemetry, pause/resume).
//! - **[`QueueTelemetry`]** — source/sink queue depth and capacity snapshot.
//! - **[`OutputEnvelope`]** — structured, provenanced output for each processed frame.
//! - **[`OutputSink`]** — user-implementable trait for receiving outputs.
//! - **[`Provenance`]** — full audit trail of stage and view-system decisions.
//! - **[`BackpressurePolicy`]** — queue behavior configuration.
//! - **[`FeedDiagnostics`]** / **[`RuntimeDiagnostics`]** — consolidated diagnostic snapshots.
//!
//! ## PTZ / view-state handling
//!
//! Moving-camera feeds use `CameraMode::Observed` with a user-supplied
//! [`ViewStateProvider`](nv_view::ViewStateProvider). The runtime polls it
//! each frame, runs an [`EpochPolicy`](nv_view::EpochPolicy), and manages
//! view epochs, continuity degradation, and trajectory segmentation
//! automatically. Fixed cameras use `CameraMode::Fixed` and skip the
//! view-state machinery entirely.
//!
//! ## Out of scope
//!
//! The runtime does **not** include domain-specific event taxonomies,
//! alerting workflows, calibration semantics, or UI concerns. Those
//! belong in layers built on top of this library.
//!
//! ## Minimal usage
//!
//! ```rust,no_run
//! use nv_runtime::*;
//! use nv_core::*;
//!
//! # struct MyStage;
//! # impl nv_perception::Stage for MyStage {
//! #     fn id(&self) -> StageId { StageId("my_stage") }
//! #     fn process(&mut self, _ctx: &nv_perception::StageContext<'_>) -> Result<nv_perception::StageOutput, StageError> {
//! #         Ok(nv_perception::StageOutput::empty())
//! #     }
//! # }
//! struct MySink;
//! impl OutputSink for MySink {
//!     fn emit(&self, _output: SharedOutput) {}
//! }
//!
//! # fn example() -> Result<(), NvError> {
//! let runtime = Runtime::builder().build()?;
//! let _feed = runtime.add_feed(
//!     FeedConfig::builder()
//!         .source(SourceSpec::rtsp("rtsp://cam/stream"))
//!         .camera_mode(CameraMode::Fixed)
//!         .stages(vec![Box::new(MyStage)])
//!         .output_sink(Box::new(MySink))
//!         .build()?
//! )?;
//! // runtime.shutdown();
//! # Ok(())
//! # }
//! ```
//!
//! ## Batch inference across feeds
//!
//! Multiple feeds can share a single GPU-accelerated batch processor via
//! [`BatchHandle`]. Create a batch handle once, then reference it from
//! each feed's pipeline.
//!
//! ```rust,no_run
//! use nv_runtime::*;
//! use nv_core::*;
//! use nv_perception::batch::{BatchProcessor, BatchEntry};
//! use std::time::Duration;
//!
//! # struct MyDetector;
//! # impl BatchProcessor for MyDetector {
//! #     fn id(&self) -> StageId { StageId("detector") }
//! #     fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
//! #         for item in items.iter_mut() { item.output = Some(nv_perception::StageOutput::empty()); }
//! #         Ok(())
//! #     }
//! # }
//! # struct MySink;
//! # impl OutputSink for MySink { fn emit(&self, _: SharedOutput) {} }
//!
//! # fn example() -> Result<(), NvError> {
//! let runtime = Runtime::builder().build()?;
//!
//! // Create a shared batch coordinator.
//! let batch = runtime.create_batch(
//!     Box::new(MyDetector),
//!     BatchConfig {
//!         max_batch_size: 8,
//!         max_latency: Duration::from_millis(50),
//!         queue_capacity: None,
//!         response_timeout: None,
//!         max_in_flight_per_feed: 1,
//!         startup_timeout: None,
//!     },
//! )?;
//!
//! // Build per-feed pipelines referencing the shared batch.
//! let pipeline = FeedPipeline::builder()
//!     .batch(batch.clone()).expect("single batch point")
//!     .build();
//!
//! let _feed = runtime.add_feed(
//!     FeedConfig::builder()
//!         .source(SourceSpec::rtsp("rtsp://cam/stream"))
//!         .camera_mode(CameraMode::Fixed)
//!         .feed_pipeline(pipeline)
//!         .output_sink(Box::new(MySink))
//!         .build()?
//! )?;
//! # Ok(())
//! # }
//! ```

pub mod backpressure;
pub mod batch;
pub mod diagnostics;
pub(crate) mod executor;
pub mod feed;
pub mod feed_handle;
pub mod output;
pub mod pipeline;
pub mod provenance;
pub(crate) mod queue;
pub mod runtime;
pub mod shutdown;
pub(crate) mod worker;

// Re-export key types at crate root.
pub use backpressure::BackpressurePolicy;
pub use batch::{BatchConfig, BatchHandle, BatchMetrics};
pub use diagnostics::{BatchDiagnostics, FeedDiagnostics, OutputLagStatus, RuntimeDiagnostics, ViewDiagnostics, ViewStatus};
pub use feed::{FeedConfig, FeedConfigBuilder};
pub use feed_handle::{DecodeStatus, FeedHandle, QueueTelemetry};
pub use output::{AdmissionSummary, FrameInclusion, OutputEnvelope, OutputSink, SharedOutput};
pub use pipeline::{FeedPipeline, FeedPipelineBuilder, PipelineError};
pub use provenance::{
    Provenance, StageOutcomeCategory, StageProvenance, StageResult, ViewProvenance,
};
pub use runtime::{Runtime, RuntimeBuilder, RuntimeHandle};
pub use shutdown::{RestartPolicy, RestartTrigger};

// Re-export validation types from nv-perception for convenience.
pub use nv_perception::{ValidationMode, ValidationWarning};

// Re-export decode types from nv-media for convenience.
pub use nv_media::{DecodePreference, DecodeCapabilities, discover_decode_capabilities};

// Re-export health types from nv-core for convenience.
pub use nv_core::health::{DecodeOutcome, HealthEvent};
