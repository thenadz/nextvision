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
//! - **[`FeedHandle`]** — handle to a running feed (health, metrics, pause/resume).
//! - **[`OutputEnvelope`]** — structured, provenanced output for each processed frame.
//! - **[`OutputSink`]** — user-implementable trait for receiving outputs.
//! - **[`Provenance`]** — full audit trail of stage and view-system decisions.
//! - **[`BackpressurePolicy`]** — queue behavior configuration.
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
//!     fn emit(&self, _output: OutputEnvelope) {}
//! }
//!
//! # async fn example() -> Result<(), NvError> {
//! let runtime = Runtime::builder().build()?;
//! let _feed = runtime.add_feed(
//!     FeedConfig::builder()
//!         .source(SourceSpec::rtsp("rtsp://cam/stream"))
//!         .camera_mode(CameraMode::Fixed)
//!         .stages(vec![Box::new(MyStage)])
//!         .output_sink(Box::new(MySink))
//!         .build()?
//! )?;
//! // runtime.shutdown().await?;
//! # Ok(())
//! # }
//! ```

pub mod backpressure;
pub(crate) mod executor;
pub mod feed;
pub mod output;
pub mod provenance;
pub(crate) mod queue;
pub mod runtime;
pub mod shutdown;
pub(crate) mod worker;

// Re-export key types at crate root.
pub use backpressure::BackpressurePolicy;
pub use feed::{FeedConfig, FeedConfigBuilder, FeedHandle};
pub use output::{OutputEnvelope, OutputSink, SharedOutput};
pub use provenance::{
    Provenance, StageOutcomeCategory, StageProvenance, StageResult, ViewProvenance,
};
pub use runtime::{Runtime, RuntimeBuilder, RuntimeHandle};
pub use shutdown::{RestartPolicy, RestartTrigger};
