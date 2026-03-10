//! # nv-runtime
//!
//! Pipeline orchestration and runtime for the NextVision video perception library.
//!
//! This crate provides the top-level user-facing API:
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
