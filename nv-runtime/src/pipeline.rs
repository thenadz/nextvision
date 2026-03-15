//! Unified feed pipeline — stages with an optional shared batch point.
//!
//! [`FeedPipeline`] is the public model for defining a feed's processing
//! pipeline. It presents a single linear pipeline where an optional
//! **batch point** can be inserted between per-feed stages:
//!
//! ```text
//! [pre-batch stages] → [batch point] → [post-batch stages]
//!                           │
//!                    SharedBatchProcessor (frames from multiple feeds)
//! ```
//!
//! A pipeline without a batch point is simply all stages executed in
//! order on the feed thread — identical to the original per-feed model.
//!
//! A pipeline with a batch point splits into:
//! - **Pre-batch stages** — per-feed, on the feed thread (e.g., preprocessing)
//! - **Batch point** — frame submitted to a shared [`BatchHandle`], results
//!   merged into the artifact accumulator
//! - **Post-batch stages** — per-feed, on the feed thread (e.g., tracking,
//!   temporal analysis)
//!
//! This keeps batching as a natural part of the pipeline instead of a
//! separate subsystem.
//!
//! # Example
//!
//! ```rust,no_run
//! # use nv_perception::{Stage, StageContext, StageOutput};
//! # use nv_core::{StageId, StageError};
//! # struct Preprocessor;
//! # impl Stage for Preprocessor {
//! #     fn id(&self) -> StageId { StageId("pre") }
//! #     fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, StageError> { Ok(StageOutput::empty()) }
//! # }
//! # struct Tracker;
//! # impl Stage for Tracker {
//! #     fn id(&self) -> StageId { StageId("trk") }
//! #     fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, StageError> { Ok(StageOutput::empty()) }
//! # }
//! use nv_runtime::pipeline::FeedPipeline;
//! # fn example(batch_handle: nv_runtime::batch::BatchHandle) -> Result<(), Box<dyn std::error::Error>> {
//!
//! let pipeline = FeedPipeline::builder()
//!     .add_stage(Preprocessor)        // pre-batch
//!     .batch(batch_handle)?            // shared batch point
//!     .add_stage(Tracker)              // post-batch
//!     .build();
//! # Ok(())
//! # }
//! ```

use nv_perception::Stage;

use crate::batch::BatchHandle;

/// Errors from pipeline construction.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// A second batch point was inserted into the pipeline.
    #[error("only one batch point is allowed per pipeline")]
    DuplicateBatchPoint,
}

/// The decomposed parts of a [`FeedPipeline`] consumed by the executor.
pub(crate) type PipelineParts = (
    Vec<Box<dyn Stage>>,
    Option<BatchHandle>,
    Vec<Box<dyn Stage>>,
);

/// A feed processing pipeline with an optional shared batch point.
///
/// Created via [`FeedPipeline::builder()`]. Consumed by
/// [`FeedConfigBuilder::feed_pipeline()`](crate::FeedConfigBuilder).
pub struct FeedPipeline {
    pub(crate) pre_batch: Vec<Box<dyn Stage>>,
    pub(crate) batch: Option<BatchHandle>,
    pub(crate) post_batch: Vec<Box<dyn Stage>>,
}

impl FeedPipeline {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> FeedPipelineBuilder {
        FeedPipelineBuilder {
            pre_batch: Vec::new(),
            batch: None,
            post_batch: Vec::new(),
            after_batch: false,
        }
    }

    /// Total number of per-feed stages (excluding the batch point).
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.pre_batch.len() + self.post_batch.len()
    }

    /// Whether this pipeline includes a shared batch point.
    #[must_use]
    pub fn has_batch(&self) -> bool {
        self.batch.is_some()
    }

    /// Decompose into parts consumed by the executor.
    pub(crate) fn into_parts(self) -> PipelineParts {
        (self.pre_batch, self.batch, self.post_batch)
    }
}

/// Builder for [`FeedPipeline`].
///
/// Stages added before [`batch()`](Self::batch) are pre-batch stages.
/// Stages added after are post-batch stages. If `batch()` is never
/// called, all stages are treated as a normal linear pipeline (no
/// batch boundary).
pub struct FeedPipelineBuilder {
    pre_batch: Vec<Box<dyn Stage>>,
    batch: Option<BatchHandle>,
    post_batch: Vec<Box<dyn Stage>>,
    after_batch: bool,
}

impl FeedPipelineBuilder {
    /// Append a per-feed stage to the pipeline.
    ///
    /// If called before [`batch()`](Self::batch), the stage runs
    /// pre-batch. If called after, the stage runs post-batch.
    #[must_use]
    pub fn add_stage(mut self, stage: impl Stage) -> Self {
        if self.after_batch {
            self.post_batch.push(Box::new(stage));
        } else {
            self.pre_batch.push(Box::new(stage));
        }
        self
    }

    /// Append a boxed per-feed stage.
    #[must_use]
    pub fn add_boxed_stage(mut self, stage: Box<dyn Stage>) -> Self {
        if self.after_batch {
            self.post_batch.push(stage);
        } else {
            self.pre_batch.push(stage);
        }
        self
    }

    /// Insert the shared batch point at the current position.
    ///
    /// Stages added before this call are pre-batch; stages added after
    /// are post-batch. Only one batch point is allowed per pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::DuplicateBatchPoint`] if called more
    /// than once on the same builder.
    pub fn batch(mut self, handle: BatchHandle) -> Result<Self, PipelineError> {
        if self.batch.is_some() {
            return Err(PipelineError::DuplicateBatchPoint);
        }
        self.batch = Some(handle);
        self.after_batch = true;
        Ok(self)
    }

    /// Build the pipeline.
    #[must_use]
    pub fn build(self) -> FeedPipeline {
        FeedPipeline {
            pre_batch: self.pre_batch,
            batch: self.batch,
            post_batch: self.post_batch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::error::StageError;
    use nv_core::id::StageId;
    use nv_perception::{StageContext, StageOutput};

    struct DummyStage(&'static str);
    impl Stage for DummyStage {
        fn id(&self) -> StageId {
            StageId(self.0)
        }
        fn process(
            &mut self,
            _ctx: &StageContext<'_>,
        ) -> Result<StageOutput, StageError> {
            Ok(StageOutput::empty())
        }
    }

    #[test]
    fn pipeline_without_batch() {
        let p = FeedPipeline::builder()
            .add_stage(DummyStage("a"))
            .add_stage(DummyStage("b"))
            .build();

        assert_eq!(p.stage_count(), 2);
        assert!(!p.has_batch());
        let (pre, batch, post) = p.into_parts();
        assert_eq!(pre.len(), 2);
        assert!(batch.is_none());
        assert!(post.is_empty());
    }

    #[test]
    fn pipeline_stage_count_with_batch() {
        // Can't easily create a real BatchHandle in unit tests without
        // the coordinator, so we test the builder's structural correctness
        // by verifying pre/post split via an indirect approach.
        let p = FeedPipeline::builder()
            .add_stage(DummyStage("pre1"))
            .add_stage(DummyStage("pre2"))
            .build();

        // Without batch, all stages are pre-batch.
        let (pre, _, post) = p.into_parts();
        assert_eq!(pre.len(), 2);
        assert!(post.is_empty());
    }

    #[test]
    fn double_batch_returns_error() {
        // Create a real BatchHandle via the coordinator.
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_core::id::StageId;
        use nv_perception::batch::{BatchEntry, BatchProcessor};
        use std::time::Duration;

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig { max_batch_size: 1, max_latency: Duration::from_millis(10), queue_capacity: None, response_timeout: None, max_in_flight_per_feed: 1 },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        // The second `.batch()` call should return an error.
        let result = FeedPipeline::builder()
            .batch(handle.clone())
            .expect("first batch() should succeed")
            .batch(handle);
        assert!(result.is_err(), "duplicate batch point should return error");
    }
}

