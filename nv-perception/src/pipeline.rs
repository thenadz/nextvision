//! Stage pipeline composition — ordered collections with optional validation.
//!
//! [`StagePipeline`] provides a builder for composing stages into an
//! ordered pipeline. The resulting pipeline can be passed directly to
//! [`FeedConfigBuilder::pipeline`] or destructured into a `Vec<Box<dyn Stage>>`.
//!
//! # Example
//!
//! ```rust,no_run
//! use nv_perception::StagePipeline;
//! # use nv_perception::{Stage, StageContext, StageOutput};
//! # use nv_core::{StageId, StageError};
//! # struct MyDetector;
//! # impl Stage for MyDetector {
//! #     fn id(&self) -> StageId { StageId("det") }
//! #     fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, StageError> {
//! #         Ok(StageOutput::empty())
//! #     }
//! # }
//! # struct MyTracker;
//! # impl Stage for MyTracker {
//! #     fn id(&self) -> StageId { StageId("trk") }
//! #     fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, StageError> {
//! #         Ok(StageOutput::empty())
//! #     }
//! # }
//!
//! let pipeline = StagePipeline::builder()
//!     .add(MyDetector)
//!     .add(MyTracker)
//!     .build();
//!
//! assert_eq!(pipeline.len(), 2);
//! let stages = pipeline.into_stages();
//! ```

use crate::stage::{Stage, StageCategory};
use nv_core::id::StageId;

/// An ordered, validated collection of perception stages.
///
/// Built via [`StagePipeline::builder()`]. The pipeline can be inspected
/// for stage IDs and categories before being consumed as a `Vec<Box<dyn Stage>>`.
pub struct StagePipeline {
    stages: Vec<Box<dyn Stage>>,
}

impl StagePipeline {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> StagePipelineBuilder {
        StagePipelineBuilder {
            stages: Vec::new(),
        }
    }

    /// Number of stages in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the pipeline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Get the stage IDs in execution order.
    #[must_use]
    pub fn stage_ids(&self) -> Vec<StageId> {
        self.stages.iter().map(|s| s.id()).collect()
    }

    /// Get `(StageId, StageCategory)` pairs in execution order.
    #[must_use]
    pub fn categories(&self) -> Vec<(StageId, StageCategory)> {
        self.stages
            .iter()
            .map(|s| (s.id(), s.category()))
            .collect()
    }

    /// Consume the pipeline and return the ordered stage list.
    ///
    /// Suitable for passing to [`FeedConfigBuilder::stages`] or
    /// [`FeedConfigBuilder::pipeline`].
    #[must_use]
    pub fn into_stages(self) -> Vec<Box<dyn Stage>> {
        self.stages
    }
}

/// Builder for [`StagePipeline`].
pub struct StagePipelineBuilder {
    stages: Vec<Box<dyn Stage>>,
}

impl StagePipelineBuilder {
    /// Append a stage to the pipeline.
    #[must_use]
    pub fn add(mut self, stage: impl Stage) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Append a boxed stage to the pipeline.
    #[must_use]
    pub fn add_boxed(mut self, stage: Box<dyn Stage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the pipeline.
    #[must_use]
    pub fn build(self) -> StagePipeline {
        StagePipeline {
            stages: self.stages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StageContext, StageOutput};
    use nv_core::error::StageError;

    struct TestStage {
        name: &'static str,
        cat: StageCategory,
    }

    impl Stage for TestStage {
        fn id(&self) -> StageId {
            StageId(self.name)
        }
        fn process(
            &mut self,
            _ctx: &StageContext<'_>,
        ) -> Result<StageOutput, StageError> {
            Ok(StageOutput::empty())
        }
        fn category(&self) -> StageCategory {
            self.cat
        }
    }

    #[test]
    fn builder_preserves_order() {
        let pipeline = StagePipeline::builder()
            .add(TestStage {
                name: "det",
                cat: StageCategory::FrameAnalysis,
            })
            .add(TestStage {
                name: "trk",
                cat: StageCategory::Association,
            })
            .add(TestStage {
                name: "temporal",
                cat: StageCategory::TemporalAnalysis,
            })
            .add(TestStage {
                name: "sink",
                cat: StageCategory::Sink,
            })
            .build();

        let ids: Vec<&str> = pipeline.stage_ids().iter().map(|s| s.as_str()).collect();
        assert_eq!(ids, vec!["det", "trk", "temporal", "sink"]);
    }

    #[test]
    fn categories_reported_correctly() {
        let pipeline = StagePipeline::builder()
            .add(TestStage {
                name: "det",
                cat: StageCategory::FrameAnalysis,
            })
            .add(TestStage {
                name: "trk",
                cat: StageCategory::Association,
            })
            .build();

        let cats = pipeline.categories();
        assert_eq!(cats[0].1, StageCategory::FrameAnalysis);
        assert_eq!(cats[1].1, StageCategory::Association);
    }

    #[test]
    fn into_stages_returns_owned_vec() {
        let pipeline = StagePipeline::builder()
            .add(TestStage {
                name: "a",
                cat: StageCategory::Custom,
            })
            .add(TestStage {
                name: "b",
                cat: StageCategory::Custom,
            })
            .build();

        let stages = pipeline.into_stages();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].id(), StageId("a"));
        assert_eq!(stages[1].id(), StageId("b"));
    }

    #[test]
    fn empty_pipeline() {
        let pipeline = StagePipeline::builder().build();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }
}
