//! Stage pipeline composition — ordered collections with optional validation.
//!
//! [`StagePipeline`] provides a builder for composing stages into an
//! ordered pipeline. The resulting pipeline can be passed directly to
//! `FeedConfigBuilder::pipeline` or destructured into a `Vec<Box<dyn Stage>>`.
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

use crate::stage::{Stage, StageCapabilities, StageCategory};
use nv_core::id::StageId;

/// Controls whether [`StagePipeline::validate`] / [`validate_stages`]
/// warnings are ignored, logged, or promoted to hard errors.
///
/// Used by `FeedConfigBuilder` to wire validation into
/// the normal build path without requiring callers to invoke
/// `validate()` manually.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ValidationMode {
    /// Validation is skipped entirely (default).
    #[default]
    Off,
    /// Validation runs; warnings are returned but do not prevent
    /// pipeline construction.
    Warn,
    /// Validation runs; any warning is promoted to a hard error.
    Error,
}

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
        StagePipelineBuilder { stages: Vec::new() }
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
        self.stages.iter().map(|s| (s.id(), s.category())).collect()
    }

    /// Consume the pipeline and return the ordered stage list.
    ///
    /// Suitable for passing to `FeedConfigBuilder::stages` or
    /// `FeedConfigBuilder::pipeline`.
    #[must_use]
    pub fn into_stages(self) -> Vec<Box<dyn Stage>> {
        self.stages
    }

    /// Validate stage ordering based on declared [`StageCapabilities`].
    ///
    /// Returns a (possibly empty) list of warnings. Stages that return
    /// `None` from [`Stage::capabilities()`] are silently skipped.
    ///
    /// Warnings are advisory — the pipeline will still execute regardless.
    /// This allows pipeline builders to catch common ordering mistakes
    /// (e.g., placing a tracker before a detector) at construction time.
    #[must_use]
    pub fn validate(&self) -> Vec<ValidationWarning> {
        validate_stages(&self.stages)
    }
}

/// Advisory warning from [`StagePipeline::validate()`].
///
/// These do **not** prevent pipeline execution. They flag likely
/// composition mistakes that the builder may want to address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationWarning {
    /// A stage declares that it consumes an artifact type that no
    /// earlier stage produces.
    UnsatisfiedDependency {
        /// The stage with the unsatisfied dependency.
        stage_id: StageId,
        /// Human-readable name of the missing artifact type.
        missing: &'static str,
    },
    /// Two or more stages share the same [`StageId`].
    DuplicateStageId {
        /// The duplicated stage ID.
        stage_id: StageId,
    },
}

/// Validate an ordered stage slice and return advisory warnings.
///
/// This is the same logic as [`StagePipeline::validate`] but operates
/// on a borrowed slice, making it usable by `FeedConfigBuilder`
/// without requiring a `StagePipeline`.
#[must_use]
pub fn validate_stages(stages: &[Box<dyn Stage>]) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    let mut detections_available = false;
    let mut tracks_available = false;

    for stage in stages {
        validate_one_stage(
            &**stage,
            &mut detections_available,
            &mut tracks_available,
            &mut warnings,
        );
    }

    // Check for duplicate stage IDs.
    let mut seen = std::collections::HashSet::new();
    for stage in stages {
        let id = stage.id();
        if !seen.insert(id) {
            warnings.push(ValidationWarning::DuplicateStageId { stage_id: id });
        }
    }

    warnings
}

/// Validate a batch-aware pipeline: pre-batch stages → optional batch
/// processor → post-batch stages.
///
/// Availability state (which artifact types have been produced) flows
/// through all three phases in order.  The batch processor's
/// `consumes_*` requirements are validated against pre-batch
/// availability, and its `produces_*` fields update availability for
/// the post-batch stages.  Duplicate stage IDs across all phases
/// (including the batch processor) are also reported.
///
/// This is the recommended entry-point for validating a pipeline that
/// includes a cross-feed batch processor.  For simple linear pipelines
/// without a batch processor, [`validate_stages`] is sufficient.
#[must_use]
pub fn validate_pipeline_phased(
    pre_batch: &[Box<dyn Stage>],
    batch_caps: Option<&StageCapabilities>,
    batch_id: Option<StageId>,
    post_batch: &[Box<dyn Stage>],
) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    let mut detections_available = false;
    let mut tracks_available = false;

    // Phase 1: pre-batch stages.
    for stage in pre_batch {
        validate_one_stage(
            &**stage,
            &mut detections_available,
            &mut tracks_available,
            &mut warnings,
        );
    }

    // Phase 2: batch processor capabilities (if declared).
    if let Some(caps) = batch_caps {
        if let Some(id) = batch_id {
            if caps.consumes_detections && !detections_available {
                warnings.push(ValidationWarning::UnsatisfiedDependency {
                    stage_id: id,
                    missing: "detections",
                });
            }
            if caps.consumes_tracks && !tracks_available {
                warnings.push(ValidationWarning::UnsatisfiedDependency {
                    stage_id: id,
                    missing: "tracks",
                });
            }
        }

        if caps.produces_detections {
            detections_available = true;
        }
        if caps.produces_tracks {
            tracks_available = true;
        }
    }

    // Phase 3: post-batch stages.
    for stage in post_batch {
        validate_one_stage(
            &**stage,
            &mut detections_available,
            &mut tracks_available,
            &mut warnings,
        );
    }

    // Duplicate ID check across all per-feed stages + batch processor.
    let mut seen = std::collections::HashSet::new();
    if let Some(id) = batch_id {
        seen.insert(id);
    }
    for stage in pre_batch.iter().chain(post_batch.iter()) {
        let id = stage.id();
        if !seen.insert(id) {
            warnings.push(ValidationWarning::DuplicateStageId { stage_id: id });
        }
    }

    warnings
}

/// Check a single stage's capabilities against current availability and
/// update availability from its outputs.
pub(crate) fn validate_one_stage(
    stage: &dyn Stage,
    detections_available: &mut bool,
    tracks_available: &mut bool,
    warnings: &mut Vec<ValidationWarning>,
) {
    let caps = match stage.capabilities() {
        Some(c) => c,
        None => return,
    };
    let id = stage.id();

    if caps.consumes_detections && !*detections_available {
        warnings.push(ValidationWarning::UnsatisfiedDependency {
            stage_id: id,
            missing: "detections",
        });
    }
    if caps.consumes_tracks && !*tracks_available {
        warnings.push(ValidationWarning::UnsatisfiedDependency {
            stage_id: id,
            missing: "tracks",
        });
    }
    if caps.produces_detections {
        *detections_available = true;
    }
    if caps.produces_tracks {
        *tracks_available = true;
    }
}

/// Builder for [`StagePipeline`].
pub struct StagePipelineBuilder {
    stages: Vec<Box<dyn Stage>>,
}

impl StagePipelineBuilder {
    /// Append a stage to the pipeline.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
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
    use crate::stage::StageCapabilities;
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
        fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
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

    /// A test stage with configurable capabilities.
    struct CapStage {
        name: &'static str,
        caps: Option<StageCapabilities>,
    }

    impl Stage for CapStage {
        fn id(&self) -> StageId {
            StageId(self.name)
        }
        fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
            Ok(StageOutput::empty())
        }
        fn capabilities(&self) -> Option<StageCapabilities> {
            self.caps
        }
    }

    #[test]
    fn validate_happy_path() {
        let pipeline = StagePipeline::builder()
            .add(CapStage {
                name: "det",
                caps: Some(StageCapabilities::new().produces_detections()),
            })
            .add(CapStage {
                name: "trk",
                caps: Some(
                    StageCapabilities::new()
                        .consumes_detections()
                        .produces_tracks(),
                ),
            })
            .build();

        let warnings = pipeline.validate();
        assert!(warnings.is_empty());
    }

    #[test]
    fn validate_unsatisfied_detections() {
        let pipeline = StagePipeline::builder()
            .add(CapStage {
                name: "trk",
                caps: Some(StageCapabilities::new().consumes_detections()),
            })
            .add(CapStage {
                name: "det",
                caps: Some(StageCapabilities::new().produces_detections()),
            })
            .build();

        let warnings = pipeline.validate();
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0],
            ValidationWarning::UnsatisfiedDependency {
                stage_id: StageId("trk"),
                missing: "detections",
            }
        );
    }

    #[test]
    fn validate_unsatisfied_tracks() {
        let pipeline = StagePipeline::builder()
            .add(CapStage {
                name: "temporal",
                caps: Some(StageCapabilities::new().consumes_tracks()),
            })
            .build();

        let warnings = pipeline.validate();
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            &warnings[0],
            ValidationWarning::UnsatisfiedDependency {
                missing: "tracks",
                ..
            }
        ));
    }

    #[test]
    fn validate_skips_stages_without_capabilities() {
        let pipeline = StagePipeline::builder()
            .add(CapStage {
                name: "noop",
                caps: None,
            })
            .add(CapStage {
                name: "det",
                caps: Some(StageCapabilities::new().produces_detections()),
            })
            .build();

        let warnings = pipeline.validate();
        assert!(warnings.is_empty());
    }

    #[test]
    fn validate_duplicate_stage_ids() {
        let pipeline = StagePipeline::builder()
            .add(CapStage {
                name: "det",
                caps: None,
            })
            .add(CapStage {
                name: "det",
                caps: None,
            })
            .build();

        let warnings = pipeline.validate();
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            &warnings[0],
            ValidationWarning::DuplicateStageId { stage_id } if *stage_id == StageId("det")
        ));
    }

    #[test]
    fn validate_stages_fn_matches_pipeline_validate() {
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(CapStage {
                name: "trk",
                caps: Some(StageCapabilities::new().consumes_detections()),
            }),
            Box::new(CapStage {
                name: "det",
                caps: Some(StageCapabilities::new().produces_detections()),
            }),
        ];

        let warnings = validate_stages(&stages);
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0],
            ValidationWarning::UnsatisfiedDependency {
                stage_id: StageId("trk"),
                missing: "detections",
            }
        );
    }

    #[test]
    fn validate_stages_fn_happy_path() {
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(CapStage {
                name: "det",
                caps: Some(StageCapabilities::new().produces_detections()),
            }),
            Box::new(CapStage {
                name: "trk",
                caps: Some(
                    StageCapabilities::new()
                        .consumes_detections()
                        .produces_tracks(),
                ),
            }),
        ];

        let warnings = validate_stages(&stages);
        assert!(warnings.is_empty());
    }

    // --- validate_pipeline_phased tests ---

    #[test]
    fn phased_happy_path() {
        let pre: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "det",
            caps: Some(StageCapabilities::new().produces_detections()),
        })];
        let batch_caps = StageCapabilities::new()
            .consumes_detections()
            .produces_tracks();
        let post: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "temporal",
            caps: Some(StageCapabilities::new().consumes_tracks()),
        })];

        let warnings =
            validate_pipeline_phased(&pre, Some(&batch_caps), Some(StageId("batch")), &post);
        assert!(warnings.is_empty());
    }

    #[test]
    fn phased_batch_unsatisfied_dependency() {
        let pre: Vec<Box<dyn Stage>> = vec![];
        let batch_caps = StageCapabilities::new().consumes_detections();

        let warnings =
            validate_pipeline_phased(&pre, Some(&batch_caps), Some(StageId("batch")), &[]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0],
            ValidationWarning::UnsatisfiedDependency {
                stage_id: StageId("batch"),
                missing: "detections",
            }
        );
    }

    #[test]
    fn phased_post_batch_sees_batch_outputs() {
        let pre: Vec<Box<dyn Stage>> = vec![];
        let batch_caps = StageCapabilities::new().produces_detections();
        let post: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "trk",
            caps: Some(StageCapabilities::new().consumes_detections()),
        })];

        let warnings =
            validate_pipeline_phased(&pre, Some(&batch_caps), Some(StageId("batch")), &post);
        assert!(warnings.is_empty());
    }

    #[test]
    fn phased_duplicate_across_phases() {
        let pre: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "dup",
            caps: None,
        })];
        let post: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "dup",
            caps: None,
        })];

        let warnings = validate_pipeline_phased(&pre, None, None, &post);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            &warnings[0],
            ValidationWarning::DuplicateStageId { stage_id } if *stage_id == StageId("dup")
        ));
    }

    #[test]
    fn phased_duplicate_with_batch_id() {
        let pre: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "batch",
            caps: None,
        })];

        let warnings = validate_pipeline_phased(&pre, None, Some(StageId("batch")), &[]);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            &warnings[0],
            ValidationWarning::DuplicateStageId { stage_id } if *stage_id == StageId("batch")
        ));
    }

    #[test]
    fn phased_no_batch_processor() {
        let pre: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "det",
            caps: Some(StageCapabilities::new().produces_detections()),
        })];
        let post: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            name: "trk",
            caps: Some(StageCapabilities::new().consumes_detections()),
        })];

        let warnings = validate_pipeline_phased(&pre, None, None, &post);
        assert!(warnings.is_empty());
    }
}
