//! Stage execution harness for isolated testing.
//!
//! [`StageHarness`] drives a single [`Stage`] with synthetic inputs,
//! without needing the full runtime or media pipeline. Useful for
//! unit-testing stage logic in isolation.
//!
//! # Example
//!
//! ```rust,no_run
//! use nv_test_util::harness::StageHarness;
//! use nv_test_util::mock_stage::MockDetectorStage;
//! use nv_test_util::synthetic;
//! use nv_core::{FeedId, MonotonicTs};
//! use nv_perception::PerceptionArtifacts;
//!
//! let stage = MockDetectorStage::new("test_det", 3);
//! let mut harness = StageHarness::new(stage);
//! let frame = synthetic::solid_gray(FeedId::new(1), 0, MonotonicTs::ZERO, 32, 32, 128);
//! let output = harness.process(&frame, &PerceptionArtifacts::empty()).unwrap();
//! assert_eq!(output.detections.unwrap().len(), 3);
//! ```

use nv_core::error::StageError;
use nv_core::id::FeedId;
use nv_core::metrics::StageMetrics;
use nv_frame::FrameEnvelope;
use nv_perception::{PerceptionArtifacts, Stage, StageContext, StageOutput};
use nv_view::{ViewSnapshot, ViewState};

use crate::mock_stage::NullTemporalAccess;

/// A lightweight harness for testing a single [`Stage`] in isolation.
///
/// Provides synthetic view state and a null temporal snapshot so the
/// stage can be exercised without the full runtime.
pub struct StageHarness {
    stage: Box<dyn Stage>,
    feed_id: FeedId,
    metrics: StageMetrics,
    view_snapshot: ViewSnapshot,
    temporal: NullTemporalAccess,
}

impl StageHarness {
    /// Create a harness wrapping the given stage.
    pub fn new(stage: impl Stage) -> Self {
        Self {
            stage: Box::new(stage),
            feed_id: FeedId::new(1),
            metrics: StageMetrics {
                frames_processed: 0,
                errors: 0,
            },
            view_snapshot: ViewSnapshot::new(ViewState::fixed_initial()),
            temporal: NullTemporalAccess,
        }
    }

    /// Override the feed ID (default: `FeedId(1)`).
    #[must_use]
    pub fn with_feed_id(mut self, feed_id: FeedId) -> Self {
        self.feed_id = feed_id;
        self
    }

    /// Override the view snapshot.
    #[must_use]
    pub fn with_view(mut self, view: ViewSnapshot) -> Self {
        self.view_snapshot = view;
        self
    }

    /// Process one frame through the stage.
    ///
    /// The `artifacts` parameter simulates upstream stage outputs.
    /// Pass `PerceptionArtifacts::empty()` for a first-in-pipeline stage.
    pub fn process(
        &mut self,
        frame: &FrameEnvelope,
        artifacts: &PerceptionArtifacts,
    ) -> Result<StageOutput, StageError> {
        let ctx = StageContext {
            feed_id: self.feed_id,
            frame,
            artifacts,
            view: &self.view_snapshot,
            temporal: &self.temporal,
            metrics: &self.metrics,
        };
        let result = self.stage.process(&ctx);
        match &result {
            Ok(_) => self.metrics.frames_processed += 1,
            Err(_) => self.metrics.errors += 1,
        }
        result
    }

    /// Call `on_start()` on the wrapped stage.
    pub fn start(&mut self) -> Result<(), StageError> {
        self.stage.on_start()
    }

    /// Call `on_stop()` on the wrapped stage.
    pub fn stop(&mut self) -> Result<(), StageError> {
        self.stage.on_stop()
    }

    /// Access the stage's accumulated metrics.
    #[must_use]
    pub fn metrics(&self) -> &StageMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_stage::MockDetectorStage;
    use crate::synthetic;
    use nv_core::MonotonicTs;

    #[test]
    fn harness_runs_detector() {
        let mut harness = StageHarness::new(MockDetectorStage::new("det", 3));
        let frame =
            synthetic::solid_gray(FeedId::new(1), 0, MonotonicTs::from_nanos(0), 16, 16, 128);
        let output = harness
            .process(&frame, &PerceptionArtifacts::empty())
            .unwrap();
        let dets = output.detections.expect("should produce detections");
        assert_eq!(dets.len(), 3);
    }

    #[test]
    fn harness_tracks_metrics() {
        let mut harness = StageHarness::new(MockDetectorStage::new("det", 1));
        let frame = synthetic::solid_gray(FeedId::new(1), 0, MonotonicTs::from_nanos(0), 4, 4, 128);
        harness
            .process(&frame, &PerceptionArtifacts::empty())
            .unwrap();
        harness
            .process(&frame, &PerceptionArtifacts::empty())
            .unwrap();
        assert_eq!(harness.metrics().frames_processed, 2);
        assert_eq!(harness.metrics().errors, 0);
    }

    #[test]
    fn harness_counts_errors() {
        let mut harness = StageHarness::new(crate::mock_stage::FailingStage::new("bad"));
        let frame = synthetic::solid_gray(FeedId::new(1), 0, MonotonicTs::from_nanos(0), 4, 4, 128);
        let result = harness.process(&frame, &PerceptionArtifacts::empty());
        assert!(result.is_err());
        assert_eq!(harness.metrics().errors, 1);
    }
}
