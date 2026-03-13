use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_perception::stage::{Stage, StageCapabilities, StageCategory, StageContext, StageOutput};
use nv_view::ViewEpoch;

use crate::config::TrackerConfig;
use crate::ocsort::OcSortTracker;

/// Sample multi-object tracker stage.
///
/// Consumes [`DetectionSet`](nv_perception::DetectionSet) from upstream
/// detector stages and produces [`Track`](nv_perception::Track) outputs.
///
/// This is a reference implementation using an observation-centric SORT
/// variant. Library users will typically substitute their own tracker.
///
/// # Usage
///
/// ```no_run
/// use nv_sample_tracking::{TrackerConfig, TrackerStage};
///
/// let stage = TrackerStage::new(TrackerConfig::default());
/// // Add to a StagePipeline after a detector stage.
/// ```
pub struct TrackerStage {
    tracker: OcSortTracker,
}

impl TrackerStage {
    const STAGE_ID: StageId = StageId("sample-tracker");

    /// Create a new tracker stage with the given configuration.
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            tracker: OcSortTracker::new(config),
        }
    }
}

impl Stage for TrackerStage {
    fn id(&self) -> StageId {
        Self::STAGE_ID
    }

    fn category(&self) -> StageCategory {
        StageCategory::Association
    }

    fn capabilities(&self) -> Option<StageCapabilities> {
        Some(
            StageCapabilities::new()
                .consumes_detections()
                .produces_tracks(),
        )
    }

    fn on_view_epoch_change(&mut self, _new_epoch: ViewEpoch) -> Result<(), StageError> {
        // Camera moved significantly — reset tracker state.
        self.tracker.reset();
        Ok(())
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        let detections = &ctx.artifacts.detections;
        let ts = ctx.frame.ts();
        let tracks = self.tracker.update(&detections.detections, ts);
        Ok(StageOutput::with_tracks(tracks))
    }
}
