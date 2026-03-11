use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_perception::stage::{Stage, StageCategory, StageContext, StageOutput};
use nv_view::ViewEpoch;

use crate::config::OcSortConfig;
use crate::ocsort::OcSortTracker;

/// OC-SORT multi-object tracker stage.
///
/// Consumes [`DetectionSet`](nv_perception::DetectionSet) from upstream
/// detector stages and produces [`Track`](nv_perception::Track) outputs
/// using the OC-SORT algorithm.
///
/// # Usage
///
/// ```no_run
/// use nv_byo_ocsort::{OcSortConfig, OcSortStage};
///
/// let stage = OcSortStage::new(OcSortConfig::default());
/// // Add to a StagePipeline after a detector stage.
/// ```
pub struct OcSortStage {
    tracker: OcSortTracker,
}

impl OcSortStage {
    const STAGE_ID: StageId = StageId("byo-ocsort");

    /// Create a new OC-SORT tracker stage with the given configuration.
    pub fn new(config: OcSortConfig) -> Self {
        Self {
            tracker: OcSortTracker::new(config),
        }
    }
}

impl Stage for OcSortStage {
    fn id(&self) -> StageId {
        Self::STAGE_ID
    }

    fn category(&self) -> StageCategory {
        StageCategory::Association
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
