//! Configurable mock [`Stage`] implementations for testing.

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_core::{MonotonicTs, TrackId};
use nv_perception::track::{Track, TrackObservation};
use nv_perception::{DetectionSet, Stage, StageContext, StageOutput, TemporalAccess};
use nv_view::ViewEpoch;

/// A [`TemporalAccess`] implementation that contains no tracks.
///
/// Useful for constructing [`StageContext`] in unit tests that do not
/// need temporal state.
pub struct NullTemporalAccess;

impl TemporalAccess for NullTemporalAccess {
    fn view_epoch(&self) -> ViewEpoch {
        ViewEpoch::INITIAL
    }

    fn track_count(&self) -> usize {
        0
    }

    fn get_track(&self, _id: &TrackId) -> Option<&Track> {
        None
    }

    fn recent_observations(&self, _id: &TrackId) -> &[TrackObservation] {
        &[]
    }

    fn first_seen(&self, _id: &TrackId) -> Option<MonotonicTs> {
        None
    }

    fn last_seen(&self, _id: &TrackId) -> Option<MonotonicTs> {
        None
    }
}

/// A mock stage that always returns an empty output.
///
/// Useful for testing pipeline plumbing without real perception logic.
pub struct NoOpStage {
    id: StageId,
}

impl NoOpStage {
    /// Create a no-op stage with the given name.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self { id: StageId(name) }
    }
}

impl Stage for NoOpStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        Ok(StageOutput::empty())
    }
}

/// A mock stage that returns a fixed detection set every frame.
pub struct FixedDetectionStage {
    id: StageId,
    detections: DetectionSet,
}

impl FixedDetectionStage {
    /// Create a stage that returns the given detections for every frame.
    #[must_use]
    pub fn new(name: &'static str, detections: DetectionSet) -> Self {
        Self {
            id: StageId(name),
            detections,
        }
    }
}

impl Stage for FixedDetectionStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        Ok(StageOutput {
            detections: Some(self.detections.clone()),
            ..StageOutput::empty()
        })
    }
}

/// A mock stage that fails on every frame with a configurable error.
pub struct FailingStage {
    id: StageId,
}

impl FailingStage {
    /// Create a stage that fails every frame.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self { id: StageId(name) }
    }
}

impl Stage for FailingStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        Err(StageError::ProcessingFailed {
            stage_id: self.id,
            detail: "mock failure".into(),
        })
    }
}
