//! Configurable mock [`Stage`] implementations for testing.

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_core::{BBox, DetectionId, MonotonicTs, TrackId};
use nv_perception::track::{Track, TrackObservation};
use nv_perception::{
    Detection, DetectionSet, DerivedSignal, SignalValue, Stage, StageCategory, StageContext,
    StageOutput, TemporalAccess,
};
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

    fn track_ids(&self) -> Vec<TrackId> {
        Vec::new()
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

    fn trajectory_point_count(&self, _id: &TrackId) -> usize {
        0
    }

    fn trajectory_segment_count(&self, _id: &TrackId) -> usize {
        0
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

/// A mock stage that panics on every `process()` call.
///
/// Used to test panic-catching and restart behavior in the runtime.
pub struct PanicStage {
    id: StageId,
}

impl PanicStage {
    /// Create a panic stage with the given name.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self { id: StageId(name) }
    }
}

impl Stage for PanicStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        panic!("intentional test panic in stage {:?}", self.id);
    }
}

// ---------------------------------------------------------------------------
// MockDetectorStage — generates configurable detections each frame
// ---------------------------------------------------------------------------

/// A mock detector stage that generates a configurable number of detections.
///
/// Each detection gets a unique `DetectionId`, a fixed `class_id`, and a
/// bounding box spread evenly across the frame width.
pub struct MockDetectorStage {
    id: StageId,
    num_detections: usize,
    class_id: u32,
    confidence: f32,
    next_det_id: u64,
}

impl MockDetectorStage {
    /// Create a mock detector that generates `num_detections` per frame.
    #[must_use]
    pub fn new(name: &'static str, num_detections: usize) -> Self {
        Self {
            id: StageId(name),
            num_detections,
            class_id: 0,
            confidence: 0.9,
            next_det_id: 0,
        }
    }

    /// Set the class ID for generated detections.
    #[must_use]
    pub fn with_class_id(mut self, class_id: u32) -> Self {
        self.class_id = class_id;
        self
    }

    /// Set the confidence for generated detections.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

impl Stage for MockDetectorStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn category(&self) -> StageCategory {
        StageCategory::FrameAnalysis
    }

    fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        let mut detections = Vec::with_capacity(self.num_detections);
        let step = if self.num_detections > 0 {
            1.0_f32 / self.num_detections as f32
        } else {
            1.0_f32
        };

        for i in 0..self.num_detections {
            let x = i as f32 * step;
            detections.push(Detection {
                id: DetectionId::new(self.next_det_id),
                class_id: self.class_id,
                confidence: self.confidence,
                bbox: BBox::new(x, 0.0, x + step, 0.5),
                embedding: None,
                metadata: nv_core::TypedMetadata::new(),
            });
            self.next_det_id += 1;
        }

        Ok(StageOutput::with_detections(DetectionSet { detections }))
    }
}

// ---------------------------------------------------------------------------
// MockTrackerStage — maps detections 1:1 to tracks
// ---------------------------------------------------------------------------

/// A mock tracker / association stage.
///
/// Reads the detection set from upstream artifacts and maps each detection
/// to a track 1:1. Maintains a simple monotonic track ID counter (no real
/// re-identification or association logic).
pub struct MockTrackerStage {
    id: StageId,
    next_track_id: u64,
}

impl MockTrackerStage {
    /// Create a mock tracker stage.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            id: StageId(name),
            next_track_id: 0,
        }
    }
}

impl Stage for MockTrackerStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn category(&self) -> StageCategory {
        StageCategory::Association
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        let ts = ctx.frame.ts();
        let tracks: Vec<Track> = ctx
            .artifacts
            .detections
            .detections
            .iter()
            .map(|det| {
                let id = TrackId::new(self.next_track_id);
                self.next_track_id += 1;
                Track {
                    id,
                    class_id: det.class_id,
                    state: nv_perception::TrackState::Confirmed,
                    current: TrackObservation {
                        ts,
                        bbox: det.bbox,
                        confidence: det.confidence,
                        state: nv_perception::TrackState::Confirmed,
                        detection_id: Some(det.id),
                    },
                    metadata: nv_core::TypedMetadata::new(),
                }
            })
            .collect();

        Ok(StageOutput::with_tracks(tracks))
    }
}

// ---------------------------------------------------------------------------
// MockTemporalStage — reads temporal state, produces signals
// ---------------------------------------------------------------------------

/// A mock temporal analysis stage.
///
/// Reads the temporal store's active track count and produces a
/// `"track_count"` scalar signal. Demonstrates how stages can use
/// temporal state for higher-level analysis.
pub struct MockTemporalStage {
    id: StageId,
}

impl MockTemporalStage {
    /// Create a mock temporal analysis stage.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self { id: StageId(name) }
    }
}

impl Stage for MockTemporalStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn category(&self) -> StageCategory {
        StageCategory::TemporalAnalysis
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        let track_count = ctx.temporal.track_count();
        let signal = DerivedSignal {
            name: "track_count",
            value: SignalValue::Scalar(track_count as f64),
            ts: ctx.frame.ts(),
        };
        Ok(StageOutput::with_signal(signal))
    }
}

// ---------------------------------------------------------------------------
// CountingSinkStage — counts processed frames and captures artifact stats
// ---------------------------------------------------------------------------

/// A mock sink stage that counts how many frames it has processed and
/// records the last-seen detection and track counts.
///
/// Useful for verifying that the full pipeline executed and upstream
/// artifacts propagated correctly.
pub struct CountingSinkStage {
    id: StageId,
    /// Number of frames processed.
    pub frames_seen: u64,
    /// Number of detections seen on the last frame.
    pub last_detection_count: usize,
    /// Number of tracks seen on the last frame.
    pub last_track_count: usize,
    /// Number of signals seen on the last frame.
    pub last_signal_count: usize,
}

impl CountingSinkStage {
    /// Create a counting sink stage.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            id: StageId(name),
            frames_seen: 0,
            last_detection_count: 0,
            last_track_count: 0,
            last_signal_count: 0,
        }
    }
}

impl Stage for CountingSinkStage {
    fn id(&self) -> StageId {
        self.id
    }

    fn category(&self) -> StageCategory {
        StageCategory::Sink
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        self.frames_seen += 1;
        self.last_detection_count = ctx.artifacts.detections.len();
        self.last_track_count = ctx.artifacts.tracks.len();
        self.last_signal_count = ctx.artifacts.signals.len();
        // Sink stages return empty output — they observe, not produce.
        Ok(StageOutput::empty())
    }
}
