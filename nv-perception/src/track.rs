//! Track types — tracked objects across frames.

use nv_core::{BBox, DetectionId, MonotonicTs, TrackId, TypedMetadata};

/// Lifecycle state of a tracked object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackState {
    /// Track has been initialized but not yet confirmed by repeated observations.
    Tentative,
    /// Track has been confirmed by multiple consistent observations.
    Confirmed,
    /// No observation this frame — position is predicted (coasted).
    Coasted,
    /// Coasted too long — pending deletion by the temporal store.
    Lost,
}

/// One observation of a track in a single frame.
///
/// Records the spatial and temporal state at the moment of observation.
#[derive(Clone, Debug)]
pub struct TrackObservation {
    /// Timestamp of this observation.
    pub ts: MonotonicTs,
    /// Bounding box in normalized coordinates.
    pub bbox: BBox,
    /// Confidence score for this observation.
    pub confidence: f32,
    /// Track state at time of observation.
    pub state: TrackState,
    /// The detection that was associated with this track, if any.
    /// `None` when the track is coasting (no matching detection).
    pub detection_id: Option<DetectionId>,
}

/// A live tracked object.
///
/// Produced by tracker stages. The `current` field holds the latest observation.
/// Historical observations are managed by the temporal store.
#[derive(Clone, Debug)]
pub struct Track {
    /// Unique track identifier within this feed session.
    pub id: TrackId,
    /// Numeric class identifier (from the associated detections).
    pub class_id: u32,
    /// Current lifecycle state.
    pub state: TrackState,
    /// Most recent observation.
    pub current: TrackObservation,
    /// Extensible metadata (re-id features, custom scores, etc.).
    pub metadata: TypedMetadata,
}
