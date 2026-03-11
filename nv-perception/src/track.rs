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

impl TrackObservation {
    /// Create a new observation with optional detection association.
    #[must_use]
    pub fn new(
        ts: MonotonicTs,
        bbox: BBox,
        confidence: f32,
        state: TrackState,
        detection_id: Option<DetectionId>,
    ) -> Self {
        Self {
            ts,
            bbox,
            confidence,
            state,
            detection_id,
        }
    }
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

impl Track {
    /// Create a new track with the given identity and current observation.
    ///
    /// Metadata starts empty — use the builder or set `metadata` directly
    /// to attach custom data.
    #[must_use]
    pub fn new(id: TrackId, class_id: u32, state: TrackState, current: TrackObservation) -> Self {
        Self {
            id,
            class_id,
            state,
            current,
            metadata: TypedMetadata::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::BBox;

    #[test]
    fn track_observation_new() {
        let obs = TrackObservation::new(
            MonotonicTs::from_nanos(1_000_000),
            BBox::new(0.1, 0.2, 0.3, 0.4),
            0.95,
            TrackState::Confirmed,
            Some(DetectionId::new(7)),
        );
        assert_eq!(obs.ts, MonotonicTs::from_nanos(1_000_000));
        assert!((obs.confidence - 0.95).abs() < f32::EPSILON);
        assert_eq!(obs.state, TrackState::Confirmed);
        assert_eq!(obs.detection_id, Some(DetectionId::new(7)));
    }

    #[test]
    fn track_new_has_empty_metadata() {
        let obs = TrackObservation::new(
            MonotonicTs::from_nanos(0),
            BBox::new(0.0, 0.0, 0.5, 0.5),
            0.9,
            TrackState::Tentative,
            None,
        );
        let track = Track::new(TrackId::new(1), 0, TrackState::Tentative, obs);
        assert_eq!(track.id, TrackId::new(1));
        assert_eq!(track.class_id, 0);
        assert!(track.metadata.is_empty());
    }
}
