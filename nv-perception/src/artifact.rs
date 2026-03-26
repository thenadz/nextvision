//! Accumulated perception artifacts and stage output merging.

use crate::detection::DetectionSet;
use crate::scene::SceneFeature;
use crate::signal::DerivedSignal;
use crate::track::Track;
use nv_core::TypedMetadata;

/// Accumulated outputs of all stages that have run so far for this frame.
///
/// Built up incrementally by the pipeline executor. Stage N+1 sees the
/// accumulated result of stages 0..N through [`StageContext::artifacts`](super::StageContext).
///
/// ## Merge semantics
///
/// | Field | Behavior |
/// |---|---|
/// | `detections` | **Replace** — latest `Some(DetectionSet)` wins |
/// | `tracks` | **Replace** — latest `Some(Vec<Track>)` wins |
/// | `signals` | **Append** — all signals accumulate |
/// | `scene_features` | **Append** — all scene features accumulate |
/// | `stage_artifacts` | **Merge** — last-writer-wins per `TypeId` |
///
/// ## Extension seam: `stage_artifacts`
///
/// The [`stage_artifacts`](Self::stage_artifacts) field is the primary
/// inter-stage communication channel for data that does not fit the
/// built-in fields. Any `Clone + Send + Sync + 'static` value can be
/// stored by type.
///
/// A pre-processing stage can assemble a sliding window of frames
/// (e.g., `Arc<[FrameEnvelope]>`) and store it as a typed artifact for a
/// downstream temporal or clip-based model to consume, without any changes
/// to the core pipeline execution model.
#[derive(Clone, Debug, Default)]
pub struct PerceptionArtifacts {
    /// Current detection set (replaced by each stage that returns detections).
    pub detections: DetectionSet,
    /// Current track set (replaced by each stage that returns tracks).
    ///
    /// When [`tracks_authoritative`](Self::tracks_authoritative) is `true`,
    /// this is the complete set of active tracks for the frame. Tracks
    /// previously known to the temporal store but absent here are considered
    /// normally ended (`TrackEnded`).
    pub tracks: Vec<Track>,
    /// Whether any stage produced authoritative track output this frame.
    ///
    /// Set to `true` when at least one stage returns `Some(tracks)` in its
    /// [`StageOutput`](super::StageOutput). When `false`, no stage claimed
    /// ownership of the track set, and the executor must **not** infer
    /// track endings from the (default-empty) `tracks` field.
    pub tracks_authoritative: bool,
    /// Accumulated signals from all stages.
    pub signals: Vec<DerivedSignal>,
    /// Scene-level features accumulated from all stages.
    pub scene_features: Vec<SceneFeature>,
    /// Typed artifacts from stages — keyed by `TypeId`, last-writer-wins.
    ///
    /// This is the extension seam for arbitrary inter-stage data: feature
    /// maps, prepared input tensors, multi-frame windows, calibration
    /// metadata, or any domain-specific payload. Downstream stages access
    /// stored values via `StageContext::artifacts.stage_artifacts.get::<T>()`.
    pub stage_artifacts: TypedMetadata,
}

impl PerceptionArtifacts {
    /// Create empty perception artifacts.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Merge a [`StageOutput`](super::StageOutput) into the accumulator.
    ///
    /// Applies the merge semantics documented above.
    pub fn merge(&mut self, output: super::StageOutput) {
        if let Some(detections) = output.detections {
            self.detections = detections;
        }
        if let Some(tracks) = output.tracks {
            self.tracks = tracks;
            self.tracks_authoritative = true;
        }
        self.signals.extend(output.signals);
        self.scene_features.extend(output.scene_features);
        self.stage_artifacts.merge(output.artifacts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::Detection;
    use crate::signal::{DerivedSignal, SignalValue};
    use crate::stage::StageOutput;
    use crate::track::Track;
    use nv_core::id::{DetectionId, TrackId};
    use nv_core::{BBox, MonotonicTs, TypedMetadata};

    fn make_detection(id: u64) -> Detection {
        Detection::builder(DetectionId::new(id), 0, 0.9, BBox::new(0.1, 0.2, 0.3, 0.4)).build()
    }

    fn make_track(id: u64) -> Track {
        use crate::track::{TrackObservation, TrackState};
        let obs = TrackObservation::new(
            MonotonicTs::from_nanos(0),
            BBox::new(0.1, 0.2, 0.3, 0.4),
            0.9,
            TrackState::Confirmed,
            None,
        );
        Track::new(TrackId::new(id), 0, TrackState::Confirmed, obs)
    }

    fn make_signal(name: &'static str) -> DerivedSignal {
        DerivedSignal {
            name,
            value: SignalValue::Scalar(1.0),
            ts: MonotonicTs::from_nanos(0),
        }
    }

    #[test]
    fn merge_empty_output_is_noop() {
        let mut arts = PerceptionArtifacts::empty();
        arts.merge(StageOutput::empty());

        assert!(arts.detections.is_empty());
        assert!(arts.tracks.is_empty());
        assert!(!arts.tracks_authoritative);
        assert!(arts.signals.is_empty());
        assert!(arts.scene_features.is_empty());
    }

    #[test]
    fn merge_detections_replace() {
        let mut arts = PerceptionArtifacts::empty();

        // First stage sets detections.
        let dets1 = DetectionSet::from(vec![make_detection(1), make_detection(2)]);
        arts.merge(StageOutput::with_detections(dets1));
        assert_eq!(arts.detections.len(), 2);

        // Second stage replaces with a single detection.
        let dets2 = DetectionSet::from(vec![make_detection(3)]);
        arts.merge(StageOutput::with_detections(dets2));
        assert_eq!(arts.detections.len(), 1);
        assert_eq!(arts.detections.detections[0].id, DetectionId::new(3));
    }

    #[test]
    fn merge_none_detections_preserves_existing() {
        let mut arts = PerceptionArtifacts::empty();

        let dets = DetectionSet::from(vec![make_detection(1)]);
        arts.merge(StageOutput::with_detections(dets));
        assert_eq!(arts.detections.len(), 1);

        // Merge output with no detections — previous set preserved.
        arts.merge(StageOutput::empty());
        assert_eq!(arts.detections.len(), 1);
    }

    #[test]
    fn merge_tracks_replace_and_set_authoritative() {
        let mut arts = PerceptionArtifacts::empty();
        assert!(!arts.tracks_authoritative);

        let tracks = vec![make_track(1), make_track(2)];
        arts.merge(StageOutput::with_tracks(tracks));
        assert_eq!(arts.tracks.len(), 2);
        assert!(arts.tracks_authoritative);

        // Replace with fewer tracks.
        let tracks2 = vec![make_track(3)];
        arts.merge(StageOutput::with_tracks(tracks2));
        assert_eq!(arts.tracks.len(), 1);
        assert_eq!(arts.tracks[0].id, TrackId::new(3));
    }

    #[test]
    fn merge_signals_append() {
        let mut arts = PerceptionArtifacts::empty();

        arts.merge(StageOutput::with_signal(make_signal("sig_a")));
        assert_eq!(arts.signals.len(), 1);

        arts.merge(StageOutput::with_signal(make_signal("sig_b")));
        assert_eq!(arts.signals.len(), 2);
        assert_eq!(arts.signals[0].name, "sig_a");
        assert_eq!(arts.signals[1].name, "sig_b");
    }

    #[test]
    fn merge_stage_artifacts_last_writer_wins() {
        #[derive(Clone, Debug, PartialEq)]
        struct MyData(u32);

        let mut arts = PerceptionArtifacts::empty();

        let mut meta1 = TypedMetadata::new();
        meta1.insert(MyData(1));
        arts.merge(StageOutput {
            artifacts: meta1,
            ..StageOutput::default()
        });
        assert_eq!(arts.stage_artifacts.get::<MyData>(), Some(&MyData(1)));

        // Second merge overwrites.
        let mut meta2 = TypedMetadata::new();
        meta2.insert(MyData(42));
        arts.merge(StageOutput {
            artifacts: meta2,
            ..StageOutput::default()
        });
        assert_eq!(arts.stage_artifacts.get::<MyData>(), Some(&MyData(42)));
    }
}
