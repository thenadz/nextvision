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
#[derive(Clone, Debug, Default)]
pub struct PerceptionArtifacts {
    /// Current detection set (replaced by each stage that returns detections).
    pub detections: DetectionSet,
    /// Current track set (replaced by each stage that returns tracks).
    pub tracks: Vec<Track>,
    /// Accumulated signals from all stages.
    pub signals: Vec<DerivedSignal>,
    /// Scene-level features accumulated from all stages.
    pub scene_features: Vec<SceneFeature>,
    /// Typed artifacts from stages — keyed by `TypeId`, last-writer-wins.
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
        }
        self.signals.extend(output.signals);
        self.scene_features.extend(output.scene_features);
        self.stage_artifacts.merge(output.artifacts);
    }
}
