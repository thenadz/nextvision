//! The [`Stage`] trait — the core user-implementable perception contract.
//!
//! Stages are the primary extension point for adding perception capabilities
//! to the pipeline. Each stage processes one frame at a time and produces
//! structured output.
//!
//! # Design intent: one trait, composed pipelines
//!
//! The library intentionally uses a **single** `Stage` trait rather than
//! separate trait hierarchies for detection, tracking, classification, etc.
//! This keeps the abstraction minimal and avoids a taxonomy that would
//! either leak domain assumptions or force awkward categorizations.
//!
//! Instead, the pipeline _composes_ stages linearly: earlier stages write
//! fields into [`StageOutput`] that later stages read from
//! [`StageContext::artifacts`]. Specialization happens by convention (which
//! fields a stage populates), not by type hierarchy.
//!
//! # Supported model patterns
//!
//! The single-trait design naturally supports several model architectures:
//!
//! - **Classical detector → tracker**: a `FrameAnalysis` stage produces
//!   detections, then an `Association` stage consumes them and produces tracks.
//! - **Joint detection+tracking**: a single `FrameAnalysis` stage sets both
//!   `detections` and `tracks` in its [`StageOutput`]. No separate tracker
//!   stage is needed.
//! - **Direct track emitter**: a stage produces `tracks` without intermediate
//!   detections. Set `detection_id` to `None` on each [`TrackObservation`].
//! - **Richer observations**: per-observation metadata (embeddings, features,
//!   model-specific scores) is stored in [`TrackObservation::metadata`].
//!   Per-track metadata goes in [`Track::metadata`]. Per-frame shared data
//!   goes in [`StageOutput::artifacts`].
//!
//! # What a stage should do
//!
//! - Process a single frame and return structured results.
//! - Read upstream artifacts from [`StageContext::artifacts`].
//! - Read temporal history from [`StageContext::temporal`].
//! - Populate only the [`StageOutput`] fields it owns.
//! - Remain stateless across feeds — internal state is per-feed.
//!
//! # What a stage should NOT do
//!
//! - Block on network I/O (manage async bridges internally).
//! - Mutate shared global state.
//! - Produce side-channel output bypassing [`StageOutput`].
//! - Depend on stage execution order beyond what upstream artifacts provide.
//! - Accumulate unbounded internal state (use the temporal store instead).

use crate::artifact::PerceptionArtifacts;
use crate::detection::DetectionSet;
use crate::scene::SceneFeature;
use crate::signal::DerivedSignal;
use crate::temporal_access::TemporalAccess;
use crate::track::Track;
use nv_core::TypedMetadata;
use nv_core::error::StageError;
use nv_core::id::{FeedId, StageId};
use nv_core::metrics::StageMetrics;
use nv_frame::FrameEnvelope;
use nv_view::{ViewEpoch, ViewSnapshot};

/// Optional category hint for a perception stage.
///
/// Does not affect execution order or behavior — the pipeline executor
/// treats all stages uniformly. Categories serve as:
///
/// - **Documentation** — makes pipeline composition self-describing.
/// - **Metrics grouping** — category-aware dashboards and provenance.
/// - **Composition validation** — pipeline builders can warn about
///   unusual orderings (e.g., a tracker before a detector).
///
/// A stage may produce **any combination** of output fields regardless of
/// its declared category. Categories describe the stage's *primary role*,
/// not a hard constraint on what it can output. For example, a joint
/// detection+tracking model that reads pixels and produces both detections
/// and tracks in one forward pass is best categorized as `FrameAnalysis`.
///
/// Stages report their category via [`Stage::category()`], which defaults
/// to [`StageCategory::Custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageCategory {
    /// Reads frame pixel data, produces perception artifacts.
    ///
    /// This is the natural category for any model that takes raw pixels
    /// as primary input, regardless of what it produces:
    ///
    /// - Classical object detectors (outputs: detections)
    /// - Joint detection+tracking models (outputs: detections + tracks)
    /// - Direct track-emitting models (outputs: tracks only)
    /// - Feature extractors, scene classifiers (outputs: scene features)
    /// - Background subtractors (outputs: detections or scene features)
    FrameAnalysis,
    /// Reads upstream artifacts (detections, features) and produces tracks.
    ///
    /// Use this for stages whose primary role is *association* — matching
    /// observations across time — when the tracking logic is separate from
    /// the pixel-reading model.
    ///
    /// Examples: multi-object tracker, re-identification matcher.
    Association,
    /// Reads temporal state and accumulated artifacts, produces derived
    /// signals or scene-level features.
    ///
    /// Examples: trajectory analyzer, anomaly detector, dwell-time estimator.
    TemporalAnalysis,
    /// Reads accumulated artifacts and performs side-effect output.
    ///
    /// Returns empty [`StageOutput`] — does not modify the artifact accumulator.
    /// Examples: structured logger, metric exporter, event publisher.
    Sink,
    /// Uncategorized or multi-purpose stage.
    Custom,
}

/// Declares what artifact types a stage produces and consumes.
///
/// Used by [`StagePipeline::validate()`](crate::StagePipeline::validate) to
/// detect unsatisfied dependencies (e.g., a tracker that consumes detections
/// placed before the detector that produces them).
///
/// Stages report capabilities via [`Stage::capabilities()`]. The default
/// implementation returns `None`, meaning the stage opts out of validation.
///
/// # Validated fields
///
/// Currently, [`validate_stages()`](crate::validate_stages) checks:
/// - `consumes_detections` / `produces_detections`
/// - `consumes_tracks` / `produces_tracks`
///
/// The remaining fields (`consumes_temporal`, `produces_signals`,
/// `produces_scene_features`) are informational — they are available
/// for external tooling and future validation but are not enforced
/// by the built-in validator.
///
/// # Example
///
/// ```
/// use nv_perception::StageCapabilities;
///
/// let caps = StageCapabilities::new()
///     .consumes_detections()
///     .produces_tracks();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StageCapabilities {
    /// Stage reads detections from the artifact accumulator.
    pub consumes_detections: bool,
    /// Stage reads tracks from the artifact accumulator.
    pub consumes_tracks: bool,
    /// Stage reads temporal state.
    pub consumes_temporal: bool,
    /// Stage produces detections.
    pub produces_detections: bool,
    /// Stage produces tracks.
    pub produces_tracks: bool,
    /// Stage produces signals.
    pub produces_signals: bool,
    /// Stage produces scene features.
    pub produces_scene_features: bool,
}

impl StageCapabilities {
    /// Create empty capabilities (nothing consumed or produced).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark this stage as consuming detections.
    #[must_use]
    pub fn consumes_detections(mut self) -> Self {
        self.consumes_detections = true;
        self
    }

    /// Mark this stage as consuming tracks.
    #[must_use]
    pub fn consumes_tracks(mut self) -> Self {
        self.consumes_tracks = true;
        self
    }

    /// Mark this stage as consuming temporal state.
    #[must_use]
    pub fn consumes_temporal(mut self) -> Self {
        self.consumes_temporal = true;
        self
    }

    /// Mark this stage as producing detections.
    #[must_use]
    pub fn produces_detections(mut self) -> Self {
        self.produces_detections = true;
        self
    }

    /// Mark this stage as producing tracks.
    #[must_use]
    pub fn produces_tracks(mut self) -> Self {
        self.produces_tracks = true;
        self
    }

    /// Mark this stage as producing signals.
    #[must_use]
    pub fn produces_signals(mut self) -> Self {
        self.produces_signals = true;
        self
    }

    /// Mark this stage as producing scene features.
    #[must_use]
    pub fn produces_scene_features(mut self) -> Self {
        self.produces_scene_features = true;
        self
    }
}

/// Context provided to every stage invocation.
///
/// Contains the current frame, accumulated artifacts from prior stages,
/// read-only view and temporal snapshots,  and stage-level metrics.
///
/// All references are valid for the duration of the `process()` call.
pub struct StageContext<'a> {
    /// The feed this frame belongs to.
    pub feed_id: FeedId,
    /// The current video frame.
    pub frame: &'a FrameEnvelope,
    /// Accumulated outputs of all prior stages for this frame.
    pub artifacts: &'a PerceptionArtifacts,
    /// View-state snapshot for this frame.
    pub view: &'a ViewSnapshot,
    /// Read-only temporal state snapshot.
    ///
    /// Provides typed access to track histories, observation windows, and
    /// view-epoch context. Implemented by `nv_temporal::TemporalStoreSnapshot`.
    pub temporal: &'a dyn TemporalAccess,
    /// Performance metrics for this stage.
    pub metrics: &'a StageMetrics,
}

/// Output returned by a stage's `process()` method.
///
/// Each field is optional — a stage only sets the fields it produces.
/// The pipeline executor merges this into the [`PerceptionArtifacts`] accumulator.
///
/// # Joint-model and direct-track-emitter patterns
///
/// A stage is not limited to producing a single artifact type. Common
/// patterns beyond classical detection:
///
/// - **Joint det+track model**: set both `detections` and `tracks` in a
///   single `StageOutput`. The executor merges both into the accumulator.
///   Tracks produced this way can carry per-observation metadata via
///   [`TrackObservation::metadata`](crate::TrackObservation::metadata).
/// - **Direct track emitter**: set only `tracks` (leave `detections` as
///   `None`). No intermediate `DetectionSet` is required. Set
///   `TrackObservation::detection_id` to `None` since there are no
///   upstream detections to reference.
/// - **Richer observations**: stages that produce typed artifacts
///   (embeddings, feature maps, attention weights) can store them in
///   [`TrackObservation::metadata`](crate::TrackObservation::metadata)
///   (per-observation) or [`Track::metadata`](crate::Track::metadata)
///   (per-track), or pass them as `artifacts` (per-frame typed metadata)
///   for downstream stage consumption.
#[derive(Clone, Debug, Default)]
pub struct StageOutput {
    /// New or updated detection set.
    ///
    /// If `Some`, **replaces** the current detection set in the accumulator.
    /// If `None`, the previous detection set is kept.
    pub detections: Option<DetectionSet>,

    /// New or updated track set.
    ///
    /// `Some(Vec<Track>)` is **authoritative** for this frame: it replaces
    /// the current track set in the accumulator and marks the output as
    /// authoritative. Previously-known tracks absent from this set are
    /// treated as normally ended (`TrackEnded`) by the temporal store.
    ///
    /// `None` means this stage does not produce tracks — the previous
    /// track set is kept and authoritativeness is unchanged.
    pub tracks: Option<Vec<Track>>,

    /// Derived signals — always **appended** to the accumulator.
    pub signals: Vec<DerivedSignal>,

    /// Scene-level features — always **appended** to the accumulator.
    pub scene_features: Vec<SceneFeature>,

    /// Typed artifacts for downstream stages — **merged** (last-writer-wins by `TypeId`).
    ///
    /// This is the primary extension seam for inter-stage communication
    /// beyond the built-in fields. Any `Clone + Send + Sync + 'static`
    /// value can be stored here by type, and downstream stages retrieve
    /// it from [`StageContext::artifacts.stage_artifacts`](crate::PerceptionArtifacts::stage_artifacts).
    ///
    /// Example uses:
    /// - Feature maps or embeddings from a detector for a downstream classifier.
    /// - Prepared multi-frame input tensors for a temporal/clip-based model.
    /// - Confidence calibration metadata from an upstream stage.
    pub artifacts: TypedMetadata,
}

impl StageOutput {
    /// Create an empty stage output (no detections, tracks, signals, or artifacts).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a stage output containing only detections.
    #[must_use]
    pub fn with_detections(detections: DetectionSet) -> Self {
        Self {
            detections: Some(detections),
            ..Self::default()
        }
    }

    /// Create a stage output containing only tracks.
    #[must_use]
    pub fn with_tracks(tracks: Vec<Track>) -> Self {
        Self {
            tracks: Some(tracks),
            ..Self::default()
        }
    }

    /// Create a stage output containing only signals.
    #[must_use]
    pub fn with_signals(signals: Vec<DerivedSignal>) -> Self {
        Self {
            signals,
            ..Self::default()
        }
    }

    /// Create a stage output containing a single signal.
    #[must_use]
    pub fn with_signal(signal: DerivedSignal) -> Self {
        Self {
            signals: vec![signal],
            ..Self::default()
        }
    }

    /// Create a stage output containing only scene features.
    #[must_use]
    pub fn with_scene_features(features: Vec<SceneFeature>) -> Self {
        Self {
            scene_features: features,
            ..Self::default()
        }
    }

    /// Create a stage output containing a single typed artifact.
    ///
    /// This is useful for stages that produce a single custom artifact
    /// type for downstream consumption.
    #[must_use]
    pub fn with_artifact<T: Clone + Send + Sync + 'static>(value: T) -> Self {
        let mut artifacts = TypedMetadata::new();
        artifacts.insert(value);
        Self {
            artifacts,
            ..Self::default()
        }
    }

    /// Start building a [`StageOutput`] incrementally.
    ///
    /// # Example
    ///
    /// ```
    /// use nv_perception::StageOutput;
    ///
    /// let output = StageOutput::build()
    ///     .detections(Default::default())
    ///     .artifact(42_u32)
    ///     .finish();
    /// ```
    #[must_use]
    pub fn build() -> StageOutputBuilder {
        StageOutputBuilder {
            inner: Self::default(),
        }
    }
}

/// Incremental builder for [`StageOutput`].
///
/// Created via [`StageOutput::build()`]. Each setter returns `self` for chaining.
pub struct StageOutputBuilder {
    inner: StageOutput,
}

impl StageOutputBuilder {
    /// Set the detection set.
    #[must_use]
    pub fn detections(mut self, detections: DetectionSet) -> Self {
        self.inner.detections = Some(detections);
        self
    }

    /// Set the track set.
    #[must_use]
    pub fn tracks(mut self, tracks: Vec<Track>) -> Self {
        self.inner.tracks = Some(tracks);
        self
    }

    /// Append a signal.
    #[must_use]
    pub fn signal(mut self, signal: DerivedSignal) -> Self {
        self.inner.signals.push(signal);
        self
    }

    /// Append signals.
    #[must_use]
    pub fn signals(mut self, signals: Vec<DerivedSignal>) -> Self {
        self.inner.signals.extend(signals);
        self
    }

    /// Append a scene feature.
    #[must_use]
    pub fn scene_feature(mut self, feature: SceneFeature) -> Self {
        self.inner.scene_features.push(feature);
        self
    }

    /// Insert a typed artifact.
    #[must_use]
    pub fn artifact<T: Clone + Send + Sync + 'static>(mut self, value: T) -> Self {
        self.inner.artifacts.insert(value);
        self
    }

    /// Consume the builder and produce the [`StageOutput`].
    #[must_use]
    pub fn finish(self) -> StageOutput {
        self.inner
    }
}

/// The core user-implementable perception trait.
///
/// This is the **only** extension point for adding perception logic to the
/// pipeline. Stages run in a fixed linear order; each stage sees the
/// accumulated [`PerceptionArtifacts`](super::PerceptionArtifacts) from all
/// prior stages.
///
/// All methods take `&mut self`. The executor holds exclusive ownership of each
/// stage on the feed's dedicated OS thread — stages are never shared across
/// threads or called concurrently within a feed.
///
/// Requires `Send + 'static` (moved onto the feed thread at startup).
/// Does **not** require `Sync` — stages do not need to be shareable.
///
/// # Lifecycle
///
/// 1. `on_start()` — called once when the feed starts.
/// 2. `process()` — called once per frame, in order.
/// 3. `on_view_epoch_change()` — called when the camera view changes significantly.
/// 4. `on_stop()` — called once when the feed stops.
///
/// # Error handling
///
/// - `process()` returning `Err(StageError)` drops that frame and emits a health event.
///   The feed continues processing subsequent frames.
/// - `on_start()` returning `Err` prevents the feed from starting.
/// - A panic in `process()` is caught; the feed restarts per its restart policy.
///
/// # Supported model patterns
///
/// The library supports multiple perception model patterns through the
/// same `Stage` trait. A stage may produce *any combination* of output
/// fields — the pipeline does not enforce a single pattern.
///
/// | Pattern | Reads | Writes | Category |
/// |---|---|---|---|
/// | Classical detector | frame pixels | `detections` | `FrameAnalysis` |
/// | Classical tracker | `detections`, temporal | `tracks` | `Association` |
/// | Joint det+track model | frame pixels, temporal | `detections` + `tracks` | `FrameAnalysis` |
/// | Direct track emitter | frame pixels | `tracks` (skip detections) | `FrameAnalysis` |
/// | Classifier / enricher | `detections` or `tracks` | `artifacts` (typed metadata) | `Custom` |
/// | Scene analysis | frame pixels, temporal | `scene_features`, `signals` | `TemporalAnalysis` |
///
/// These are conventions, not enforced constraints. A stage may read and write
/// any combination of fields.
pub trait Stage: Send + 'static {
    /// Unique name for this stage (used in provenance, metrics, and logging).
    ///
    /// Must be a compile-time `&'static str`. Each stage in a pipeline should
    /// have a distinct ID.
    fn id(&self) -> StageId;

    /// Process one frame.
    ///
    /// **Must not block on I/O.** Stages wrapping inference servers should
    /// manage their own connection pool or async bridge internally.
    ///
    /// Called on the feed's dedicated stage-execution thread.
    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError>;

    /// Called once when the feed starts.
    ///
    /// Allocate GPU resources, load models, open connections here.
    /// Returning `Err` prevents the feed from starting.
    fn on_start(&mut self) -> Result<(), StageError> {
        Ok(())
    }

    /// Called once on feed shutdown.
    ///
    /// Release resources here. Best-effort — errors are logged but not fatal.
    fn on_stop(&mut self) -> Result<(), StageError> {
        Ok(())
    }

    /// Called when the view epoch changes (significant camera motion detected).
    ///
    /// Stages that maintain internal state dependent on camera view
    /// (background models, local maps, feature caches) should reset or
    /// adapt here. The `new_epoch` value identifies the new view epoch.
    fn on_view_epoch_change(&mut self, _new_epoch: ViewEpoch) -> Result<(), StageError> {
        Ok(())
    }

    /// Optional category hint for this stage.
    ///
    /// Defaults to [`StageCategory::Custom`]. Override to enable
    /// composition validation and category-aware metrics.
    fn category(&self) -> StageCategory {
        StageCategory::Custom
    }

    /// Declare this stage's input/output capabilities for pipeline validation.
    ///
    /// Returns `None` by default, opting the stage out of dependency
    /// validation. Override to enable
    /// [`StagePipeline::validate()`](crate::StagePipeline::validate).
    fn capabilities(&self) -> Option<StageCapabilities> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_output_with_artifact() {
        #[derive(Clone, Debug, PartialEq)]
        struct CustomScore(f64);

        let output = StageOutput::with_artifact(CustomScore(0.42));
        assert!(output.detections.is_none());
        assert!(output.tracks.is_none());
        assert_eq!(
            output.artifacts.get::<CustomScore>(),
            Some(&CustomScore(0.42))
        );
    }

    #[test]
    fn stage_output_builder() {
        #[derive(Clone, Debug, PartialEq)]
        struct Tag(u32);

        let dets = DetectionSet::empty();
        let output = StageOutput::build()
            .detections(dets)
            .artifact(Tag(7))
            .finish();

        assert!(output.detections.is_some());
        assert_eq!(output.artifacts.get::<Tag>(), Some(&Tag(7)));
    }

    #[test]
    fn stage_capabilities_builder() {
        let caps = StageCapabilities::new()
            .consumes_detections()
            .produces_tracks();

        assert!(caps.consumes_detections);
        assert!(!caps.consumes_tracks);
        assert!(caps.produces_tracks);
        assert!(!caps.produces_detections);
    }
}
