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
use nv_core::error::StageError;
use nv_core::id::{FeedId, StageId};
use nv_core::metrics::StageMetrics;
use nv_core::TypedMetadata;
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
/// Stages report their category via [`Stage::category()`], which defaults
/// to [`StageCategory::Custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageCategory {
    /// Reads frame pixel data, produces detections and/or scene features.
    ///
    /// Examples: object detector, feature extractor, background subtractor.
    FrameAnalysis,
    /// Reads detections (and optionally temporal state), produces tracks.
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
/// # Typical stage categories (by convention, not by type)
///
/// | Category | Reads | Writes |
/// |---|---|---|
/// | Detection | frame pixels | `detections` |
/// | Tracking | `detections`, temporal | `tracks` |
/// | Classification | `detections` or `tracks` | `artifacts` (typed metadata) |
/// | Scene analysis | frame pixels, temporal | `scene_features`, `signals` |
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
}
