//! Output types: [`OutputEnvelope`] and [`OutputSink`] trait.

use nv_core::id::FeedId;
use nv_core::timestamp::{MonotonicTs, WallTs};
use nv_core::TypedMetadata;
use nv_perception::{DerivedSignal, DetectionSet, SceneFeature, Track};
use nv_view::ViewState;

use crate::provenance::Provenance;

/// Structured output for one processed frame.
///
/// Contains the complete perception result, view state, and full provenance.
/// Delivered to the user via [`OutputSink::emit`].
#[derive(Debug)]
pub struct OutputEnvelope {
    /// Which feed produced this output.
    pub feed_id: FeedId,
    /// Monotonic frame sequence number.
    pub frame_seq: u64,
    /// Monotonic timestamp of the source frame.
    pub ts: MonotonicTs,
    /// Wall-clock timestamp of the source frame.
    pub wall_ts: WallTs,
    /// Final detection set after all stages.
    pub detections: DetectionSet,
    /// Final track set after all stages.
    pub tracks: Vec<Track>,
    /// All derived signals from all stages.
    pub signals: Vec<DerivedSignal>,
    /// Scene-level features from all stages.
    pub scene_features: Vec<SceneFeature>,
    /// View state at the time of this frame.
    pub view: ViewState,
    /// Full provenance: stage timings, view decisions, pipeline latency.
    pub provenance: Provenance,
    /// Extensible output metadata.
    pub metadata: TypedMetadata,
}

/// User-implementable trait: receives structured outputs from the pipeline.
///
/// `emit()` is called on a dedicated output thread per feed. It is
/// deliberately **not** async and **not** fallible:
///
/// - If the sink needs async I/O, it should buffer and channel internally.
/// - If the sink fails, it should log and drop — the perception pipeline
///   must never block on downstream consumption.
pub trait OutputSink: Send + Sync + 'static {
    /// Receive a processed output envelope.
    fn emit(&self, output: OutputEnvelope);
}
