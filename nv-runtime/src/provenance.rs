//! Provenance types — audit trail for stage and view-system decisions.

use nv_core::id::StageId;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_view::view_state::{ViewEpoch, ViewVersion};
use nv_view::{EpochDecision, MotionSource, TransitionPhase};

/// Full provenance for one processed frame.
///
/// Every output carries this, making production debugging tractable.
///
/// # Timing semantics
///
/// All timestamps are [`MonotonicTs`] values — they share the same monotonic
/// clock domain as [`FrameEnvelope::ts()`](nv_frame::FrameEnvelope::ts).
///
/// - `frame_receive_ts` — when the frame was dequeued from the bounded queue.
/// - `pipeline_complete_ts` — when all stages finished and output was constructed.
/// - Per-stage `start_ts` / `end_ts` — real wall-clock offsets converted to
///   monotonic nanoseconds from the pipeline epoch for ordering consistency.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// Per-stage provenance records, in execution order.
    pub stages: Vec<StageProvenance>,
    /// View-system provenance for this frame.
    pub view_provenance: ViewProvenance,
    /// Timestamp when the frame was dequeued from the bounded queue
    /// (start of pipeline processing for this frame).
    pub frame_receive_ts: MonotonicTs,
    /// Timestamp when the pipeline completed processing this frame.
    pub pipeline_complete_ts: MonotonicTs,
    /// Total pipeline latency (receive → complete).
    pub total_latency: Duration,
    /// Wall-clock age of the frame at processing time.
    ///
    /// Computed as `WallTs::now() - frame.wall_ts()` when the frame
    /// enters pipeline processing. A large value indicates the frame
    /// was stale before the executor even touched it — typically due
    /// to buffer-pool starvation, TCP backlog, or slow decode.
    ///
    /// `None` if wall-clock age could not be determined (e.g., if the
    /// frame's wall timestamp is in the future due to clock skew).
    pub frame_age: Option<Duration>,
    /// Time the frame spent waiting in the bounded queue.
    ///
    /// Measured from `push()` to `pop()` using `Instant`. A
    /// consistently low value combined with high `frame_age` proves
    /// that staleness originates upstream of the queue (e.g., in the
    /// decoder or TCP receive buffer), not from queue backlog.
    pub queue_hold_time: std::time::Duration,
    /// Whether this output includes the source frame.
    ///
    /// Always `true` for [`FrameInclusion::Always`], always `false` for
    /// [`FrameInclusion::Never`], and periodic for
    /// [`FrameInclusion::Sampled`].
    pub frame_included: bool,
}

/// Per-stage provenance record.
#[derive(Debug, Clone)]
pub struct StageProvenance {
    /// Which stage.
    pub stage_id: StageId,
    /// When the stage started processing (monotonic nanos from pipeline epoch).
    pub start_ts: MonotonicTs,
    /// When the stage finished processing.
    pub end_ts: MonotonicTs,
    /// Stage processing latency.
    pub latency: Duration,
    /// Whether the stage succeeded, failed, or skipped.
    pub result: StageResult,
}

/// Outcome of a stage's processing for one frame.
#[derive(Debug, Clone, PartialEq)]
pub enum StageResult {
    /// Stage completed successfully.
    Ok,
    /// Stage failed on this frame.
    Error(StageOutcomeCategory),
    /// Stage opted out for this frame.
    Skipped,
}

/// Typed failure category for stage provenance.
///
/// A summary for programmatic filtering and dashboarding.
/// Full diagnostic detail remains in tracing logs.
#[derive(Debug, Clone, PartialEq)]
pub enum StageOutcomeCategory {
    /// Inference or computation failed.
    ProcessingFailed,
    /// Stage ran out of a resource (GPU OOM, buffer limit, etc.).
    ResourceExhausted,
    /// Model or external dependency unavailable.
    DependencyUnavailable,
    /// Stage panicked (caught by executor).
    Panic,
    /// Uncategorized — carries a short, stable tag chosen by the stage author.
    Other {
        /// A static tag for filtering (e.g., `"calibration_stale"`).
        tag: &'static str,
    },
}

/// Per-frame provenance of the view system's decisions.
///
/// Consumers can audit exactly why the view system made the choices
/// it did for any given frame.
#[derive(Debug, Clone)]
pub struct ViewProvenance {
    /// The motion source used for this frame.
    pub motion_source: MotionSource,
    /// The epoch decision made this frame.
    /// `None` if the view system was not consulted (no motion detected,
    /// or `CameraMode::Fixed`).
    pub epoch_decision: Option<EpochDecision>,
    /// Transition phase at this frame.
    pub transition: TransitionPhase,
    /// Stability score at this frame.
    pub stability_score: f32,
    /// View epoch at this frame.
    pub epoch: ViewEpoch,
    /// View version at this frame.
    pub version: ViewVersion,
}
