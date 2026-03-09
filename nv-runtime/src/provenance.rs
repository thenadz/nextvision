//! Provenance types — audit trail for stage and view-system decisions.

use nv_core::id::StageId;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_view::view_state::{ViewEpoch, ViewVersion};
use nv_view::{CameraMotionState, EpochDecision, MotionSource, TransitionPhase};

/// Full provenance for one processed frame.
///
/// Every output carries this, making production debugging tractable.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// Per-stage provenance records, in execution order.
    pub stages: Vec<StageProvenance>,
    /// View-system provenance for this frame.
    pub view_provenance: ViewProvenance,
    /// Timestamp when the frame was received from the source.
    pub frame_receive_ts: MonotonicTs,
    /// Timestamp when the pipeline completed processing this frame.
    pub pipeline_complete_ts: MonotonicTs,
    /// Total pipeline latency (receive → complete).
    pub total_latency: Duration,
}

/// Per-stage provenance record.
#[derive(Debug, Clone)]
pub struct StageProvenance {
    /// Which stage.
    pub stage_id: StageId,
    /// When the stage started processing.
    pub start_ts: MonotonicTs,
    /// When the stage finished processing.
    pub end_ts: MonotonicTs,
    /// Stage processing latency.
    pub latency: Duration,
    /// Whether the stage succeeded, failed, or skipped.
    pub result: StageResult,
}

/// Outcome of a stage's processing for one frame.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
