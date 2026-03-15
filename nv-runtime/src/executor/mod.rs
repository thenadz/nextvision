//! Pipeline executor — runs perception stages on a single frame.
//!
//! The executor is owned by the feed worker thread and holds all per-feed
//! mutable state: stages, temporal store, view state, and stage metrics.
//!
//! It is NOT shared across threads. The feed worker calls
//! [`process_frame()`](PipelineExecutor::process_frame) once per frame
//! in a tight loop.
//!
//! # View orchestration (Issue 5)
//!
//! For `CameraMode::Observed` feeds the executor polls the
//! [`ViewStateProvider`] once per frame, runs the [`EpochPolicy`], and
//! applies the resulting [`EpochDecision`]:
//!
//! - `Continue` — no change.
//! - `Degrade` — degrade context validity.
//! - `Compensate` — degrade + apply compensation transform to existing
//!   trajectory data so positions align with the new view.
//! - `Segment` — increment epoch, segment trajectories, notify stages
//!   via `on_view_epoch_change`.
//!
//! # Temporal commit (Issue 6)
//!
//! After all stages finish, the executor commits the merged track set
//! into the [`TemporalStore`], enforcing retention.
//!
//! ## Authoritative track semantics
//!
//! When at least one stage returns `Some(tracks)` in its output, the
//! merged track set is considered **authoritative** for the frame.
//! Tracks previously in the temporal store but absent from this
//! authoritative set are ended via
//! [`TemporalStore::end_track`](nv_temporal::TemporalStore::end_track),
//! which closes the active trajectory segment with
//! [`SegmentBoundary::TrackEnded`](nv_temporal::SegmentBoundary::TrackEnded).
//!
//! When no stage produces tracks (non-authoritative frame), missing
//! tracks are **not** ended — the executor cannot distinguish "all
//! tracks left" from "no tracker ran this frame."
//!
//! # Stage error semantics (Issue 9)
//!
//! A stage returning `Err(StageError)` causes the frame to be *dropped*:
//! remaining stages are skipped and no output is produced for that frame.
//! A health event is emitted. The feed continues with the next frame.
//! A stage *panic* exits the loop for restart evaluation.
//!
//! # Provenance timing (Issue 8)
//!
//! All provenance timestamps are measured with [`Instant`] on the stage
//! thread and converted to [`MonotonicTs`] offsets from the pipeline
//! epoch so they sit in the same clock domain as frame timestamps.

mod frame_processing;
mod lifecycle;
mod view;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use nv_core::config::CameraMode;
use nv_core::id::FeedId;
use nv_core::metrics::StageMetrics;
use nv_core::timestamp::MonotonicTs;
use nv_core::TrackId;
use nv_perception::Stage;
use nv_temporal::{RetentionPolicy, TemporalStore};
use nv_view::{
    ContextValidity, EpochPolicy, ViewSnapshot, ViewState, ViewStateProvider,
};

use crate::batch::BatchHandle;
use crate::output::FrameInclusion;

/// Minimum interval between `BatchSubmissionRejected` health events
/// per feed. During sustained overload, rejections are accumulated
/// and reported in a single coalesced event per window.
const BATCH_REJECTION_THROTTLE: std::time::Duration = std::time::Duration::from_secs(1);

/// Minimum interval between `BatchTimeout` health events per feed.
/// Under sustained timeout conditions, timeouts are accumulated and
/// reported in a single coalesced event per window.
const BATCH_TIMEOUT_THROTTLE: std::time::Duration = std::time::Duration::from_secs(1);

/// Minimum interval between `BatchInFlightExceeded` health events per
/// feed. Under sustained in-flight cap hits (prior timed-out items
/// still in the coordinator), rejections are accumulated and reported
/// in a single coalesced event per window.
const BATCH_IN_FLIGHT_THROTTLE: std::time::Duration = std::time::Duration::from_secs(1);

/// Per-feed pipeline executor.
///
/// Owns stages, temporal store, view state, and (optionally) the
/// view-state provider and epoch policy.  Called once per frame by
/// the feed worker thread.
pub(crate) struct PipelineExecutor {
    pub(super) feed_id: FeedId,
    pub(super) camera_mode: CameraMode,
    /// Pre-batch stages (or all stages if no batch point).
    pub(super) stages: Vec<Box<dyn Stage>>,
    /// Optional shared batch handle. When present, after pre-batch stages
    /// the executor submits the frame to the batch coordinator and merges
    /// the result before running post-batch stages.
    pub(super) batch: Option<BatchHandle>,
    /// Post-batch stages (empty when no batch point).
    pub(super) post_batch_stages: Vec<Box<dyn Stage>>,
    pub(super) temporal: TemporalStore,
    pub(super) view_state: ViewState,
    pub(super) view_snapshot: ViewSnapshot,
    pub(super) view_state_provider: Option<Box<dyn ViewStateProvider>>,
    pub(super) epoch_policy: Box<dyn EpochPolicy>,
    /// Metrics for pre-batch stages followed by post-batch stages.
    pub(super) stage_metrics: Vec<StageMetrics>,
    pub(super) frames_processed: u64,
    /// Monotonic clock anchor — set at executor creation. Provenance
    /// timestamps are `anchor_ts + elapsed` so they share the same
    /// domain as [`FrameEnvelope::ts()`].
    pub(super) clock_anchor: Instant,
    pub(super) clock_anchor_ts: MonotonicTs,
    /// Duration the camera has been in the current motion state.
    pub(super) motion_state_start: Instant,
    /// Whether to include the source frame in output envelopes.
    pub(super) frame_inclusion: FrameInclusion,
    /// Reusable buffer for track-ending: current frame's track IDs.
    pub(super) track_id_buf: HashSet<TrackId>,
    /// Reusable buffer for track-ending: IDs of tracks to end.
    pub(super) ended_buf: Vec<TrackId>,
    /// Throttle state for BatchSubmissionRejected health events:
    /// accumulated rejection count since last emission.
    pub(super) batch_rejection_count: u64,
    /// Last time a BatchSubmissionRejected event was emitted.
    pub(super) last_batch_rejection_event: Option<Instant>,
    /// Throttle state for BatchTimeout health events:
    /// accumulated timeout count since last emission.
    pub(super) batch_timeout_count: u64,
    /// Last time a BatchTimeout event was emitted.
    pub(super) last_batch_timeout_event: Option<Instant>,
    /// Per-feed in-flight counter shared with the coordinator via
    /// PendingEntry. The coordinator decrements when it processes or
    /// drains each item; the executor checks before submit.
    pub(super) batch_in_flight: Option<Arc<std::sync::atomic::AtomicUsize>>,
    /// Throttle state for BatchInFlightExceeded health events:
    /// accumulated rejection count since last emission.
    pub(super) batch_in_flight_rejection_count: u64,
    /// Last time a BatchInFlightExceeded event was emitted.
    pub(super) last_batch_in_flight_rejection_event: Option<Instant>,
    /// Reference to the feed's shutdown flag. Used to distinguish expected
    /// coordinator shutdown (feed/runtime is shutting down) from unexpected
    /// coordinator death (coordinator crashed while feed is alive).
    pub(super) feed_shutdown: Arc<AtomicBool>,
    /// Whether we've already emitted a health event for unexpected
    /// coordinator loss. Prevents per-frame storms.
    pub(super) coordinator_loss_emitted: bool,
}

impl PipelineExecutor {
    /// Create a new executor with the given stages, policies, and optional
    /// view-state provider.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feed_id: FeedId,
        stages: Vec<Box<dyn Stage>>,
        batch: Option<BatchHandle>,
        post_batch_stages: Vec<Box<dyn Stage>>,
        retention: RetentionPolicy,
        camera_mode: CameraMode,
        view_state_provider: Option<Box<dyn ViewStateProvider>>,
        epoch_policy: Box<dyn EpochPolicy>,
        frame_inclusion: FrameInclusion,
        feed_shutdown: Arc<AtomicBool>,
    ) -> Self {
        let view_state = match camera_mode {
            CameraMode::Fixed => ViewState::fixed_initial(),
            CameraMode::Observed => ViewState::observed_initial(),
        };
        let view_snapshot = ViewSnapshot::new(view_state.clone());
        let total_stage_count = stages.len() + post_batch_stages.len();
        let batch_in_flight = batch.as_ref().map(|_| {
            Arc::new(std::sync::atomic::AtomicUsize::new(0))
        });
        let now = Instant::now();
        Self {
            feed_id,
            camera_mode,
            stages,
            batch,
            post_batch_stages,
            temporal: TemporalStore::new(retention),
            view_state,
            view_snapshot,
            view_state_provider,
            epoch_policy,
            stage_metrics: vec![
                StageMetrics {
                    frames_processed: 0,
                    errors: 0,
                };
                total_stage_count
            ],
            frames_processed: 0,
            clock_anchor: now,
            clock_anchor_ts: MonotonicTs::from_nanos(0),
            motion_state_start: now,
            frame_inclusion,
            track_id_buf: HashSet::new(),
            ended_buf: Vec::new(),
            batch_rejection_count: 0,
            last_batch_rejection_event: None,
            batch_timeout_count: 0,
            last_batch_timeout_event: None,
            batch_in_flight,
            batch_in_flight_rejection_count: 0,
            last_batch_in_flight_rejection_event: None,
            feed_shutdown,
            coordinator_loss_emitted: false,
        }
    }

    /// Current view epoch.
    pub fn view_epoch(&self) -> u64 {
        self.view_snapshot.epoch().as_u64()
    }

    /// Current active track count.
    pub fn track_count(&self) -> usize {
        self.temporal.track_count()
    }

    /// Current view-system stability score in `[0.0, 1.0]`.
    pub fn stability_score(&self) -> f32 {
        self.view_snapshot.stability_score()
    }

    /// Context validity as a `u8` ordinal (0 = Valid, 1 = Degraded, 2 = Invalid).
    ///
    /// Used by the worker to store in an `AtomicU8` without importing nv-view types.
    pub fn context_validity_ordinal(&self) -> u8 {
        match self.view_snapshot.validity() {
            ContextValidity::Valid => 0,
            ContextValidity::Degraded { .. } => 1,
            ContextValidity::Invalid => 2,
        }
    }
}

/// Convert an [`Instant`] to [`MonotonicTs`] using given anchor values.
///
/// Free function to avoid borrow conflicts when called inside
/// `self.stages.iter_mut()` loops.
fn instant_to_ts_impl(anchor: Instant, anchor_ts: MonotonicTs, t: Instant) -> MonotonicTs {
    let elapsed = t.duration_since(anchor);
    MonotonicTs::from_nanos(anchor_ts.as_nanos() + elapsed.as_nanos() as u64)
}

#[cfg(test)]
impl PipelineExecutor {
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }
}
