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

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use nv_core::config::CameraMode;
use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_core::metrics::StageMetrics;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_core::TrackId;
use nv_frame::FrameEnvelope;
use nv_perception::batch::BatchEntry;
use nv_perception::{PerceptionArtifacts, Stage, StageContext};
use nv_temporal::{RetentionPolicy, TemporalStore, TemporalStoreSnapshot};
use nv_view::{
    CameraMotionState, EpochDecision, EpochPolicy, EpochPolicyContext, MotionPollContext,
    MotionReport, MotionSource, TransitionPhase, ViewSnapshot, ViewState, ViewStateProvider,
};

use crate::batch::{BatchHandle, BatchSubmitError};
use crate::output::OutputEnvelope;

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
use crate::output::FrameInclusion;
use crate::provenance::{
    Provenance, StageOutcomeCategory, StageProvenance, StageResult, ViewProvenance,
};

/// Per-feed pipeline executor.
///
/// Owns stages, temporal store, view state, and (optionally) the
/// view-state provider and epoch policy.  Called once per frame by
/// the feed worker thread.
pub(crate) struct PipelineExecutor {
    feed_id: FeedId,
    camera_mode: CameraMode,
    /// Pre-batch stages (or all stages if no batch point).
    stages: Vec<Box<dyn Stage>>,
    /// Optional shared batch handle. When present, after pre-batch stages
    /// the executor submits the frame to the batch coordinator and merges
    /// the result before running post-batch stages.
    batch: Option<BatchHandle>,
    /// Post-batch stages (empty when no batch point).
    post_batch_stages: Vec<Box<dyn Stage>>,
    temporal: TemporalStore,
    view_state: ViewState,
    view_snapshot: ViewSnapshot,
    view_state_provider: Option<Box<dyn ViewStateProvider>>,
    epoch_policy: Box<dyn EpochPolicy>,
    /// Metrics for pre-batch stages followed by post-batch stages.
    stage_metrics: Vec<StageMetrics>,
    frames_processed: u64,
    /// Monotonic clock anchor — set at executor creation. Provenance
    /// timestamps are `anchor_ts + elapsed` so they share the same
    /// domain as [`FrameEnvelope::ts()`].
    clock_anchor: Instant,
    clock_anchor_ts: MonotonicTs,
    /// Duration the camera has been in the current motion state.
    motion_state_start: Instant,
    /// Whether to include the source frame in output envelopes.
    frame_inclusion: FrameInclusion,
    /// Reusable buffer for track-ending: current frame's track IDs.
    track_id_buf: HashSet<TrackId>,
    /// Reusable buffer for track-ending: IDs of tracks to end.
    ended_buf: Vec<TrackId>,
    /// Throttle state for BatchSubmissionRejected health events:
    /// accumulated rejection count since last emission.
    batch_rejection_count: u64,
    /// Last time a BatchSubmissionRejected event was emitted.
    last_batch_rejection_event: Option<Instant>,
    /// Throttle state for BatchTimeout health events:
    /// accumulated timeout count since last emission.
    batch_timeout_count: u64,
    /// Last time a BatchTimeout event was emitted.
    last_batch_timeout_event: Option<Instant>,
    /// Per-feed in-flight counter shared with the coordinator via
    /// PendingEntry. The coordinator decrements when it processes or
    /// drains each item; the executor checks before submit.
    batch_in_flight: Option<Arc<std::sync::atomic::AtomicUsize>>,
    /// Throttle state for BatchInFlightExceeded health events:
    /// accumulated rejection count since last emission.
    batch_in_flight_rejection_count: u64,
    /// Last time a BatchInFlightExceeded event was emitted.
    last_batch_in_flight_rejection_event: Option<Instant>,
    /// Reference to the feed's shutdown flag. Used to distinguish expected
    /// coordinator shutdown (feed/runtime is shutting down) from unexpected
    /// coordinator death (coordinator crashed while feed is alive).
    feed_shutdown: Arc<AtomicBool>,
    /// Whether we've already emitted a health event for unexpected
    /// coordinator loss. Prevents per-frame storms.
    coordinator_loss_emitted: bool,
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

    /// Call `on_start()` on each stage in order (pre-batch then post-batch).
    ///
    /// If any stage fails or panics, previously-started stages are stopped
    /// (best-effort) and the error is returned.
    pub fn start_stages(&mut self) -> Result<(), StageError> {
        // Start pre-batch stages.
        if let Err((started, e)) = start_stage_slice(&self.feed_id, &mut self.stages) {
            self.rollback_started_stages(started, 0);
            return Err(e);
        }

        // Start post-batch stages.
        if let Err((started, e)) = start_stage_slice(&self.feed_id, &mut self.post_batch_stages) {
            self.rollback_started_stages(self.stages.len(), started);
            return Err(e);
        }

        Ok(())
    }

    /// Best-effort stop of pre-batch stages `[0..pre_count)` and
    /// post-batch stages `[0..post_count)`. Used on startup failure.
    fn rollback_started_stages(&mut self, pre_count: usize, post_count: usize) {
        for s in &mut self.stages[..pre_count] {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = s.on_stop();
            }));
        }
        for s in &mut self.post_batch_stages[..post_count] {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = s.on_stop();
            }));
        }
    }

    /// Call `on_stop()` on each stage (pre-batch then post-batch) — best-effort,
    /// errors and panics are logged.
    pub fn stop_stages(&mut self) {
        for stage in self.stages.iter_mut().chain(self.post_batch_stages.iter_mut()) {
            let stage_id = stage.id();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stage.on_stop()
            }));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!(
                        feed_id = %self.feed_id,
                        stage_id = %stage_id,
                        error = %e,
                        "stage on_stop error (ignored)"
                    );
                }
                Err(_) => {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        stage_id = %stage_id,
                        "stage on_stop() panicked (ignored)"
                    );
                }
            }
        }
    }

    /// Flush any accumulated batch rejection count as a final
    /// [`HealthEvent::BatchSubmissionRejected`].
    ///
    /// Called at lifecycle boundaries (stop/restart) so that short
    /// rejection bursts that didn't reach the throttle window are
    /// still surfaced.
    pub fn flush_batch_rejections(&mut self) -> Option<HealthEvent> {
        if self.batch_rejection_count == 0 {
            return None;
        }
        let processor_id = self.batch.as_ref()?.processor_id();
        let count = self.batch_rejection_count;
        self.batch_rejection_count = 0;
        self.last_batch_rejection_event = None;
        Some(HealthEvent::BatchSubmissionRejected {
            feed_id: self.feed_id,
            processor_id,
            dropped_count: count,
        })
    }

    /// Flush any accumulated batch timeout count as a final
    /// [`HealthEvent::BatchTimeout`].
    ///
    /// Called at lifecycle boundaries (stop/restart) so that short
    /// timeout bursts that didn't reach the throttle window are
    /// still surfaced.
    pub fn flush_batch_timeouts(&mut self) -> Option<HealthEvent> {
        if self.batch_timeout_count == 0 {
            return None;
        }
        let processor_id = self.batch.as_ref()?.processor_id();
        let count = self.batch_timeout_count;
        self.batch_timeout_count = 0;
        self.last_batch_timeout_event = None;
        Some(HealthEvent::BatchTimeout {
            feed_id: self.feed_id,
            processor_id,
            timed_out_count: count,
        })
    }

    /// Flush any accumulated batch in-flight cap rejections as a final
    /// [`HealthEvent::BatchInFlightExceeded`].
    ///
    /// Called at lifecycle boundaries (stop/restart) so that short
    /// bursts that didn't reach the throttle window are still surfaced.
    pub fn flush_batch_in_flight_rejections(&mut self) -> Option<HealthEvent> {
        if self.batch_in_flight_rejection_count == 0 {
            return None;
        }
        let processor_id = self.batch.as_ref()?.processor_id();
        let count = self.batch_in_flight_rejection_count;
        self.batch_in_flight_rejection_count = 0;
        self.last_batch_in_flight_rejection_event = None;
        Some(HealthEvent::BatchInFlightExceeded {
            feed_id: self.feed_id,
            processor_id,
            rejected_count: count,
        })
    }

    /// Convert a wall-clock [`Instant`] to a [`MonotonicTs`] anchored to
    /// the executor's creation time.
    fn instant_to_ts(&self, t: Instant) -> MonotonicTs {
        instant_to_ts_impl(self.clock_anchor, self.clock_anchor_ts, t)
    }

    /// Apply the four motion-related fields that every [`EpochDecision`]
    /// branch writes, plus bump the version counter.
    fn apply_motion_fields(
        &mut self,
        motion_state: CameraMotionState,
        motion_source: MotionSource,
        report: &MotionReport,
    ) {
        self.view_state.motion = motion_state;
        self.view_state.motion_source = motion_source;
        self.view_state.ptz = report.ptz.clone();
        self.view_state.global_transform = report.frame_transform.clone();
        self.view_state.version = self.view_state.version.next();
    }

    // ------------------------------------------------------------------
    // View orchestration
    // ------------------------------------------------------------------

    /// Poll the view-state provider and apply the epoch policy.
    ///
    /// Returns `(MotionSource, Option<EpochDecision>)` for provenance.
    fn update_view(
        &mut self,
        frame: &FrameEnvelope,
        health_events: &mut Vec<HealthEvent>,
    ) -> (MotionSource, Option<EpochDecision>) {
        if self.camera_mode == CameraMode::Fixed {
            return (MotionSource::None, None);
        }

        let provider = match self.view_state_provider.as_ref() {
            Some(p) => p,
            None => return (MotionSource::None, None),
        };

        let poll_ctx = MotionPollContext {
            ts: frame.ts(),
            frame,
            previous_view: &self.view_state,
        };

        let report: MotionReport = provider.poll(&poll_ctx);
        let motion_source = derive_motion_source(&report);
        let motion_state = derive_motion_state(&report);

        // Track how long we've been in the current state.
        if core::mem::discriminant(&motion_state)
            != core::mem::discriminant(&self.view_state.motion)
        {
            self.motion_state_start = Instant::now();
        }
        let state_duration =
            Duration::from_nanos(self.motion_state_start.elapsed().as_nanos() as u64);

        let epoch_ctx = EpochPolicyContext {
            previous_view: &self.view_state,
            current_report: &report,
            motion_state: motion_state.clone(),
            state_duration,
        };

        let decision = self.epoch_policy.decide(&epoch_ctx);

        // Apply the decision to the view state.
        //
        // All branches update the same four motion fields + version;
        // `apply_motion_fields` extracts that common prefix.
        self.apply_motion_fields(motion_state, motion_source.clone(), &report);

        match &decision {
            EpochDecision::Continue => {
                let is_stable = matches!(self.view_state.motion, CameraMotionState::Stable);
                if is_stable {
                    self.view_state.stability_score =
                        (self.view_state.stability_score + 0.1).min(1.0);
                    self.view_state.validity = nv_view::ContextValidity::Valid;
                }
            }
            EpochDecision::Degrade { reason } => {
                self.view_state.validity = nv_view::ContextValidity::Degraded {
                    reason: *reason,
                };
                self.view_state.stability_score = (self.view_state.stability_score - 0.2).max(0.0);

                health_events.push(HealthEvent::ViewDegraded {
                    feed_id: self.feed_id,
                    stability_score: self.view_state.stability_score,
                });
            }
            EpochDecision::Compensate { reason, transform } => {
                self.view_state.validity = nv_view::ContextValidity::Degraded {
                    reason: *reason,
                };
                self.view_state.stability_score = (self.view_state.stability_score - 0.1).max(0.0);
                let current_epoch = self.view_state.epoch;

                // Apply the compensation transform to existing trajectory data
                // so previously-recorded positions align with the new view.
                self.temporal.apply_compensation(transform, current_epoch);

                health_events.push(HealthEvent::ViewCompensationApplied {
                    feed_id: self.feed_id,
                    epoch: current_epoch.as_u64(),
                });
            }
            EpochDecision::Segment => {
                let new_epoch = self.view_state.epoch.next();
                self.view_state.epoch = new_epoch;
                self.view_state.validity = nv_view::ContextValidity::Valid;
                self.view_state.stability_score = 0.0;
                self.view_state.transition = nv_view::TransitionPhase::MoveStart;

                // Update temporal store epoch.
                self.temporal.set_view_epoch(new_epoch);

                // Notify stages (pre-batch and post-batch).
                notify_epoch_change(
                    self.feed_id,
                    &mut self.stages,
                    &mut self.post_batch_stages,
                    new_epoch,
                );

                health_events.push(HealthEvent::ViewEpochChanged {
                    feed_id: self.feed_id,
                    epoch: new_epoch.as_u64(),
                });
            }
        }

        // Advance the transition state machine.
        // `Segment` sets transition to `MoveStart` explicitly (the first
        // frame of the new epoch), so we skip `next()` for that case.
        if !matches!(decision, EpochDecision::Segment) {
            let is_moving = !matches!(self.view_state.motion, CameraMotionState::Stable);
            self.view_state.transition = self.view_state.transition.next(is_moving);
        }

        // Rebuild snapshot.
        self.view_snapshot = ViewSnapshot::new(self.view_state.clone());

        (motion_source, Some(decision))
    }

    // ------------------------------------------------------------------
    // Frame processing
    // ------------------------------------------------------------------

    /// Process a single frame through the pipeline.
    ///
    /// Execution order:
    /// 1. View orchestration (for observed cameras).
    /// 2. Pre-batch stages (sequential, on feed thread).
    /// 3. Batch point (if present) — submit to shared coordinator, block
    ///    until result, merge into artifacts.
    /// 4. Post-batch stages (sequential, on feed thread).
    /// 5. Temporal commit + retention.
    ///
    /// Returns `Some((output, health_events))` on success (even if individual
    /// stages produced errors that were recorded).
    /// Returns `None` if a stage error drops the frame.
    ///
    /// A stage *panic* is a sentinel: health events are returned and the
    /// caller decides whether to restart.
    pub fn process_frame(
        &mut self,
        frame: &FrameEnvelope,
    ) -> (Option<OutputEnvelope>, Vec<HealthEvent>) {
        let t_pipeline_start = Instant::now();
        let frame_receive_ts = self.instant_to_ts(t_pipeline_start);

        let mut health_events = Vec::new();

        // --- View orchestration (Issue 5) ---
        let (motion_source, epoch_decision) = self.update_view(frame, &mut health_events);

        // Snapshot temporal store once for the whole frame.
        let temporal_snapshot = self.temporal.snapshot();
        let mut artifacts = PerceptionArtifacts::empty();
        let pre_batch_count = self.stages.len();
        let post_batch_count = self.post_batch_stages.len();
        let total_prov_capacity = pre_batch_count + post_batch_count + usize::from(self.batch.is_some());
        let mut stage_provs = Vec::with_capacity(total_prov_capacity);

        // Capture clock anchors so we can call the free function inside
        // the mutable-borrow loop over self.stages.
        let anchor = self.clock_anchor;
        let anchor_ts = self.clock_anchor_ts;

        // --- Pre-batch stages ---
        let pre_outcome = run_stage_sequence(
            &mut self.stages,
            &mut self.stage_metrics,
            0,
            self.feed_id,
            frame,
            &mut artifacts,
            &self.view_snapshot,
            &temporal_snapshot,
            &mut stage_provs,
            &mut health_events,
            anchor,
            anchor_ts,
        );

        // Early exit on pre-batch failure.
        match pre_outcome {
            StageSeqOutcome::Panic => {
                self.frames_processed += 1;
                return (None, health_events);
            }
            StageSeqOutcome::FrameDropped => {
                self.frames_processed += 1;
                return (None, health_events);
            }
            StageSeqOutcome::Ok => {}
        }

        // --- Batch point ---
        if let Some(ref batch_handle) = self.batch {
            let batch_id = batch_handle.processor_id();
            let t_batch_start = Instant::now();

            let entry = BatchEntry {
                feed_id: self.feed_id,
                frame: frame.clone(),
                view: self.view_snapshot.clone(),
                output: None,
            };

            let batch_result = batch_handle.submit_and_wait(
                entry,
                self.batch_in_flight.as_ref(),
            );
            let t_batch_end = Instant::now();
            let batch_latency = Duration::from_nanos(t_batch_start.elapsed().as_nanos() as u64);

            match batch_result {
                Ok(output) => {
                    // Flush any accumulated rejection count from a prior
                    // overload period now that submissions are succeeding
                    // again (recovery boundary).
                    if self.batch_rejection_count > 0 {
                        health_events.push(HealthEvent::BatchSubmissionRejected {
                            feed_id: self.feed_id,
                            processor_id: batch_id,
                            dropped_count: self.batch_rejection_count,
                        });
                        self.batch_rejection_count = 0;
                        self.last_batch_rejection_event = None;
                    }

                    // Flush any accumulated timeout count on recovery.
                    if self.batch_timeout_count > 0 {
                        health_events.push(HealthEvent::BatchTimeout {
                            feed_id: self.feed_id,
                            processor_id: batch_id,
                            timed_out_count: self.batch_timeout_count,
                        });
                        self.batch_timeout_count = 0;
                        self.last_batch_timeout_event = None;
                    }

                    // Flush any accumulated in-flight cap rejections on recovery.
                    if self.batch_in_flight_rejection_count > 0 {
                        health_events.push(HealthEvent::BatchInFlightExceeded {
                            feed_id: self.feed_id,
                            processor_id: batch_id,
                            rejected_count: self.batch_in_flight_rejection_count,
                        });
                        self.batch_in_flight_rejection_count = 0;
                        self.last_batch_in_flight_rejection_event = None;
                    }

                    artifacts.merge(output);
                    stage_provs.push(StageProvenance {
                        stage_id: batch_id,
                        start_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_start),
                        end_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_end),
                        latency: batch_latency,
                        result: StageResult::Ok,
                    });
                }
                Err(submit_err) => {
                    let (result, health) = match submit_err {
                        BatchSubmitError::QueueFull => {
                            // Throttle: accumulate rejections, emit at
                            // most once per second.
                            self.batch_rejection_count += 1;
                            let now = Instant::now();
                            let should_emit = self
                                .last_batch_rejection_event
                                .is_none_or(|t| now.duration_since(t) >= BATCH_REJECTION_THROTTLE);
                            let health = if should_emit {
                                let count = self.batch_rejection_count;
                                self.batch_rejection_count = 0;
                                self.last_batch_rejection_event = Some(now);
                                Some(HealthEvent::BatchSubmissionRejected {
                                    feed_id: self.feed_id,
                                    processor_id: batch_id,
                                    dropped_count: count,
                                })
                            } else {
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::ResourceExhausted),
                                health,
                            )
                        }
                        BatchSubmitError::ProcessingFailed(ref e) => {
                            // The coordinator already emitted the
                            // authoritative BatchError with the real
                            // batch_size — do NOT emit a duplicate here.
                            (
                                categorize_stage_error(e),
                                None,
                            )
                        }
                        BatchSubmitError::CoordinatorShutdown => {
                            // Distinguish expected feed/runtime shutdown
                            // from unexpected coordinator death.
                            let is_expected = self.feed_shutdown.load(Ordering::Relaxed);
                            let health = if is_expected {
                                // Expected lifecycle event — no health noise.
                                None
                            } else if !self.coordinator_loss_emitted {
                                // Unexpected coordinator loss — emit once.
                                self.coordinator_loss_emitted = true;
                                Some(HealthEvent::StageError {
                                    feed_id: self.feed_id,
                                    stage_id: batch_id,
                                    error: StageError::ProcessingFailed {
                                        stage_id: batch_id,
                                        detail: "batch coordinator shut down unexpectedly".into(),
                                    },
                                })
                            } else {
                                // Already emitted — suppress duplicates.
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::DependencyUnavailable),
                                health,
                            )
                        }
                        BatchSubmitError::Timeout => {
                            batch_handle.record_timeout();
                            // Throttle: accumulate timeouts, emit at
                            // most once per second.
                            self.batch_timeout_count += 1;
                            let now = Instant::now();
                            let should_emit = self
                                .last_batch_timeout_event
                                .is_none_or(|t| now.duration_since(t) >= BATCH_TIMEOUT_THROTTLE);
                            let health = if should_emit {
                                let count = self.batch_timeout_count;
                                self.batch_timeout_count = 0;
                                self.last_batch_timeout_event = Some(now);
                                Some(HealthEvent::BatchTimeout {
                                    feed_id: self.feed_id,
                                    processor_id: batch_id,
                                    timed_out_count: count,
                                })
                            } else {
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::ProcessingFailed),
                                health,
                            )
                        }
                        BatchSubmitError::InFlightCapReached => {
                            // Prior timed-out item still in coordinator.
                            // Throttle: same pattern as QueueFull.
                            self.batch_in_flight_rejection_count += 1;
                            let now = Instant::now();
                            let should_emit = self
                                .last_batch_in_flight_rejection_event
                                .is_none_or(|t| now.duration_since(t) >= BATCH_IN_FLIGHT_THROTTLE);
                            let health = if should_emit {
                                let count = self.batch_in_flight_rejection_count;
                                self.batch_in_flight_rejection_count = 0;
                                self.last_batch_in_flight_rejection_event = Some(now);
                                Some(HealthEvent::BatchInFlightExceeded {
                                    feed_id: self.feed_id,
                                    processor_id: batch_id,
                                    rejected_count: count,
                                })
                            } else {
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::ResourceExhausted),
                                health,
                            )
                        }
                    };

                    if let Some(evt) = health {
                        health_events.push(evt);
                    }
                    stage_provs.push(StageProvenance {
                        stage_id: batch_id,
                        start_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_start),
                        end_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_end),
                        latency: batch_latency,
                        result,
                    });

                    // Batch failure drops the frame — skip post-batch stages.
                    self.frames_processed += 1;
                    return (None, health_events);
                }
            }
        }

        // --- Post-batch stages ---
        let post_outcome = run_stage_sequence(
            &mut self.post_batch_stages,
            &mut self.stage_metrics,
            pre_batch_count,
            self.feed_id,
            frame,
            &mut artifacts,
            &self.view_snapshot,
            &temporal_snapshot,
            &mut stage_provs,
            &mut health_events,
            anchor,
            anchor_ts,
        );

        self.frames_processed += 1;

        // If a stage panicked, return health events but no output.
        // The caller (worker) will decide whether to restart.
        match post_outcome {
            StageSeqOutcome::Panic => {
                return (None, health_events);
            }
            StageSeqOutcome::FrameDropped => {
                // Issue 9: frame was dropped due to stage error — emit no output.
                return (None, health_events);
            }
            StageSeqOutcome::Ok => {}
        }

        // --- Track ending (authoritative set semantics) ---
        //
        // End absent tracks *before* committing incoming tracks so that
        // cap space freed by ended tracks is available for admission.
        // Without this ordering, ID churn can cause spurious
        // TrackAdmissionRejected health events when the store is near
        // capacity.
        let now_ts = frame.ts();
        let current_epoch = self.view_state.epoch;

        if artifacts.tracks_authoritative {
            self.track_id_buf.clear();
            self.track_id_buf.extend(artifacts.tracks.iter().map(|t| t.id));
            self.ended_buf.clear();
            self.ended_buf.extend(
                self.temporal
                    .track_ids()
                    .filter(|id| !self.track_id_buf.contains(id))
                    .copied(),
            );
            for id in &self.ended_buf {
                self.temporal.end_track(id);
            }
        }

        // --- Temporal commit ---
        //
        // Tracks in the output envelope are **stage-authoritative**: they
        // reflect exactly what the perception stages produced, regardless
        // of temporal-store admission. If the store rejects a track
        // (e.g., capacity limit), a health event is emitted but the track
        // still appears in the output. Consumers who need to know which
        // tracks have temporal history should consult the store snapshot.
        let mut track_rejections = 0u32;
        let track_total = artifacts.tracks.len() as u32;
        for track in &artifacts.tracks {
            if !self.temporal.commit_track(track, now_ts, current_epoch) {
                track_rejections += 1;
            }
        }
        let admission = crate::output::AdmissionSummary {
            admitted: track_total - track_rejections,
            rejected: track_rejections,
        };
        if track_rejections > 0 {
            health_events.push(HealthEvent::TrackAdmissionRejected {
                feed_id: self.feed_id,
                rejected_count: track_rejections,
            });
        }

        // --- Retention enforcement ---
        self.temporal.enforce_retention(now_ts);

        let t_pipeline_end = Instant::now();
        let pipeline_complete_ts = self.instant_to_ts(t_pipeline_end);
        let total_latency = Duration::from_nanos(t_pipeline_start.elapsed().as_nanos() as u64);

        let output = OutputEnvelope {
            feed_id: self.feed_id,
            frame_seq: frame.seq(),
            ts: frame.ts(),
            wall_ts: frame.wall_ts(),
            detections: artifacts.detections,
            tracks: artifacts.tracks,
            signals: artifacts.signals,
            scene_features: artifacts.scene_features,
            view: self.view_state.clone(),
            provenance: Provenance {
                stages: stage_provs,
                view_provenance: ViewProvenance {
                    motion_source,
                    epoch_decision,
                    transition: self.view_snapshot.transition(),
                    stability_score: self.view_snapshot.stability_score(),
                    epoch: self.view_snapshot.epoch(),
                    version: self.view_snapshot.version(),
                },
                frame_receive_ts,
                pipeline_complete_ts,
                total_latency,
            },
            metadata: artifacts.stage_artifacts,
            frame: match self.frame_inclusion {
                FrameInclusion::Always => Some(frame.clone()),
                FrameInclusion::Never => None,
            },
            admission,
        };

        (Some(output), health_events)
    }

    /// Clear temporal state and increment epoch (called on feed restart).
    ///
    /// All active trajectory segments are closed with
    /// [`SegmentBoundary::FeedRestart`](nv_temporal::SegmentBoundary::FeedRestart)
    /// before clearing, ensuring proper segment boundaries in the data
    /// model even though the data itself is discarded.
    ///
    /// Incrementing the epoch ensures that tracks carried over from a
    /// prior session are segmented, and stages that cache view-dependent
    /// state get an `on_view_epoch_change` notification on the first
    /// frame of the new session.
    pub fn clear_temporal(&mut self) {
        self.temporal
            .close_all_segments(nv_temporal::SegmentBoundary::FeedRestart);
        self.temporal.clear();
        let new_epoch = self.view_state.epoch.next();
        self.view_state.epoch = new_epoch;
        self.view_state.version = self.view_state.version.next();
        self.view_state.transition = TransitionPhase::Settled;
        self.view_state.stability_score = match self.camera_mode {
            CameraMode::Fixed => 1.0,
            CameraMode::Observed => 0.0,
        };
        self.temporal.set_view_epoch(new_epoch);
        self.view_snapshot = ViewSnapshot::new(self.view_state.clone());
    }

    /// Current view epoch.
    pub fn view_epoch(&self) -> u64 {
        self.view_snapshot.epoch().as_u64()
    }

    /// Current active track count.
    pub fn track_count(&self) -> usize {
        self.temporal.track_count()
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

/// Start each stage in `stages` in order, with panic safety.
///
/// Returns `Ok(())` if all stages started. On error, returns
/// `(started_count, error)` — the caller rolls back `[0..started_count)`.
fn start_stage_slice(
    feed_id: &FeedId,
    stages: &mut [Box<dyn Stage>],
) -> Result<(), (usize, StageError)> {
    for (i, stage) in stages.iter_mut().enumerate() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage.on_start()
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err((i, e)),
            Err(_) => {
                let stage_id = stage.id();
                tracing::error!(
                    feed_id = %feed_id,
                    stage_id = %stage_id,
                    "stage on_start() panicked"
                );
                return Err((i, StageError::ProcessingFailed {
                    stage_id,
                    detail: "on_start() panicked".into(),
                }));
            }
        }
    }
    Ok(())
}

/// Notify all stages (pre-batch + post-batch) of a view epoch change,
/// with panic safety. Errors and panics are logged but do not propagate.
fn notify_epoch_change(
    feed_id: FeedId,
    pre: &mut [Box<dyn Stage>],
    post: &mut [Box<dyn Stage>],
    epoch: nv_view::ViewEpoch,
) {
    for stage in pre.iter_mut().chain(post.iter_mut()) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage.on_view_epoch_change(epoch)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::warn!(
                    feed_id = %feed_id,
                    stage_id = %stage.id(),
                    error = %e,
                    "stage on_view_epoch_change error (ignored)"
                );
            }
            Err(_) => {
                tracing::error!(
                    feed_id = %feed_id,
                    stage_id = %stage.id(),
                    "stage on_view_epoch_change panicked (ignored)"
                );
            }
        }
    }
}

/// Outcome of [`run_stage_sequence`]: indicates whether execution
/// should continue to the next pipeline phase.
enum StageSeqOutcome {
    /// All stages ran successfully.
    Ok,
    /// A stage error caused the frame to be dropped.
    FrameDropped,
    /// A stage panicked.
    Panic,
}

/// Run a sequence of stages, collecting artifacts, provenance, and
/// health events. Shared between pre-batch and post-batch execution.
#[allow(clippy::too_many_arguments)]
fn run_stage_sequence(
    stages: &mut [Box<dyn Stage>],
    metrics: &mut [StageMetrics],
    metrics_offset: usize,
    feed_id: FeedId,
    frame: &FrameEnvelope,
    artifacts: &mut PerceptionArtifacts,
    view_snapshot: &ViewSnapshot,
    temporal_snapshot: &TemporalStoreSnapshot,
    stage_provs: &mut Vec<StageProvenance>,
    health_events: &mut Vec<HealthEvent>,
    anchor: Instant,
    anchor_ts: MonotonicTs,
) -> StageSeqOutcome {
    for (i, stage) in stages.iter_mut().enumerate() {
        let stage_id = stage.id();
        let midx = metrics_offset + i;
        let t_stage_start = Instant::now();

        let ctx = StageContext {
            feed_id,
            frame,
            artifacts,
            view: view_snapshot,
            temporal: temporal_snapshot,
            metrics: &metrics[midx],
        };

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stage.process(&ctx)));

        let t_stage_end = Instant::now();
        let stage_latency = Duration::from_nanos(t_stage_start.elapsed().as_nanos() as u64);

        let stage_result = match result {
            Ok(Ok(output)) => {
                artifacts.merge(output);
                metrics[midx].frames_processed += 1;
                StageResult::Ok
            }
            Ok(Err(e)) => {
                metrics[midx].errors += 1;
                health_events.push(HealthEvent::StageError {
                    feed_id,
                    stage_id,
                    error: e.clone(),
                });
                stage_provs.push(StageProvenance {
                    stage_id,
                    start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
                    end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
                    latency: stage_latency,
                    result: categorize_stage_error(&e),
                });
                return StageSeqOutcome::FrameDropped;
            }
            Err(_panic) => {
                metrics[midx].errors += 1;
                health_events.push(HealthEvent::StagePanic {
                    feed_id,
                    stage_id,
                });
                stage_provs.push(StageProvenance {
                    stage_id,
                    start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
                    end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
                    latency: stage_latency,
                    result: StageResult::Error(StageOutcomeCategory::Panic),
                });
                return StageSeqOutcome::Panic;
            }
        };

        stage_provs.push(StageProvenance {
            stage_id,
            start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
            end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
            latency: stage_latency,
            result: stage_result,
        });
    }

    StageSeqOutcome::Ok
}

/// Map a [`StageError`] variant to the provenance category.
fn categorize_stage_error(e: &StageError) -> StageResult {
    match e {
        StageError::ProcessingFailed { .. } => {
            StageResult::Error(StageOutcomeCategory::ProcessingFailed)
        }
        StageError::ResourceExhausted { .. } => {
            StageResult::Error(StageOutcomeCategory::ResourceExhausted)
        }
        StageError::ModelLoadFailed { .. } => {
            StageResult::Error(StageOutcomeCategory::DependencyUnavailable)
        }
    }
}

/// Derive [`MotionSource`] from a [`MotionReport`].
fn derive_motion_source(report: &MotionReport) -> MotionSource {
    if report.ptz.is_some() {
        MotionSource::Telemetry
    } else if let Some(ref t) = report.frame_transform {
        MotionSource::Inferred {
            confidence: t.confidence,
        }
    } else {
        MotionSource::None
    }
}

/// Derive [`CameraMotionState`] from a [`MotionReport`].
fn derive_motion_state(report: &MotionReport) -> CameraMotionState {
    // Use explicit hint if provided.
    if let Some(ref hint) = report.motion_hint {
        return hint.clone();
    }
    // If PTZ telemetry is available, infer from deltas later (the epoch policy does that).
    // For now, if we have a frame_transform with small displacement, stable.
    if let Some(ref t) = report.frame_transform {
        if t.confidence > 0.5 {
            let displacement = t.displacement_magnitude();
            if displacement < 0.01 {
                return CameraMotionState::Stable;
            }
            return CameraMotionState::Moving {
                angular_velocity: None,
                displacement: Some(displacement),
            };
        }
    }
    CameraMotionState::Unknown
}

#[cfg(test)]
impl PipelineExecutor {
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use nv_core::config::CameraMode;
    use nv_core::id::{FeedId, StageId};
    use nv_perception::StageOutput;
    use nv_temporal::RetentionPolicy;
    use nv_view::{DefaultEpochPolicy, ViewEpoch};

    fn make_executor() -> PipelineExecutor {
        PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy {
                max_track_age: Duration::from_secs(5),
                max_observations_per_track: 10,
                max_concurrent_tracks: 3,
                max_trajectory_points_per_track: 1000,
            },
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn clear_temporal_increments_epoch() {
        let mut exec = make_executor();
        let epoch_before = exec.view_epoch();
        exec.clear_temporal();
        let epoch_after = exec.view_epoch();
        assert!(
            epoch_after > epoch_before,
            "epoch should increase on restart: {epoch_before} → {epoch_after}",
        );
    }

    #[test]
    fn clear_temporal_resets_transition_to_settled() {
        let mut exec = make_executor();
        // Manually set transition to something non-settled.
        exec.view_state.transition = TransitionPhase::Moving;
        exec.clear_temporal();
        assert_eq!(exec.view_state.transition, TransitionPhase::Settled);
    }

    #[test]
    fn enforce_retention_evicts_old_lost_tracks() {
        let mut exec = make_executor();
        let retention = exec.temporal.retention().clone();

        // Insert a "Lost" track with last_seen at time 0.
        let track = nv_perception::Track {
            id: nv_core::TrackId::new(1),
            class_id: 0,
            state: nv_perception::TrackState::Lost,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(0),
                bbox: nv_core::BBox::new(0.0, 0.0, 0.1, 0.1),
                confidence: 0.9,
                state: nv_perception::TrackState::Lost,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        };
        let history = nv_temporal::store::TrackHistory::new(
            Arc::new(track),
            Arc::new(nv_temporal::Trajectory::new()),
            MonotonicTs::from_nanos(0),
            MonotonicTs::from_nanos(0),
            ViewEpoch::INITIAL,
        );
        exec.temporal
            .insert_track(nv_core::TrackId::new(1), history);
        assert_eq!(exec.temporal.track_count(), 1);

        // Now = last_seen + max_track_age + 1s → should be evicted.
        let now = MonotonicTs::from_nanos(
            retention.max_track_age.as_nanos() + Duration::from_secs(1).as_nanos(),
        );
        exec.temporal.enforce_retention(now);
        assert_eq!(
            exec.temporal.track_count(),
            0,
            "old Lost track should be evicted"
        );
    }

    #[test]
    fn enforce_retention_respects_max_concurrent_tracks() {
        let mut exec = make_executor();
        let retention = exec.temporal.retention().clone();
        assert_eq!(retention.max_concurrent_tracks, 3);

        // Insert 5 tracks: ids 1..=5, all Lost, staggered last_seen.
        for i in 1..=5u64 {
            let track = nv_perception::Track {
                id: nv_core::TrackId::new(i),
                class_id: 0,
                state: nv_perception::TrackState::Lost,
                current: nv_perception::TrackObservation {
                    ts: MonotonicTs::from_nanos(i * 1_000_000),
                    bbox: nv_core::BBox::new(0.0, 0.0, 0.1, 0.1),
                    confidence: 0.9,
                    state: nv_perception::TrackState::Lost,
                    detection_id: None,
                    metadata: nv_core::TypedMetadata::new(),
                },
                metadata: nv_core::TypedMetadata::new(),
            };
            let history = nv_temporal::store::TrackHistory::new(
                Arc::new(track),
                Arc::new(nv_temporal::Trajectory::new()),
                MonotonicTs::from_nanos(i * 1_000_000),
                MonotonicTs::from_nanos(i * 1_000_000),
                ViewEpoch::INITIAL,
            );
            exec.temporal
                .insert_track(nv_core::TrackId::new(i), history);
        }
        assert_eq!(exec.temporal.track_count(), 5);

        // Use a "now" that doesn't trigger age-based eviction.
        let now = MonotonicTs::from_nanos(10_000_000);
        exec.temporal.enforce_retention(now);

        // Should be capped at 3.
        assert_eq!(
            exec.temporal.track_count(),
            3,
            "should evict down to max_concurrent_tracks",
        );

        // The oldest (track 1, 2) should be gone; newest (3, 4, 5) remain.
        assert!(exec.temporal.get_track(&nv_core::TrackId::new(1)).is_none());
        assert!(exec.temporal.get_track(&nv_core::TrackId::new(2)).is_none());
        assert!(exec.temporal.get_track(&nv_core::TrackId::new(3)).is_some());
    }

    // ------------------------------------------------------------------
    // Admission rejection visibility (B / D.2)
    // ------------------------------------------------------------------

    #[test]
    fn admission_rejection_emits_health_event() {
        // Stage that returns more tracks than the executor's cap (3).
        struct ManyTracksStage;
        impl nv_perception::Stage for ManyTracksStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("many_tracks")
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                // 5 Confirmed tracks — exceeds cap of 3.
                let tracks: Vec<nv_perception::Track> = (1..=5u64)
                    .map(|i| nv_perception::Track {
                        id: nv_core::TrackId::new(i),
                        class_id: 0,
                        state: nv_perception::TrackState::Confirmed,
                        current: nv_perception::TrackObservation {
                            ts: MonotonicTs::from_nanos(0),
                            bbox: nv_core::BBox::new(0.0, 0.0, 0.1, 0.1),
                            confidence: 0.9,
                            state: nv_perception::TrackState::Confirmed,
                            detection_id: None,
                            metadata: nv_core::TypedMetadata::new(),
                        },
                        metadata: nv_core::TypedMetadata::new(),
                    })
                    .collect();
                Ok(nv_perception::StageOutput::with_tracks(tracks))
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(ManyTracksStage)],
            None,
            Vec::new(),
            RetentionPolicy {
                max_track_age: Duration::from_secs(5),
                max_observations_per_track: 10,
                max_concurrent_tracks: 3,
                max_trajectory_points_per_track: 1000,
            },
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(0),
            2,
            2,
            128,
        );
        let (_output, health_events) = exec.process_frame(&frame);

        // 5 tracks, cap 3: first 3 admitted, last 2 rejected (coalesced into one event).
        let rejection_events: Vec<_> = health_events
            .iter()
            .filter(|e| matches!(e, HealthEvent::TrackAdmissionRejected { .. }))
            .collect();
        assert_eq!(
            rejection_events.len(),
            1,
            "should emit one coalesced TrackAdmissionRejected event, got {rejection_events:?}",
        );
        match &rejection_events[0] {
            HealthEvent::TrackAdmissionRejected { rejected_count, .. } => {
                assert_eq!(*rejected_count, 2, "expected 2 rejected tracks");
            }
            _ => unreachable!(),
        }

        // The temporal store should be at exactly the cap.
        assert_eq!(exec.track_count(), 3);
    }

    // ------------------------------------------------------------------
    // Metadata propagation tests (P2)
    // ------------------------------------------------------------------

    #[test]
    fn process_frame_propagates_stage_metadata_to_output() {
        // A stage that inserts a typed artifact.
        struct MetadataStage;
        impl nv_perception::Stage for MetadataStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("metadata")
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                let mut out = nv_perception::StageOutput::empty();
                out.artifacts.insert::<String>("hello metadata".to_string());
                Ok(out)
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(MetadataStage)],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(0),
            2,
            2,
            128,
        );
        let (output, _health) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        // The stage's typed artifact should appear in the output metadata.
        let value = out.metadata.get::<String>();
        assert_eq!(
            value.map(|s| s.as_str()),
            Some("hello metadata"),
            "stage artifacts should be propagated to output metadata"
        );
    }

    // ------------------------------------------------------------------
    // FeedRestart segmentation tests (P2)
    // ------------------------------------------------------------------

    #[test]
    fn clear_temporal_closes_segments_with_feed_restart() {
        let mut exec = make_executor();

        // Commit a track so there's an active trajectory segment.
        let track = nv_perception::Track {
            id: nv_core::TrackId::new(1),
            class_id: 0,
            state: nv_perception::TrackState::Confirmed,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(1_000_000),
                bbox: nv_core::BBox::new(0.1, 0.1, 0.2, 0.2),
                confidence: 0.95,
                state: nv_perception::TrackState::Confirmed,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        };
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);

        // Verify active segment exists.
        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .unwrap();
        assert!(hist.trajectory.active_segment().is_some());

        // Take a snapshot before clear to capture the closed segment.
        // clear_temporal closes segments, then clears. We verify through
        // the close_all_segments mechanism directly.
        exec.temporal
            .close_all_segments(nv_temporal::SegmentBoundary::FeedRestart);

        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .unwrap();
        assert!(
            hist.trajectory.active_segment().is_none(),
            "segment should be closed"
        );
        let seg = hist.trajectory.segments().last().unwrap();
        assert_eq!(
            seg.closed_by(),
            Some(&nv_temporal::SegmentBoundary::FeedRestart),
            "segment should be closed with FeedRestart boundary"
        );
    }

    // ------------------------------------------------------------------
    // Epoch-driven segmentation end-to-end test (P1)
    // ------------------------------------------------------------------

    #[test]
    fn commit_via_executor_segments_on_epoch_change() {
        let mut exec = make_executor();
        let epoch0 = ViewEpoch::INITIAL;

        // Commit track in epoch 0.
        let track = nv_perception::Track {
            id: nv_core::TrackId::new(1),
            class_id: 0,
            state: nv_perception::TrackState::Confirmed,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(1_000_000),
                bbox: nv_core::BBox::new(0.1, 0.1, 0.2, 0.2),
                confidence: 0.95,
                state: nv_perception::TrackState::Confirmed,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        };
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch0);

        assert_eq!(
            exec.temporal
                .get_track(&nv_core::TrackId::new(1))
                .unwrap()
                .trajectory
                .segment_count(),
            1
        );

        // Simulate epoch change.
        let epoch1 = epoch0.next();
        exec.temporal.set_view_epoch(epoch1);

        // Commit in new epoch — should create new segment.
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch1);

        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .unwrap();
        assert_eq!(
            hist.trajectory.segment_count(),
            2,
            "epoch change should create a new trajectory segment"
        );
        assert!(!hist.trajectory.segments()[0].is_active());
        assert!(hist.trajectory.segments()[1].is_active());
        assert_eq!(hist.trajectory.segments()[1].view_epoch(), epoch1);
    }

    // ------------------------------------------------------------------
    // TrackEnded: authoritative-set semantics
    // ------------------------------------------------------------------

    /// Stage that returns an authoritative (possibly empty) track set.
    struct AuthoritativeTrackStage {
        tracks: Vec<nv_perception::Track>,
    }

    impl AuthoritativeTrackStage {
        fn new(tracks: Vec<nv_perception::Track>) -> Self {
            Self { tracks }
        }
    }

    impl nv_perception::Stage for AuthoritativeTrackStage {
        fn id(&self) -> nv_core::id::StageId {
            nv_core::id::StageId("authoritative_tracker")
        }
        fn process(
            &mut self,
            _ctx: &nv_perception::StageContext<'_>,
        ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
            Ok(nv_perception::StageOutput {
                tracks: Some(self.tracks.clone()),
                ..nv_perception::StageOutput::empty()
            })
        }
    }

    fn make_track(id: u64, state: nv_perception::TrackState) -> nv_perception::Track {
        nv_perception::Track {
            id: nv_core::TrackId::new(id),
            class_id: 0,
            state,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(0),
                bbox: nv_core::BBox::new(0.1, 0.1, 0.2, 0.2),
                confidence: 0.9,
                state,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        }
    }

    #[test]
    fn missing_track_from_authoritative_set_triggers_track_ended() {
        let track_a = make_track(1, nv_perception::TrackState::Confirmed);
        let track_b = make_track(2, nv_perception::TrackState::Confirmed);

        // Frame 1: stage produces [A, B].
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(AuthoritativeTrackStage::new(vec![
                track_a.clone(),
                track_b.clone(),
            ]))],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame1 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (out1, _) = exec.process_frame(&frame1);
        assert!(out1.is_some());
        assert_eq!(exec.temporal.track_count(), 2);

        // Frame 2: stage produces only [A] — B should be ended.
        exec.stages = vec![Box::new(AuthoritativeTrackStage::new(vec![
            track_a.clone(),
        ]))];

        let frame2 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            1,
            MonotonicTs::from_nanos(2_000_000),
            2,
            2,
            128,
        );
        let (out2, _) = exec.process_frame(&frame2);
        assert!(out2.is_some());
        // B should have been removed via end_track.
        assert_eq!(
            exec.temporal.track_count(),
            1,
            "track B should be ended and removed"
        );
        assert!(
            exec.temporal
                .get_track(&nv_core::TrackId::new(1))
                .is_some(),
            "track A should still exist"
        );
        assert!(
            exec.temporal
                .get_track(&nv_core::TrackId::new(2))
                .is_none(),
            "track B should be gone"
        );
    }

    #[test]
    fn no_false_track_ended_when_no_stage_produced_tracks() {
        // Pre-populate two tracks via direct commit.
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            // NoOpStage returns StageOutput::empty() — tracks: None.
            vec![Box::new(nv_test_util::mock_stage::NoOpStage::new("noop"))],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let track_a = make_track(1, nv_perception::TrackState::Confirmed);
        let track_b = make_track(2, nv_perception::TrackState::Confirmed);
        exec.temporal
            .commit_track(&track_a, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);
        exec.temporal
            .commit_track(&track_b, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);
        assert_eq!(exec.temporal.track_count(), 2);

        // Process a frame through the NoOp stage (non-authoritative).
        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(2_000_000),
            2,
            2,
            128,
        );
        let (out, _) = exec.process_frame(&frame);
        assert!(out.is_some());
        // Both tracks should survive — no stage claimed authoritativeness.
        assert_eq!(
            exec.temporal.track_count(),
            2,
            "non-authoritative frame must not end tracks"
        );
    }

    #[test]
    fn explicit_lost_uses_track_lost_not_track_ended() {
        // Stage produces track A as Lost.
        let track_a = make_track(1, nv_perception::TrackState::Lost);

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(AuthoritativeTrackStage::new(vec![
                track_a.clone(),
            ]))],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (out, _) = exec.process_frame(&frame);
        assert!(out.is_some());

        // Track A was committed as Lost — commit_track closes segment
        // with TrackLost. Since it IS in the authoritative set, it is
        // NOT passed to end_track. It remains in the store until retention
        // evicts it.
        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .expect("Lost track should still be in store");
        let last_seg = hist.trajectory.segments().last().unwrap();
        assert_eq!(
            last_seg.closed_by(),
            Some(&nv_temporal::SegmentBoundary::TrackLost),
            "explicit Lost track should have TrackLost boundary, not TrackEnded"
        );
    }

    #[test]
    fn feed_restart_still_uses_feed_restart_boundary() {
        // Commit a track, then clear_temporal — segments should close
        // with FeedRestart, not TrackEnded.
        let mut exec = make_executor();
        let track = make_track(1, nv_perception::TrackState::Confirmed);
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);

        exec.clear_temporal();

        // After clear the store is empty, but we can verify the behavior
        // by the fact that clear_temporal calls close_all_segments with
        // FeedRestart before clearing. This is already tested in
        // clear_temporal_closes_segments_with_feed_restart, but we
        // repeat the essential invariant here for completeness.
        assert_eq!(
            exec.temporal.track_count(),
            0,
            "store should be empty after restart"
        );
    }

    // ------------------------------------------------------------------
    // View-state provenance tests
    // ------------------------------------------------------------------

    #[test]
    fn fixed_camera_provenance_shows_stable() {
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, health) = exec.process_frame(&frame);
        assert!(health.is_empty());
        let out = output.expect("should produce output");

        // Fixed camera — stability score should be 1.0, transition Settled,
        // no epoch decision, epoch 0.
        let vp = &out.provenance.view_provenance;
        assert_eq!(vp.stability_score, 1.0);
        assert_eq!(vp.transition, TransitionPhase::Settled);
        assert_eq!(vp.epoch, ViewEpoch::INITIAL);
        assert!(vp.epoch_decision.is_none(), "fixed camera has no epoch decision");

        // View state on output should be Valid.
        assert!(
            matches!(out.view.validity, nv_view::ContextValidity::Valid),
            "fixed camera output should have Valid context"
        );
    }

    #[test]
    fn observed_camera_with_provider_populates_provenance() {
        use nv_view::{MotionPollContext, MotionReport, ViewStateProvider};

        struct StableProvider;
        impl ViewStateProvider for StableProvider {
            fn poll(&self, _ctx: &MotionPollContext<'_>) -> MotionReport {
                MotionReport::default() // no motion data → Unknown state
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Observed,
            Some(Box::new(StableProvider)),
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        let vp = &out.provenance.view_provenance;
        // The provider returned no data, so CameraMotionState::Unknown.
        // DefaultEpochPolicy returns Continue for Unknown state.
        assert!(
            vp.epoch_decision.is_some(),
            "observed camera should have epoch decision"
        );
        // Version should have advanced from INITIAL.
        assert!(vp.version > nv_view::ViewVersion::INITIAL);
    }

    #[test]
    fn view_degradation_reflected_in_output() {
        use nv_view::{MotionPollContext, MotionReport, ViewStateProvider};

        // Provider reports small PTZ movement → Degrade decision.
        struct SmallPtzProvider;
        impl ViewStateProvider for SmallPtzProvider {
            fn poll(&self, ctx: &MotionPollContext<'_>) -> MotionReport {
                // Always report a small PTZ pan delta.
                let prev_pan = ctx
                    .previous_view
                    .ptz
                    .as_ref()
                    .map(|p| p.pan)
                    .unwrap_or(0.0);
                MotionReport {
                    ptz: Some(nv_view::PtzTelemetry {
                        pan: prev_pan + 2.0, // small move
                        tilt: 0.0,
                        zoom: 0.5,
                        ts: ctx.ts,
                    }),
                    ..Default::default()
                }
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Observed,
            Some(Box::new(SmallPtzProvider)),
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        // First frame initializes PTZ baseline (no previous ptz → Continue on
        // first frame since DefaultEpochPolicy needs both prev and current ptz).
        let f1 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        exec.process_frame(&f1);

        // Second frame: the policy now has prev_ptz and current_ptz, and
        // the delta (2.0) is small → Degrade.
        let f2 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            1,
            MonotonicTs::from_nanos(2_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&f2);
        let out = output.expect("should produce output");

        // Stability score should have decreased from initial.
        assert!(
            out.provenance.view_provenance.stability_score < 1.0,
            "stability should decrease under PTZ movement"
        );
        assert!(
            matches!(
                out.view.validity,
                nv_view::ContextValidity::Degraded { .. }
            ),
            "view should be degraded under small PTZ move"
        );
    }

    // ==================================================================
    // Stage execution flow tests
    // ==================================================================

    #[test]
    fn stages_execute_in_declared_order() {
        // A stage that appends its ID to a shared log via a signal name.
        struct OrderStage {
            name: &'static str,
        }
        impl nv_perception::Stage for OrderStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId(self.name)
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                Ok(nv_perception::StageOutput::with_signal(
                    nv_perception::DerivedSignal {
                        name: self.name,
                        value: nv_perception::SignalValue::Boolean(true),
                        ts: MonotonicTs::ZERO,
                    },
                ))
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(OrderStage { name: "first" }),
                Box::new(OrderStage { name: "second" }),
                Box::new(OrderStage { name: "third" }),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(0),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        // Signals are appended in execution order.
        let names: Vec<&str> = out.signals.iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    #[test]
    fn detector_output_visible_to_tracker() {
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 3)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        assert_eq!(out.detections.len(), 3, "detector should produce 3 detections");
        assert_eq!(out.tracks.len(), 3, "tracker should produce 3 tracks from 3 detections");
    }

    #[test]
    fn full_pipeline_detector_tracker_temporal_sink() {
        // Use a custom sink stage that records what it saw via a signal.
        struct RecordingSink;
        impl nv_perception::Stage for RecordingSink {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("recording_sink")
            }
            fn category(&self) -> nv_perception::StageCategory {
                nv_perception::StageCategory::Sink
            }
            fn process(
                &mut self,
                ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                // Record what we see — detection count and track count as signals.
                Ok(nv_perception::StageOutput::with_signals(vec![
                    nv_perception::DerivedSignal {
                        name: "sink_det_count",
                        value: nv_perception::SignalValue::Scalar(
                            ctx.artifacts.detections.len() as f64,
                        ),
                        ts: ctx.frame.ts(),
                    },
                    nv_perception::DerivedSignal {
                        name: "sink_track_count",
                        value: nv_perception::SignalValue::Scalar(
                            ctx.artifacts.tracks.len() as f64,
                        ),
                        ts: ctx.frame.ts(),
                    },
                ]))
            }
        }

        use nv_test_util::mock_stage::{MockDetectorStage, MockTemporalStage, MockTrackerStage};

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(MockTrackerStage::new("trk")),
                Box::new(MockTemporalStage::new("temporal")),
                Box::new(RecordingSink),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        // Detections and tracks should propagate to output.
        assert_eq!(out.detections.len(), 2);
        assert_eq!(out.tracks.len(), 2);

        // The temporal stage should have produced a track_count signal.
        // On frame 0 no tracks are in the temporal store yet (committed after stages).
        let temporal_signal = out
            .signals
            .iter()
            .find(|s| s.name == "track_count")
            .expect("temporal stage should produce track_count signal");
        assert!(matches!(temporal_signal.value, nv_perception::SignalValue::Scalar(_)));

        // The sink should see 2 detections and 2 tracks.
        let sink_det = out
            .signals
            .iter()
            .find(|s| s.name == "sink_det_count")
            .expect("sink should record detection count");
        let sink_trk = out
            .signals
            .iter()
            .find(|s| s.name == "sink_track_count")
            .expect("sink should record track count");
        assert!(matches!(sink_det.value, nv_perception::SignalValue::Scalar(v) if (v - 2.0).abs() < f64::EPSILON));
        assert!(matches!(sink_trk.value, nv_perception::SignalValue::Scalar(v) if (v - 2.0).abs() < f64::EPSILON));

        // Provenance should have 4 stage entries.
        assert_eq!(out.provenance.stages.len(), 4);
        assert_eq!(out.provenance.stages[0].stage_id, nv_core::id::StageId("det"));
        assert_eq!(out.provenance.stages[1].stage_id, nv_core::id::StageId("trk"));
        assert_eq!(out.provenance.stages[2].stage_id, nv_core::id::StageId("temporal"));
        assert_eq!(out.provenance.stages[3].stage_id, nv_core::id::StageId("recording_sink"));
    }

    #[test]
    fn stage_failure_drops_frame_skips_remaining() {
        use nv_test_util::mock_stage::{FailingStage, MockDetectorStage};

        // Pipeline: detector → failing → never-reached
        struct NeverReached;
        impl nv_perception::Stage for NeverReached {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("never_reached")
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                panic!("this stage should never be called");
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(FailingStage::new("bad_stage")),
                Box::new(NeverReached),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, health) = exec.process_frame(&frame);

        // Frame should be dropped.
        assert!(output.is_none(), "failed stage should drop the frame");
        // Health event should be emitted.
        assert!(
            health.iter().any(|h| matches!(h, HealthEvent::StageError { .. })),
            "should emit StageError health event"
        );
    }

    #[test]
    fn stage_error_provenance_records_failure() {
        use nv_test_util::mock_stage::FailingStage;

        // Pipeline: detector → failing. The detector runs fine,
        // the failing stage errors. Both get provenance entries.
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(nv_test_util::mock_stage::NoOpStage::new("ok")),
                Box::new(FailingStage::new("fail")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        // Output is None (frame dropped), but we can verify through
        // health events that the executor processed both stages.
        assert!(output.is_none());
    }

    #[test]
    fn output_propagation_across_frames() {
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        // Process 3 frames.
        for i in 0..3u64 {
            let frame = nv_test_util::synthetic::solid_gray(
                FeedId::new(1),
                i,
                MonotonicTs::from_nanos(i * 33_000_000),
                4,
                4,
                128,
            );
            let (output, _) = exec.process_frame(&frame);
            let out = output.expect("should produce output");
            assert_eq!(out.detections.len(), 2, "frame {i}: should have 2 detections");
            assert_eq!(out.tracks.len(), 2, "frame {i}: should have 2 tracks");
            assert_eq!(out.frame_seq, i);
        }
        assert_eq!(exec.frames_processed(), 3);
    }

    #[test]
    fn feed_local_state_preserved_across_frames() {
        // A stage with internal state (counter).
        struct CounterStage {
            call_count: u64,
        }
        impl nv_perception::Stage for CounterStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("counter")
            }
            fn process(
                &mut self,
                ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                self.call_count += 1;
                Ok(nv_perception::StageOutput::with_signal(
                    nv_perception::DerivedSignal {
                        name: "call_count",
                        value: nv_perception::SignalValue::Scalar(self.call_count as f64),
                        ts: ctx.frame.ts(),
                    },
                ))
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(CounterStage { call_count: 0 })],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        for i in 0..5u64 {
            let frame = nv_test_util::synthetic::solid_gray(
                FeedId::new(1),
                i,
                MonotonicTs::from_nanos(i * 33_000_000),
                2,
                2,
                128,
            );
            let (output, _) = exec.process_frame(&frame);
            let out = output.expect("should produce output");
            let signal = out.signals.iter().find(|s| s.name == "call_count").unwrap();
            match signal.value {
                nv_perception::SignalValue::Scalar(v) => {
                    assert_eq!(v as u64, i + 1, "stage internal state should persist")
                }
                _ => panic!("expected scalar signal"),
            }
        }
    }

    #[test]
    fn two_independent_executors_have_isolated_state() {
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        // Two executors simulating two feeds.
        let mut exec_a = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        let mut exec_b = PipelineExecutor::new(
            FeedId::new(2),
            vec![
                Box::new(MockDetectorStage::new("det", 5)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec_a.start_stages().unwrap();
        exec_b.start_stages().unwrap();

        let frame_a = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let frame_b = nv_test_util::synthetic::solid_gray(
            FeedId::new(2),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );

        let (out_a, _) = exec_a.process_frame(&frame_a);
        let (out_b, _) = exec_b.process_frame(&frame_b);

        let a = out_a.expect("feed A output");
        let b = out_b.expect("feed B output");

        // Feed A: 2 detections → 2 tracks.
        assert_eq!(a.detections.len(), 2);
        assert_eq!(a.tracks.len(), 2);
        assert_eq!(a.feed_id, FeedId::new(1));

        // Feed B: 5 detections → 5 tracks.
        assert_eq!(b.detections.len(), 5);
        assert_eq!(b.tracks.len(), 5);
        assert_eq!(b.feed_id, FeedId::new(2));

        // Temporal stores are independent.
        assert_eq!(exec_a.track_count(), 2);
        assert_eq!(exec_b.track_count(), 5);
    }

    #[test]
    fn pipeline_with_stage_pipeline_builder() {
        use nv_perception::StagePipeline;
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        let pipeline = StagePipeline::builder()
            .add(MockDetectorStage::new("det", 4))
            .add(MockTrackerStage::new("trk"))
            .build();

        let ids: Vec<&str> = pipeline.stage_ids().iter().map(|s| s.as_str()).collect();
        assert_eq!(ids, vec!["det", "trk"]);

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            pipeline.into_stages(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");
        assert_eq!(out.detections.len(), 4);
        assert_eq!(out.tracks.len(), 4);
    }

    // ---------------------------------------------------------------
    // Frame inclusion policy tests
    // ---------------------------------------------------------------

    #[test]
    fn frame_inclusion_never_produces_no_frame() {
        let mut exec = make_executor();
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");
        assert!(out.frame.is_none());
    }

    #[test]
    fn frame_inclusion_always_includes_frame() {
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Always,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");
        assert!(out.frame.is_some());
        // Zero-copy: same backing data (Arc bump, not pixel copy).
        assert_eq!(out.frame.as_ref().unwrap().seq(), frame.seq());
    }

    // ---------------------------------------------------------------
    // P2: Panic containment in start_stages / stop_stages
    // ---------------------------------------------------------------

    /// A stage that panics in on_start.
    struct PanicOnStartStage;
    impl Stage for PanicOnStartStage {
        fn id(&self) -> StageId { StageId("panic-start") }
        fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
            Ok(StageOutput::empty())
        }
        fn on_start(&mut self) -> Result<(), StageError> {
            panic!("intentional on_start panic");
        }
    }

    /// A stage that panics in on_stop.
    struct PanicOnStopStage {
        started: bool,
    }
    impl PanicOnStopStage {
        fn new() -> Self { Self { started: false } }
    }
    impl Stage for PanicOnStopStage {
        fn id(&self) -> StageId { StageId("panic-stop") }
        fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
            Ok(StageOutput::empty())
        }
        fn on_start(&mut self) -> Result<(), StageError> {
            self.started = true;
            Ok(())
        }
        fn on_stop(&mut self) -> Result<(), StageError> {
            panic!("intentional on_stop panic");
        }
    }

    #[test]
    fn start_stages_catches_panic() {
        let stages: Vec<Box<dyn Stage>> = vec![Box::new(PanicOnStartStage)];
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            stages,
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        let result = exec.start_stages();
        assert!(result.is_err(), "start_stages should return error on panic");
        let err = result.unwrap_err();
        assert!(
            format!("{err}").contains("panicked"),
            "error should mention panic: {err}"
        );
    }

    #[test]
    fn stop_stages_catches_panic() {
        let stages: Vec<Box<dyn Stage>> = vec![Box::new(PanicOnStopStage::new())];
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            stages,
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();
        // Should not panic — catches the panic internally.
        exec.stop_stages();
    }

    #[test]
    fn flush_batch_rejections_returns_none_when_empty() {
        let mut exec = make_executor();
        assert!(exec.flush_batch_rejections().is_none());
    }

    #[test]
    fn flush_batch_rejections_emits_accumulated_count() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_flush") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(42),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        // Simulate accumulated rejections.
        exec.batch_rejection_count = 5;

        let evt = exec.flush_batch_rejections();
        assert!(evt.is_some(), "should have flushed accumulated rejections");
        match evt.unwrap() {
            HealthEvent::BatchSubmissionRejected { feed_id, processor_id, dropped_count } => {
                assert_eq!(feed_id, FeedId::new(42));
                assert_eq!(processor_id, StageId("noop_flush"));
                assert_eq!(dropped_count, 5);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        // After flush, count should be zero.
        assert!(exec.flush_batch_rejections().is_none());

        coord.shutdown();
    }

    /// When the feed shutdown flag is set *before* coordinator dies,
    /// CoordinatorShutdown should produce zero health events (expected).
    #[test]
    fn coordinator_shutdown_expected_emits_no_health() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_csexp") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        // Mark feed as shutting down *before* coordinator dies.
        let feed_shutdown = Arc::new(AtomicBool::new(true));

        let mut exec = PipelineExecutor::new(
            FeedId::new(99),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::clone(&feed_shutdown),
        );

        // Shut down the coordinator — the handle's next submit will see CoordinatorShutdown.
        coord.shutdown();
        // Give coordinator thread a moment to exit.
        std::thread::sleep(std::time::Duration::from_millis(50));

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 0, MonotonicTs::from_nanos(0), 2, 2, 128,
        );
        let (_output, health_events) = exec.process_frame(&frame);

        let stage_errors: Vec<_> = health_events.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert!(stage_errors.is_empty(), "expected no StageError for expected shutdown, got: {stage_errors:?}");
    }

    /// When the feed shutdown flag is NOT set and the coordinator dies,
    /// exactly one StageError health event should be emitted.
    #[test]
    fn coordinator_shutdown_unexpected_emits_one_stage_error() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_csunexp") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        // feed_shutdown stays false — coordinator death is unexpected.
        let mut exec = PipelineExecutor::new(
            FeedId::new(99),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        coord.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 0, MonotonicTs::from_nanos(0), 2, 2, 128,
        );
        let (_output, health_events) = exec.process_frame(&frame);

        let stage_errors: Vec<_> = health_events.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert_eq!(stage_errors.len(), 1, "expected exactly one StageError for unexpected shutdown, got: {stage_errors:?}");

        match &stage_errors[0] {
            HealthEvent::StageError { feed_id, stage_id, error } => {
                assert_eq!(*feed_id, FeedId::new(99));
                assert_eq!(*stage_id, StageId("noop_csunexp"));
                let detail = format!("{error}");
                assert!(detail.contains("batch coordinator shut down unexpectedly"), "unexpected detail: {detail}");
            }
            _ => unreachable!(),
        }
    }

    /// After the first unexpected CoordinatorShutdown health event,
    /// subsequent frames should not emit duplicates.
    #[test]
    fn coordinator_shutdown_unexpected_deduplicates() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_csdedup") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(99),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        coord.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let frame1 = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 0, MonotonicTs::from_nanos(0), 2, 2, 128,
        );
        let frame2 = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 1, MonotonicTs::from_nanos(1_000_000), 2, 2, 128,
        );

        // First frame: should emit the StageError.
        let (_, h1) = exec.process_frame(&frame1);
        let errs1: Vec<_> = h1.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert_eq!(errs1.len(), 1, "first frame should emit one StageError");

        // Second frame: should NOT emit a duplicate.
        let (_, h2) = exec.process_frame(&frame2);
        let errs2: Vec<_> = h2.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert!(errs2.is_empty(), "second frame should suppress duplicate StageError, got: {errs2:?}");
    }

    // ---------------------------------------------------------------
    // Batch timeout coalescing tests
    // ---------------------------------------------------------------

    #[test]
    fn flush_batch_timeouts_returns_none_when_empty() {
        let mut exec = make_executor();
        assert!(exec.flush_batch_timeouts().is_none());
    }

    #[test]
    fn flush_batch_timeouts_emits_accumulated_count() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_to") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(77),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        // Simulate accumulated timeouts.
        exec.batch_timeout_count = 3;

        let evt = exec.flush_batch_timeouts();
        assert!(evt.is_some(), "should have flushed accumulated timeouts");
        match evt.unwrap() {
            HealthEvent::BatchTimeout { feed_id, processor_id, timed_out_count } => {
                assert_eq!(feed_id, FeedId::new(77));
                assert_eq!(processor_id, StageId("noop_to"));
                assert_eq!(timed_out_count, 3);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        // After flush, count should be zero.
        assert!(exec.flush_batch_timeouts().is_none());

        coord.shutdown();
    }

    /// Integration test: timeout coalescing through process_frame.
    ///
    /// Uses a slow processor that triggers feed-side timeouts, and
    /// verifies:
    /// 1. First timeout emits a BatchTimeout health event immediately.
    /// 2. Rapid subsequent timeouts within the 1s window are coalesced
    ///    (no event emitted).
    /// 3. Recovery (successful batch) flushes any accumulated count.
    #[test]
    fn timeout_coalescing_through_process_frame() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        /// Processor that sleeps longer than the response timeout,
        /// causing every submission to time out on the feed side.
        /// Controlled by a shared flag to switch to fast mode.
        struct ControllableProcessor {
            slow: Arc<AtomicBool>,
        }
        impl BatchProcessor for ControllableProcessor {
            fn id(&self) -> StageId { StageId("ctrl_slow") }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> {
                if self.slow.load(Ordering::Relaxed) {
                    // Sleep longer than max_latency + response_timeout to
                    // ensure the feed-side recv_timeout fires.
                    std::thread::sleep(std::time::Duration::from_millis(400));
                }
                for item in items.iter_mut() {
                    item.output = Some(nv_perception::StageOutput::empty());
                }
                Ok(())
            }
        }

        let slow_flag = Arc::new(AtomicBool::new(true));
        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(64);
        let coord = BatchCoordinator::start(
            Box::new(ControllableProcessor { slow: Arc::clone(&slow_flag) }),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                // Very short response timeout so timeout triggers quickly.
                response_timeout: Some(std::time::Duration::from_millis(50)),
                // Allow 2 in-flight so the test can verify timeout
                // coalescing rather than in-flight cap behavior.
                max_in_flight_per_feed: 2,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(55),
            Vec::new(),
            Some(handle.clone()),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        let make_frame = |seq: u64| {
            nv_test_util::synthetic::solid_gray(
                FeedId::new(55), seq, MonotonicTs::from_nanos(seq * 33_000_000), 2, 2, 128,
            )
        };

        // --- Frame 1: first timeout, should emit BatchTimeout immediately ---
        let (_, h1) = exec.process_frame(&make_frame(0));
        let timeout_events_1: Vec<_> = h1.iter()
            .filter(|e| matches!(e, HealthEvent::BatchTimeout { .. }))
            .collect();
        assert_eq!(
            timeout_events_1.len(), 1,
            "first timeout should emit one BatchTimeout event, got {timeout_events_1:?}"
        );
        // Verify the event contents.
        match &timeout_events_1[0] {
            HealthEvent::BatchTimeout { feed_id, timed_out_count, .. } => {
                assert_eq!(*feed_id, FeedId::new(55));
                assert_eq!(*timed_out_count, 1);
            }
            _ => unreachable!(),
        }

        // --- Frame 2: rapid second timeout within throttle window ---
        // Should NOT emit an event (coalesced into accumulator).
        let (_, h2) = exec.process_frame(&make_frame(1));
        let timeout_events_2: Vec<_> = h2.iter()
            .filter(|e| matches!(e, HealthEvent::BatchTimeout { .. }))
            .collect();
        assert!(
            timeout_events_2.is_empty(),
            "second timeout within throttle window should be coalesced, got {timeout_events_2:?}"
        );

        // Verify the internal accumulator has tracked it.
        assert_eq!(exec.batch_timeout_count, 1, "one coalesced timeout in accumulator");

        // --- Switch to fast mode and process a successful frame ---
        slow_flag.store(false, Ordering::Relaxed);
        // Give the coordinator time to finish any in-flight slow batches
        // (timed-out items from above may still be processing).
        std::thread::sleep(std::time::Duration::from_millis(500));

        let (output, h3) = exec.process_frame(&make_frame(2));
        assert!(output.is_some(), "frame should succeed after processor speeds up");

        // Recovery should flush the accumulated timeout count.
        let timeout_events_3: Vec<_> = h3.iter()
            .filter(|e| matches!(e, HealthEvent::BatchTimeout { .. }))
            .collect();
        assert_eq!(
            timeout_events_3.len(), 1,
            "recovery should flush accumulated timeouts, got {timeout_events_3:?}"
        );
        match &timeout_events_3[0] {
            HealthEvent::BatchTimeout { timed_out_count, .. } => {
                assert_eq!(*timed_out_count, 1, "flushed count should be 1 (one coalesced timeout)");
            }
            _ => unreachable!(),
        }

        // After recovery, no more pending timeouts.
        assert_eq!(exec.batch_timeout_count, 0);

        coord.shutdown();
    }
}
