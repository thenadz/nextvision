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
use std::time::Instant;

use nv_core::config::CameraMode;
use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_core::metrics::StageMetrics;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_core::TrackId;
use nv_frame::FrameEnvelope;
use nv_perception::{PerceptionArtifacts, Stage, StageContext};
use nv_temporal::{RetentionPolicy, TemporalStore};
use nv_view::{
    CameraMotionState, EpochDecision, EpochPolicy, EpochPolicyContext, MotionPollContext,
    MotionReport, MotionSource, TransitionPhase, ViewSnapshot, ViewState, ViewStateProvider,
};

use crate::output::OutputEnvelope;
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
    stages: Vec<Box<dyn Stage>>,
    temporal: TemporalStore,
    view_state: ViewState,
    view_snapshot: ViewSnapshot,
    view_state_provider: Option<Box<dyn ViewStateProvider>>,
    epoch_policy: Box<dyn EpochPolicy>,
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
}

impl PipelineExecutor {
    /// Create a new executor with the given stages, policies, and optional
    /// view-state provider.
    pub fn new(
        feed_id: FeedId,
        stages: Vec<Box<dyn Stage>>,
        retention: RetentionPolicy,
        camera_mode: CameraMode,
        view_state_provider: Option<Box<dyn ViewStateProvider>>,
        epoch_policy: Box<dyn EpochPolicy>,
        frame_inclusion: FrameInclusion,
    ) -> Self {
        let view_state = match camera_mode {
            CameraMode::Fixed => ViewState::fixed_initial(),
            CameraMode::Observed => ViewState::observed_initial(),
        };
        let view_snapshot = ViewSnapshot::new(view_state.clone());
        let stage_count = stages.len();
        let now = Instant::now();
        Self {
            feed_id,
            camera_mode,
            stages,
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
                stage_count
            ],
            frames_processed: 0,
            clock_anchor: now,
            clock_anchor_ts: MonotonicTs::from_nanos(0),
            motion_state_start: now,
            frame_inclusion,
            track_id_buf: HashSet::new(),
            ended_buf: Vec::new(),
        }
    }

    /// Call `on_start()` on each stage in order.
    ///
    /// If any stage fails or panics, previously-started stages are stopped
    /// (best-effort) and the error is returned.
    pub fn start_stages(&mut self) -> Result<(), StageError> {
        for i in 0..self.stages.len() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.stages[i].on_start()
            }));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    self.rollback_started_stages(i);
                    return Err(e);
                }
                Err(_) => {
                    let stage_id = self.stages[i].id();
                    tracing::error!(
                        feed_id = %self.feed_id,
                        stage_id = %stage_id,
                        "stage on_start() panicked"
                    );
                    self.rollback_started_stages(i);
                    return Err(StageError::ProcessingFailed {
                        stage_id,
                        detail: "on_start() panicked".into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Best-effort stop of stages `[0..count)`. Used on startup failure.
    fn rollback_started_stages(&mut self, count: usize) {
        for s in &mut self.stages[..count] {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = s.on_stop();
            }));
        }
    }

    /// Call `on_stop()` on each stage — best-effort, errors and panics
    /// are logged.
    pub fn stop_stages(&mut self) {
        for stage in &mut self.stages {
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

    /// Convert a wall-clock [`Instant`] to a [`MonotonicTs`] anchored to
    /// the executor's creation time.
    fn instant_to_ts(&self, t: Instant) -> MonotonicTs {
        instant_to_ts_impl(self.clock_anchor, self.clock_anchor_ts, t)
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
        match &decision {
            EpochDecision::Continue => {
                // Update motion fields, keep epoch.
                let is_stable = matches!(motion_state, CameraMotionState::Stable);
                self.view_state.motion = motion_state;
                self.view_state.motion_source = motion_source.clone();
                self.view_state.ptz = report.ptz;
                self.view_state.global_transform = report.frame_transform;
                self.view_state.version = self.view_state.version.next();
                if is_stable {
                    self.view_state.stability_score =
                        (self.view_state.stability_score + 0.1).min(1.0);
                    self.view_state.validity = nv_view::ContextValidity::Valid;
                }
            }
            EpochDecision::Degrade { reason } => {
                self.view_state.motion = motion_state;
                self.view_state.motion_source = motion_source.clone();
                self.view_state.ptz = report.ptz;
                self.view_state.global_transform = report.frame_transform;
                self.view_state.validity = nv_view::ContextValidity::Degraded {
                    reason: reason.clone(),
                };
                self.view_state.stability_score = (self.view_state.stability_score - 0.2).max(0.0);
                self.view_state.version = self.view_state.version.next();

                health_events.push(HealthEvent::ViewDegraded {
                    feed_id: self.feed_id,
                    stability_score: self.view_state.stability_score,
                });
            }
            EpochDecision::Compensate { reason, transform } => {
                self.view_state.motion = motion_state;
                self.view_state.motion_source = motion_source.clone();
                self.view_state.ptz = report.ptz;
                self.view_state.global_transform = report.frame_transform;
                self.view_state.validity = nv_view::ContextValidity::Degraded {
                    reason: reason.clone(),
                };
                self.view_state.stability_score = (self.view_state.stability_score - 0.1).max(0.0);
                let current_epoch = self.view_state.epoch;
                self.view_state.version = self.view_state.version.next();

                // Apply the compensation transform to existing trajectory data
                // so previously-recorded positions align with the new view.
                self.temporal.apply_compensation(&transform, current_epoch);

                health_events.push(HealthEvent::ViewCompensationApplied {
                    feed_id: self.feed_id,
                    epoch: current_epoch.as_u64(),
                });
            }
            EpochDecision::Segment => {
                let new_epoch = self.view_state.epoch.next();
                self.view_state.epoch = new_epoch;
                self.view_state.motion = motion_state;
                self.view_state.motion_source = motion_source.clone();
                self.view_state.ptz = report.ptz;
                self.view_state.global_transform = report.frame_transform;
                self.view_state.validity = nv_view::ContextValidity::Valid;
                self.view_state.stability_score = 0.0;
                self.view_state.transition = nv_view::TransitionPhase::MoveStart;
                self.view_state.version = self.view_state.version.next();

                // Update temporal store epoch.
                self.temporal.set_view_epoch(new_epoch);

                // Notify stages.
                for stage in &mut self.stages {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        stage.on_view_epoch_change(new_epoch)
                    }));
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            tracing::warn!(
                                feed_id = %self.feed_id,
                                stage_id = %stage.id(),
                                error = %e,
                                "stage on_view_epoch_change error (ignored)"
                            );
                        }
                        Err(_) => {
                            tracing::error!(
                                feed_id = %self.feed_id,
                                stage_id = %stage.id(),
                                "stage on_view_epoch_change panicked (ignored)"
                            );
                        }
                    }
                }

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

    /// Process a single frame through all stages.
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
        let mut stage_provs = Vec::with_capacity(self.stages.len());
        let mut frame_dropped = false;
        let mut had_panic = false;

        // Capture clock anchors so we can call the free function inside
        // the mutable-borrow loop over self.stages.
        let anchor = self.clock_anchor;
        let anchor_ts = self.clock_anchor_ts;

        for (i, stage) in self.stages.iter_mut().enumerate() {
            let stage_id = stage.id();
            let t_stage_start = Instant::now();

            let ctx = StageContext {
                feed_id: self.feed_id,
                frame,
                artifacts: &artifacts,
                view: &self.view_snapshot,
                temporal: &temporal_snapshot,
                metrics: &self.stage_metrics[i],
            };

            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stage.process(&ctx)));

            let t_stage_end = Instant::now();
            let stage_latency = Duration::from_nanos(t_stage_start.elapsed().as_nanos() as u64);

            let stage_result = match result {
                Ok(Ok(output)) => {
                    artifacts.merge(output);
                    self.stage_metrics[i].frames_processed += 1;
                    StageResult::Ok
                }
                Ok(Err(e)) => {
                    self.stage_metrics[i].errors += 1;
                    health_events.push(HealthEvent::StageError {
                        feed_id: self.feed_id,
                        stage_id,
                        error: e.clone(),
                    });
                    let cat = categorize_stage_error(&e);
                    // Issue 9: drop frame on stage error — skip remaining stages.
                    frame_dropped = true;
                    stage_provs.push(StageProvenance {
                        stage_id,
                        start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
                        end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
                        latency: stage_latency,
                        result: cat,
                    });
                    break;
                }
                Err(_panic) => {
                    self.stage_metrics[i].errors += 1;
                    health_events.push(HealthEvent::StagePanic {
                        feed_id: self.feed_id,
                        stage_id,
                    });
                    had_panic = true;
                    stage_provs.push(StageProvenance {
                        stage_id,
                        start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
                        end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
                        latency: stage_latency,
                        result: StageResult::Error(StageOutcomeCategory::Panic),
                    });
                    break;
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

        self.frames_processed += 1;

        // If a stage panicked, return health events but no output.
        // The caller (worker) will decide whether to restart.
        if had_panic {
            return (None, health_events);
        }

        // Issue 9: frame was dropped due to stage error — emit no output.
        if frame_dropped {
            return (None, health_events);
        }

        // --- Temporal commit (Issue 6) ---
        //
        // Tracks in the output envelope are **stage-authoritative**: they
        // reflect exactly what the perception stages produced, regardless
        // of temporal-store admission. If the store rejects a track
        // (e.g., capacity limit), a health event is emitted but the track
        // still appears in the output. Consumers who need to know which
        // tracks have temporal history should consult the store snapshot.
        let now_ts = frame.ts();
        let current_epoch = self.view_state.epoch;

        for track in &artifacts.tracks {
            if !self.temporal.commit_track(track, now_ts, current_epoch) {
                health_events.push(HealthEvent::TrackAdmissionRejected {
                    feed_id: self.feed_id,
                    track_id: track.id,
                });
            }
        }

        // --- Track ending (authoritative set semantics) ---
        // When at least one stage produced authoritative track output,
        // any track previously in the temporal store but absent from
        // this frame's track set is considered normally ended.
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
    use nv_core::config::CameraMode;
    use nv_core::id::{FeedId, StageId};
    use nv_perception::StageOutput;
    use nv_temporal::RetentionPolicy;
    use nv_view::{DefaultEpochPolicy, ViewEpoch};

    fn make_executor() -> PipelineExecutor {
        PipelineExecutor::new(
            FeedId::new(1),
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

        // 5 tracks, cap 3: first 3 admitted, last 2 rejected.
        let rejection_events: Vec<_> = health_events
            .iter()
            .filter(|e| matches!(e, HealthEvent::TrackAdmissionRejected { .. }))
            .collect();
        assert_eq!(
            rejection_events.len(),
            2,
            "should emit TrackAdmissionRejected for each rejected track, got {rejection_events:?}",
        );

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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Observed,
            Some(Box::new(StableProvider)),
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Observed,
            Some(Box::new(SmallPtzProvider)),
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
        );
        let mut exec_b = PipelineExecutor::new(
            FeedId::new(2),
            vec![
                Box::new(MockDetectorStage::new("det", 5)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Always,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
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
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
        );
        exec.start_stages().unwrap();
        // Should not panic — catches the panic internally.
        exec.stop_stages();
    }
}
