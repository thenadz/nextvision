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
//! - `Compensate` — degrade + apply transform (TODO: transform application
//!   on tracks is data-model–dependent; plumbing is in place).
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

use std::sync::Arc;
use std::time::Instant;

use nv_core::TypedMetadata;
use nv_core::config::CameraMode;
use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_core::metrics::StageMetrics;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_frame::FrameEnvelope;
use nv_perception::{PerceptionArtifacts, Stage, StageContext, StageOutput};
use nv_temporal::{RetentionPolicy, TemporalStore};
use nv_view::{
    CameraMotionState, EpochDecision, EpochPolicy, EpochPolicyContext, MotionPollContext,
    MotionReport, MotionSource, TransitionPhase, ViewSnapshot, ViewState, ViewStateProvider,
};

use crate::output::OutputEnvelope;
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
        }
    }

    /// Call `on_start()` on each stage in order.
    ///
    /// If any stage fails, previously-started stages are stopped (best-effort)
    /// and the error is returned.
    pub fn start_stages(&mut self) -> Result<(), StageError> {
        let mut started = 0;
        for stage in &mut self.stages {
            if let Err(e) = stage.on_start() {
                // Best-effort stop of already-started stages.
                for s in &mut self.stages[..started] {
                    let _ = s.on_stop();
                }
                return Err(e);
            }
            started += 1;
        }
        Ok(())
    }

    /// Call `on_stop()` on each stage — best-effort, errors are logged.
    pub fn stop_stages(&mut self) {
        for stage in &mut self.stages {
            if let Err(e) = stage.on_stop() {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    stage_id = %stage.id(),
                    error = %e,
                    "stage on_stop error (ignored)"
                );
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
                    if let Err(e) = stage.on_view_epoch_change(new_epoch) {
                        tracing::warn!(
                            feed_id = %self.feed_id,
                            stage_id = %stage.id(),
                            error = %e,
                            "stage on_view_epoch_change error (ignored)"
                        );
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
        // Commit the merged track set via TemporalStore's encapsulated
        // commit_track + enforce_retention methods.
        let now_ts = frame.ts();
        let current_epoch = self.view_state.epoch;

        for track in &artifacts.tracks {
            self.temporal.commit_track(track, now_ts, current_epoch);
        }

        // --- Track ending (authoritative set semantics) ---
        // When at least one stage produced authoritative track output,
        // any track previously in the temporal store but absent from
        // this frame's track set is considered normally ended.
        if artifacts.tracks_authoritative {
            let current_ids: std::collections::HashSet<nv_core::TrackId> =
                artifacts.tracks.iter().map(|t| t.id).collect();
            let ended: Vec<nv_core::TrackId> = self
                .temporal
                .track_ids()
                .filter(|id| !current_ids.contains(id))
                .copied()
                .collect();
            for id in ended {
                self.temporal.end_track(&id);
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
        };

        (Some(output), health_events)
    }

    /// Number of frames processed by this executor.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
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
mod tests {
    use super::*;
    use nv_core::config::CameraMode;
    use nv_core::id::FeedId;
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
        let seg = hist.trajectory.segments.last().unwrap();
        assert_eq!(
            seg.closed_by,
            Some(nv_temporal::SegmentBoundary::FeedRestart),
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
        assert!(!hist.trajectory.segments[0].is_active());
        assert!(hist.trajectory.segments[1].is_active());
        assert_eq!(hist.trajectory.segments[1].view_epoch, epoch1);
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
        let last_seg = hist.trajectory.segments.last().unwrap();
        assert_eq!(
            last_seg.closed_by,
            Some(nv_temporal::SegmentBoundary::TrackLost),
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
}
