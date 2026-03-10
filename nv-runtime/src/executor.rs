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

use nv_core::config::CameraMode;
use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_core::metrics::StageMetrics;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_core::TypedMetadata;
use nv_frame::FrameEnvelope;
use nv_perception::{PerceptionArtifacts, Stage, StageContext, StageOutput};
use nv_temporal::store::TrackHistory;
use nv_temporal::{RetentionPolicy, TemporalStore, Trajectory};
use nv_view::{
    CameraMotionState, EpochDecision, EpochPolicy, EpochPolicyContext, MotionPollContext,
    MotionReport, MotionSource, ViewSnapshot, ViewState, ViewStateProvider,
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
        let state_duration = Duration::from_nanos(
            self.motion_state_start.elapsed().as_nanos() as u64,
        );

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
                self.view_state.validity =
                    nv_view::ContextValidity::Degraded { reason: reason.clone() };
                self.view_state.stability_score =
                    (self.view_state.stability_score - 0.2).max(0.0);
                self.view_state.version = self.view_state.version.next();

                health_events.push(HealthEvent::ViewDegraded {
                    feed_id: self.feed_id,
                    stability_score: self.view_state.stability_score,
                });
            }
            EpochDecision::Compensate { reason, .. } => {
                self.view_state.motion = motion_state;
                self.view_state.motion_source = motion_source.clone();
                self.view_state.ptz = report.ptz;
                self.view_state.global_transform = report.frame_transform;
                self.view_state.validity =
                    nv_view::ContextValidity::Degraded { reason: reason.clone() };
                self.view_state.stability_score =
                    (self.view_state.stability_score - 0.1).max(0.0);
                let new_epoch = self.view_state.epoch;
                self.view_state.version = self.view_state.version.next();

                health_events.push(HealthEvent::ViewCompensationApplied {
                    feed_id: self.feed_id,
                    epoch: new_epoch.as_u64(),
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

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stage.process(&ctx)
            }));

            let t_stage_end = Instant::now();
            let stage_latency =
                Duration::from_nanos(t_stage_start.elapsed().as_nanos() as u64);

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
        // Commit the merged track set into the temporal store.
        for track in &artifacts.tracks {
            let track_id = track.id;
            let now_ts = frame.ts();
            if let Some(history) = self.temporal.get_track(&track_id) {
                let _ = history; // exists — we update via push_observation below
            }
            // Upsert track: if absent, create a new TrackHistory.
            // Use the track data from artifacts directly.
            if self.temporal.get_track(&track_id).is_none() {
                let history = TrackHistory::new(
                    Arc::new(track.clone()),
                    Arc::new(Trajectory::new()),
                    now_ts,
                    now_ts,
                    self.view_state.epoch,
                );
                self.temporal.insert_track(track_id, history);
            }
            // Update last_seen and track state.
            if let Some(history) = self.temporal.get_track_mut(&track_id) {
                history.track = Arc::new(track.clone());
                history.last_seen = now_ts;
            }
        }

        let t_pipeline_end = Instant::now();
        let pipeline_complete_ts = self.instant_to_ts(t_pipeline_end);
        let total_latency =
            Duration::from_nanos(t_pipeline_start.elapsed().as_nanos() as u64);

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
            metadata: TypedMetadata::new(),
        };

        (Some(output), health_events)
    }

    /// Number of frames processed by this executor.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Clear temporal state (called on feed restart).
    pub fn clear_temporal(&mut self) {
        self.temporal.clear();
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
