use std::time::Instant;

use nv_core::config::CameraMode;
use nv_core::health::HealthEvent;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_frame::FrameEnvelope;
use nv_view::{
    CameraMotionState, EpochDecision, EpochPolicyContext, MotionPollContext, MotionReport,
    MotionSource, ViewSnapshot,
};

use super::PipelineExecutor;
use super::lifecycle::notify_epoch_change;

impl PipelineExecutor {
    /// Convert a wall-clock [`Instant`] to a [`MonotonicTs`] anchored to
    /// the executor's creation time.
    pub(super) fn instant_to_ts(&self, t: Instant) -> MonotonicTs {
        super::instant_to_ts_impl(self.clock_anchor, self.clock_anchor_ts, t)
    }

    /// Apply the four motion-related fields that every [`EpochDecision`]
    /// branch writes, plus bump the version counter.
    pub(super) fn apply_motion_fields(
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
    pub(super) fn update_view(
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
                self.view_state.validity = nv_view::ContextValidity::Degraded { reason: *reason };
                self.view_state.stability_score = (self.view_state.stability_score - 0.2).max(0.0);

                health_events.push(HealthEvent::ViewDegraded {
                    feed_id: self.feed_id,
                    stability_score: self.view_state.stability_score,
                });
            }
            EpochDecision::Compensate { reason, transform } => {
                self.view_state.validity = nv_view::ContextValidity::Degraded { reason: *reason };
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
