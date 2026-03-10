//! Epoch policy trait and default implementation.

use nv_core::{AffineTransform2D, Duration};

use crate::camera_motion::CameraMotionState;
use crate::provider::MotionReport;
use crate::validity::DegradationReason;
use crate::view_state::ViewState;

/// Decision returned by an [`EpochPolicy`] when camera motion is detected.
#[derive(Clone, Debug)]
pub enum EpochDecision {
    /// No temporal state change. View is continuous despite motion.
    Continue,

    /// Degrade context validity but keep the current epoch.
    Degrade { reason: DegradationReason },

    /// Degrade, but also apply a compensation transform to existing
    /// track positions so they align with the new view.
    Compensate {
        reason: DegradationReason,
        transform: AffineTransform2D,
    },

    /// Full segmentation: increment epoch, close trajectory segments,
    /// notify stages.
    Segment,
}

/// Context passed to [`EpochPolicy::decide`].
pub struct EpochPolicyContext<'a> {
    /// The view state from the previous frame.
    pub previous_view: &'a ViewState,
    /// The motion report for the current frame.
    pub current_report: &'a MotionReport,
    /// The computed motion state for the current frame.
    pub motion_state: CameraMotionState,
    /// How long the camera has been in the current motion state.
    pub state_duration: Duration,
}

/// User-implementable trait: controls the response to detected camera motion.
///
/// The library ships [`DefaultEpochPolicy`] for common threshold-based decisions.
/// Users with complex PTZ deployments can implement custom policies.
pub trait EpochPolicy: Send + Sync + 'static {
    /// Decide what to do in response to detected camera motion.
    fn decide(&self, ctx: &EpochPolicyContext<'_>) -> EpochDecision;
}

/// Default epoch policy based on configurable thresholds.
///
/// Covers the common case: large PTZ jumps → segment, small motions → degrade,
/// high-confidence transforms → compensate.
#[derive(Debug, Clone)]
pub struct DefaultEpochPolicy {
    /// Pan/tilt delta (degrees) above which a PTZ move triggers `Segment`.
    /// Below this, triggers `Degrade`. Default: `15.0`.
    pub segment_angle_threshold: f32,

    /// Zoom ratio change above which a zoom move triggers `Segment`.
    /// Default: `0.3`.
    pub segment_zoom_threshold: f32,

    /// Inferred-motion displacement (normalized coords) above which
    /// triggers `Segment`. Default: `0.25`.
    pub segment_displacement_threshold: f32,

    /// Minimum confidence for a `Compensate` decision (instead of `Segment`)
    /// when a transform is available. Default: `0.8`.
    pub compensate_min_confidence: f32,

    /// If `true`, small motions below segment thresholds produce `Degrade`
    /// instead of `Continue`. Default: `true`.
    pub degrade_on_small_motion: bool,
}

impl Default for DefaultEpochPolicy {
    fn default() -> Self {
        Self {
            segment_angle_threshold: 15.0,
            segment_zoom_threshold: 0.3,
            segment_displacement_threshold: 0.25,
            compensate_min_confidence: 0.8,
            degrade_on_small_motion: true,
        }
    }
}

impl EpochPolicy for DefaultEpochPolicy {
    fn decide(&self, ctx: &EpochPolicyContext<'_>) -> EpochDecision {
        // Check PTZ telemetry for large moves.
        if let (Some(prev_ptz), Some(curr_ptz)) = (
            ctx.previous_view.ptz.as_ref(),
            ctx.current_report.ptz.as_ref(),
        ) {
            let pan_delta = (curr_ptz.pan - prev_ptz.pan).abs();
            let tilt_delta = (curr_ptz.tilt - prev_ptz.tilt).abs();
            let zoom_delta = (curr_ptz.zoom - prev_ptz.zoom).abs();

            if pan_delta > self.segment_angle_threshold
                || tilt_delta > self.segment_angle_threshold
                || zoom_delta > self.segment_zoom_threshold
            {
                // Check if compensation is possible.
                if let Some(ref transform) = ctx.current_report.frame_transform {
                    if transform.confidence >= self.compensate_min_confidence {
                        return EpochDecision::Compensate {
                            reason: DegradationReason::PtzMoving,
                            transform: transform.transform,
                        };
                    }
                }
                return EpochDecision::Segment;
            }

            if self.degrade_on_small_motion
                && (pan_delta > 0.0 || tilt_delta > 0.0 || zoom_delta > 0.0)
            {
                return EpochDecision::Degrade {
                    reason: DegradationReason::PtzMoving,
                };
            }
        }

        // Check inferred displacement.
        if let CameraMotionState::Moving { displacement, .. } = &ctx.motion_state {
            if let Some(disp) = displacement {
                if *disp > self.segment_displacement_threshold {
                    if let Some(ref transform) = ctx.current_report.frame_transform {
                        if transform.confidence >= self.compensate_min_confidence {
                            return EpochDecision::Compensate {
                                reason: DegradationReason::LargeJump,
                                transform: transform.transform,
                            };
                        }
                    }
                    return EpochDecision::Segment;
                }
                if self.degrade_on_small_motion && *disp > 0.0 {
                    return EpochDecision::Degrade {
                        reason: DegradationReason::InferredMotionLowConfidence,
                    };
                }
            }
        }

        EpochDecision::Continue
    }
}
