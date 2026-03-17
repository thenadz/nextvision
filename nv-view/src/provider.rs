//! [`ViewStateProvider`] trait and motion report types.

use crate::camera_motion::CameraMotionState;
use crate::ptz::PtzTelemetry;
use crate::transform::GlobalTransformEstimate;
use crate::view_state::ViewState;
use nv_core::MonotonicTs;
use nv_frame::FrameEnvelope;

/// What the view system receives from the provider each frame.
///
/// The provider fills in whichever fields are available. The view system
/// derives [`MotionSource`](crate::MotionSource) from the field contents
/// (see the architecture spec §9 for derivation rules).
#[derive(Clone, Debug, Default)]
pub struct MotionReport {
    /// PTZ telemetry, if available from the camera's control interface.
    pub ptz: Option<PtzTelemetry>,

    /// Frame-to-frame transform estimate (from optical flow, homography, etc.).
    ///
    /// This is the primary egomotion signal when PTZ telemetry is absent.
    pub frame_transform: Option<GlobalTransformEstimate>,

    /// Optional hint about whether the camera is moving.
    ///
    /// If `None`, the view system infers motion from the `ptz` deltas
    /// or `frame_transform` displacement magnitude.
    pub motion_hint: Option<CameraMotionState>,

    /// Discrete PTZ control events received since the previous frame.
    ///
    /// Empty when no PTZ command stream is available. Providers that
    /// monitor an ONVIF event channel or serial command bus populate
    /// this with the events that arrived between frames.
    ///
    /// The epoch policy considers these events alongside telemetry and
    /// inferred motion to make segmentation decisions.
    pub ptz_events: Vec<crate::ptz::PtzEvent>,
}

/// Context given to [`ViewStateProvider::poll`].
pub struct MotionPollContext<'a> {
    /// Monotonic timestamp of the current frame.
    pub ts: MonotonicTs,
    /// The current video frame (available for egomotion providers that
    /// need pixel data for optical flow or feature matching).
    pub frame: &'a FrameEnvelope,
    /// The previous frame's view state.
    pub previous_view: &'a ViewState,
}

/// User-implementable trait: provides camera motion information each frame.
///
/// Required when [`CameraMode::Observed`](nv_core::CameraMode::Observed) is configured.
///
/// ## Implementation categories
///
/// - **Telemetry providers**: poll ONVIF/serial, populate `ptz` field.
/// - **Egomotion providers**: run optical flow or feature matching on the frame,
///   populate `frame_transform` field.
/// - **External providers**: receive transforms or hints from an outside system.
///
/// ## Performance
///
/// `poll()` is called on the stage thread, synchronously, **before** any stage
/// executes. Its latency adds directly to every frame's pipeline latency.
///
/// - Telemetry providers should pre-fetch asynchronously and return cached data.
/// - Egomotion providers run computation directly (typically 1–5ms).
/// - External providers should return pre-computed data.
pub trait ViewStateProvider: Send + Sync + 'static {
    /// Called once per frame. Return the current motion report.
    fn poll(&self, ctx: &MotionPollContext<'_>) -> MotionReport;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_report_default_is_empty() {
        let report = MotionReport::default();
        assert!(report.ptz.is_none());
        assert!(report.frame_transform.is_none());
        assert!(report.motion_hint.is_none());
        assert!(report.ptz_events.is_empty());
    }

    #[test]
    fn motion_report_with_ptz() {
        use crate::ptz::PtzTelemetry;
        let report = MotionReport {
            ptz: Some(PtzTelemetry {
                pan: 90.0,
                tilt: 0.0,
                zoom: 0.5,
                ts: MonotonicTs::from_nanos(100),
            }),
            ..Default::default()
        };
        assert!(report.ptz.is_some());
        assert!(report.frame_transform.is_none());
    }
}
