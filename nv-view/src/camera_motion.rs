//! Camera motion state and motion source types.

/// Whether the camera is stable, moving, or in an unknown state.
///
/// Central to the view system's decision-making. Stages receive this
/// through [`ViewSnapshot`](crate::ViewSnapshot).
#[derive(Clone, Debug, PartialEq)]
pub enum CameraMotionState {
    /// Camera is not moving. Coordinates are stable across frames.
    Stable,

    /// Camera is actively moving.
    Moving {
        /// Angular velocity in degrees/second, if known.
        angular_velocity: Option<f32>,
        /// Estimated frame-to-frame displacement magnitude (normalized coordinates).
        displacement: Option<f32>,
    },

    /// Camera motion state is unknown.
    ///
    /// Occurs when no telemetry is available, no estimator is configured,
    /// or the estimator's confidence is below threshold. Treated as potentially
    /// moving for safety — never assumed stable.
    Unknown,
}

/// How the current motion state was determined.
///
/// Critical for downstream trust decisions — telemetry-sourced motion
/// is more trustworthy than inferred motion from noisy optical flow.
#[derive(Clone, Debug, PartialEq)]
pub enum MotionSource {
    /// Determined from PTZ telemetry (ONVIF, serial, etc.).
    Telemetry,

    /// Inferred from video analysis (optical flow, feature matching, homography).
    Inferred {
        /// Confidence score in `[0.0, 1.0]`.
        confidence: f32,
    },

    /// From a user-supplied external system.
    External,

    /// No motion information available for this frame.
    None,
}
