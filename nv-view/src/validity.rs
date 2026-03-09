//! Context validity and degradation reasons.

/// Whether temporal state is valid under the current camera view.
///
/// Stages and output consumers use this to decide how much to trust
/// spatial relationships derived from temporal state.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextValidity {
    /// View is stable; all temporal state within this epoch is valid.
    Valid,

    /// View is changing; temporal state is degraded.
    ///
    /// Tracks and trajectories from earlier in this epoch may be unreliable
    /// for direct spatial comparison with current-frame positions.
    Degraded { reason: DegradationReason },

    /// View has changed so much that prior context is invalid.
    ///
    /// A new epoch has been (or should be) opened.
    Invalid,
}

/// Why temporal context is degraded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DegradationReason {
    /// Camera is performing a PTZ move.
    PtzMoving,
    /// A large sudden jump was detected.
    LargeJump,
    /// The zoom level changed significantly.
    ZoomChange,
    /// Occlusion or blur degraded the frame.
    OcclusionOrBlur,
    /// Motion was inferred from video with low confidence.
    InferredMotionLowConfidence,
    /// Reason is unspecified or could not be determined.
    Unknown,
}
