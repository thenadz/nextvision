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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_is_not_degraded() {
        let v = ContextValidity::Valid;
        assert!(!matches!(v, ContextValidity::Degraded { .. }));
        assert!(!matches!(v, ContextValidity::Invalid));
    }

    #[test]
    fn degraded_carries_reason() {
        let v = ContextValidity::Degraded {
            reason: DegradationReason::PtzMoving,
        };
        match v {
            ContextValidity::Degraded { reason } => {
                assert_eq!(reason, DegradationReason::PtzMoving);
            }
            _ => panic!("expected Degraded"),
        }
    }

    #[test]
    fn degradation_reasons_are_distinct() {
        assert_ne!(DegradationReason::PtzMoving, DegradationReason::LargeJump);
        assert_ne!(DegradationReason::ZoomChange, DegradationReason::Unknown);
    }

    #[test]
    fn invalid_is_terminal() {
        let v = ContextValidity::Invalid;
        assert!(matches!(v, ContextValidity::Invalid));
    }
}
