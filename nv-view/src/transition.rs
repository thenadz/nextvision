//! Transition phase state machine for camera motion events.

/// Where in a camera motion transition the current frame sits.
///
/// This is a strict state machine updated every frame by the view system.
///
/// ## State transitions
///
/// | Previous | Current motion | Next |
/// |---|---|---|
/// | `Settled` | `Stable` | `Settled` |
/// | `Settled` | `Moving`/`Unknown` | `MoveStart` |
/// | `MoveStart` | `Moving`/`Unknown` | `Moving` |
/// | `MoveStart` | `Stable` | `MoveEnd` |
/// | `Moving` | `Moving`/`Unknown` | `Moving` |
/// | `Moving` | `Stable` | `MoveEnd` |
/// | `MoveEnd` | `Stable` | `Settled` |
/// | `MoveEnd` | `Moving`/`Unknown` | `MoveStart` |
///
/// `MoveStart` and `MoveEnd` each last exactly one frame (edge-triggered).
/// `Settled` and `Moving` persist across multiple frames (level-triggered).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransitionPhase {
    /// No transition in progress. Camera is (or has been) stable.
    Settled,
    /// First frame of detected motion.
    MoveStart,
    /// Camera is mid-move.
    Moving,
    /// First frame of detected stability after motion.
    MoveEnd,
}

impl TransitionPhase {
    /// Compute the next phase given the current motion state.
    ///
    /// `is_moving` should be `true` if the camera motion state is
    /// `Moving` or `Unknown` (conservative: unknown = potentially moving).
    #[must_use]
    pub fn next(self, is_moving: bool) -> Self {
        match (self, is_moving) {
            (Self::Settled, false) => Self::Settled,
            (Self::Settled, true) => Self::MoveStart,
            (Self::MoveStart, true) => Self::Moving,
            (Self::MoveStart, false) => Self::MoveEnd,
            (Self::Moving, true) => Self::Moving,
            (Self::Moving, false) => Self::MoveEnd,
            (Self::MoveEnd, false) => Self::Settled,
            (Self::MoveEnd, true) => Self::MoveStart,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settled_stays_settled() {
        assert_eq!(
            TransitionPhase::Settled.next(false),
            TransitionPhase::Settled
        );
    }

    #[test]
    fn settled_to_move_start() {
        assert_eq!(
            TransitionPhase::Settled.next(true),
            TransitionPhase::MoveStart
        );
    }

    #[test]
    fn move_start_to_moving() {
        assert_eq!(
            TransitionPhase::MoveStart.next(true),
            TransitionPhase::Moving
        );
    }

    #[test]
    fn move_start_one_frame_jitter() {
        assert_eq!(
            TransitionPhase::MoveStart.next(false),
            TransitionPhase::MoveEnd
        );
    }

    #[test]
    fn moving_to_move_end() {
        assert_eq!(
            TransitionPhase::Moving.next(false),
            TransitionPhase::MoveEnd
        );
    }

    #[test]
    fn move_end_to_settled() {
        assert_eq!(
            TransitionPhase::MoveEnd.next(false),
            TransitionPhase::Settled
        );
    }

    #[test]
    fn move_end_resume_motion() {
        assert_eq!(
            TransitionPhase::MoveEnd.next(true),
            TransitionPhase::MoveStart
        );
    }

    #[test]
    fn full_cycle() {
        let mut phase = TransitionPhase::Settled;
        phase = phase.next(true); // MoveStart
        assert_eq!(phase, TransitionPhase::MoveStart);
        phase = phase.next(true); // Moving
        assert_eq!(phase, TransitionPhase::Moving);
        phase = phase.next(true); // Moving
        assert_eq!(phase, TransitionPhase::Moving);
        phase = phase.next(false); // MoveEnd
        assert_eq!(phase, TransitionPhase::MoveEnd);
        phase = phase.next(false); // Settled
        assert_eq!(phase, TransitionPhase::Settled);
    }
}
