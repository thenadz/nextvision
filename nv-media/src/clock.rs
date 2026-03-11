//! PTS tracking and stream discontinuity detection.
//!
//! [`PtsTracker`] monitors presentation timestamps across consecutive frames
//! and classifies each new PTS relative to its predecessor.
//!
//! A *discontinuity* occurs when the gap between consecutive frames exceeds
//! a configurable threshold. Common causes:
//!
//! - Server-side stream restart
//! - Network interruption causing frame loss
//! - Server-side seek or recording loop
//! - PTS clock reset
//!
//! The default threshold is 5 seconds, which accommodates typical RTSP jitter
//! and low-FPS sources while catching genuine disruptions.

/// Default discontinuity threshold: 5 seconds in nanoseconds.
const DEFAULT_DISCONTINUITY_THRESHOLD_NS: u64 = 5_000_000_000;

/// Tracks presentation timestamps across frames and detects discontinuities.
pub(crate) struct PtsTracker {
    /// Last observed PTS (nanoseconds), or `None` before the first frame.
    last_pts_ns: Option<u64>,
    /// Total frames observed (not reset on `reset()`).
    frame_count: u64,
    /// Maximum acceptable gap between consecutive frames (ns).
    discontinuity_threshold_ns: u64,
}

/// Result of observing a new PTS value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PtsResult {
    /// First frame — no predecessor to compare against.
    First,
    /// Normal progression — PTS advanced within expected bounds.
    Normal {
        /// Absolute delta from the previous PTS (nanoseconds).
        delta_ns: u64,
    },
    /// Discontinuity detected — gap exceeds the threshold.
    Discontinuity {
        /// Absolute gap size in nanoseconds.
        gap_ns: u64,
        /// Previous PTS (nanoseconds).
        prev_ns: u64,
        /// Current PTS (nanoseconds).
        current_ns: u64,
    },
}

impl PtsTracker {
    /// Create a PTS tracker with the default discontinuity threshold (5 s).
    pub fn new() -> Self {
        Self {
            last_pts_ns: None,
            frame_count: 0,
            discontinuity_threshold_ns: DEFAULT_DISCONTINUITY_THRESHOLD_NS,
        }
    }

    /// Process a new presentation timestamp and classify the result.
    pub fn observe(&mut self, pts_ns: u64) -> PtsResult {
        self.frame_count += 1;

        let result = match self.last_pts_ns {
            None => PtsResult::First,
            Some(prev) => {
                // Absolute gap — handles both forward jumps and backward
                // jumps (clock resets, server-side seeks).
                let gap = pts_ns.abs_diff(prev);

                if gap > self.discontinuity_threshold_ns {
                    PtsResult::Discontinuity {
                        gap_ns: gap,
                        prev_ns: prev,
                        current_ns: pts_ns,
                    }
                } else {
                    PtsResult::Normal { delta_ns: gap }
                }
            }
        };

        self.last_pts_ns = Some(pts_ns);
        result
    }

}

#[cfg(test)]
impl PtsTracker {
    pub fn with_threshold_ns(threshold_ns: u64) -> Self {
        Self {
            last_pts_ns: None,
            frame_count: 0,
            discontinuity_threshold_ns: threshold_ns,
        }
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn reset(&mut self) {
        self.last_pts_ns = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_frame_is_first() {
        let mut t = PtsTracker::new();
        assert_eq!(t.observe(0), PtsResult::First);
        assert_eq!(t.frame_count(), 1);
    }

    #[test]
    fn normal_progression() {
        let mut t = PtsTracker::new();
        t.observe(0);
        let r = t.observe(33_333_333); // ~30 fps
        assert_eq!(
            r,
            PtsResult::Normal {
                delta_ns: 33_333_333
            }
        );
    }

    #[test]
    fn forward_discontinuity() {
        let mut t = PtsTracker::with_threshold_ns(1_000_000_000); // 1 s
        t.observe(0);
        let r = t.observe(2_000_000_000); // 2 s jump
        assert!(matches!(
            r,
            PtsResult::Discontinuity {
                gap_ns: 2_000_000_000,
                ..
            }
        ));
    }

    #[test]
    fn backward_discontinuity() {
        let mut t = PtsTracker::with_threshold_ns(1_000_000_000);
        t.observe(5_000_000_000);
        let r = t.observe(1_000_000_000); // jumped back 4 s
        assert!(matches!(
            r,
            PtsResult::Discontinuity {
                gap_ns: 4_000_000_000,
                ..
            }
        ));
    }

    #[test]
    fn reset_makes_next_frame_first() {
        let mut t = PtsTracker::new();
        t.observe(100);
        t.reset();
        assert_eq!(t.observe(200), PtsResult::First);
        // frame_count survives reset
        assert_eq!(t.frame_count(), 2);
    }

    #[test]
    fn small_jitter_is_normal() {
        let mut t = PtsTracker::with_threshold_ns(1_000_000_000);
        t.observe(1_000_000_000);
        // 500 ms jitter — under threshold
        let r = t.observe(1_500_000_000);
        assert!(matches!(r, PtsResult::Normal { .. }));
    }

    #[test]
    fn threshold_boundary_is_normal() {
        let mut t = PtsTracker::with_threshold_ns(1_000_000_000);
        t.observe(0);
        // Exactly at threshold — should be Normal (not Discontinuity)
        let r = t.observe(1_000_000_000);
        assert!(matches!(r, PtsResult::Normal { .. }));
    }
}
