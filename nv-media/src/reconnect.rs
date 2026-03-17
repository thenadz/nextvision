//! Reconnection tracking and backoff computation.
//!
//! [`ReconnectTracker`] is a self-contained FSM helper that tracks
//! reconnection attempts, computes backoff delays, and enforces the
//! configured attempt budget. Used by [`MediaSource`](crate::source::MediaSource).

use std::time::{Duration, Instant};

use nv_core::config::{BackoffKind, ReconnectPolicy};

/// Tracks reconnection attempts and computes backoff delays.
#[derive(Debug)]
pub(crate) struct ReconnectTracker {
    policy: ReconnectPolicy,
    /// Current attempt counter (reset after a successful reconnection).
    attempt: u32,
    /// Timestamp of the last reconnection attempt.
    last_attempt_at: Option<Instant>,
    /// Lifetime reconnection count (never reset).
    total_reconnects: u32,
}

impl ReconnectTracker {
    pub(crate) fn new(policy: ReconnectPolicy) -> Self {
        Self {
            policy,
            attempt: 0,
            last_attempt_at: None,
            total_reconnects: 0,
        }
    }

    /// Compute the backoff delay for the **current** attempt.
    pub(crate) fn backoff_delay(&self) -> Duration {
        let base = self.policy.base_delay;
        let delay = match self.policy.backoff {
            BackoffKind::Exponential => {
                // 2^attempt, saturating to avoid overflow
                let multiplier = 1u32.checked_shl(self.attempt).unwrap_or(u32::MAX);
                base.saturating_mul(multiplier)
            }
            BackoffKind::Linear => base.saturating_mul(self.attempt + 1),
        };
        delay.min(self.policy.max_delay)
    }

    /// Record a reconnection attempt.
    pub(crate) fn record_attempt(&mut self) {
        self.attempt += 1;
        self.total_reconnects += 1;
        self.last_attempt_at = Some(Instant::now());
    }

    /// Reset the attempt counter (called after a successful connection).
    pub(crate) fn reset_attempts(&mut self) {
        self.attempt = 0;
        self.last_attempt_at = None;
    }

    /// Whether the policy allows another reconnection attempt.
    pub(crate) fn can_retry(&self) -> bool {
        self.policy.max_attempts == 0 || self.attempt < self.policy.max_attempts
    }

    /// Current attempt number.
    pub(crate) fn current_attempt(&self) -> u32 {
        self.attempt
    }
}

#[cfg(test)]
impl ReconnectTracker {
    pub(crate) fn total_reconnects(&self) -> u32 {
        self.total_reconnects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limited_reconnect(max: u32) -> ReconnectPolicy {
        ReconnectPolicy {
            max_attempts: max,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff: BackoffKind::Exponential,
        }
    }

    #[test]
    fn exponential_backoff_delay() {
        let policy = ReconnectPolicy {
            max_attempts: 5,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff: BackoffKind::Exponential,
        };
        let mut t = ReconnectTracker::new(policy);
        assert_eq!(t.backoff_delay(), Duration::from_secs(1));
        t.record_attempt();
        assert_eq!(t.backoff_delay(), Duration::from_secs(2));
        t.record_attempt();
        assert_eq!(t.backoff_delay(), Duration::from_secs(4));
        t.record_attempt();
        assert_eq!(t.backoff_delay(), Duration::from_secs(8));
    }

    #[test]
    fn exponential_backoff_capped() {
        let policy = ReconnectPolicy {
            max_attempts: 0,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            backoff: BackoffKind::Exponential,
        };
        let mut t = ReconnectTracker::new(policy);
        for _ in 0..10 {
            t.record_attempt();
        }
        assert!(t.backoff_delay() <= Duration::from_secs(5));
    }

    #[test]
    fn linear_backoff_delay() {
        let policy = ReconnectPolicy {
            max_attempts: 5,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(30),
            backoff: BackoffKind::Linear,
        };
        let mut t = ReconnectTracker::new(policy);
        assert_eq!(t.backoff_delay(), Duration::from_secs(2));
        t.record_attempt();
        assert_eq!(t.backoff_delay(), Duration::from_secs(4));
        t.record_attempt();
        assert_eq!(t.backoff_delay(), Duration::from_secs(6));
    }

    #[test]
    fn reconnect_budget_exhausted() {
        let policy = limited_reconnect(2);
        let mut t = ReconnectTracker::new(policy);
        assert!(t.can_retry());
        t.record_attempt();
        assert!(t.can_retry());
        t.record_attempt();
        assert!(!t.can_retry());
    }

    #[test]
    fn reset_after_success() {
        let policy = limited_reconnect(3);
        let mut t = ReconnectTracker::new(policy);
        t.record_attempt();
        t.record_attempt();
        t.reset_attempts();
        assert!(t.can_retry());
        assert_eq!(t.current_attempt(), 0);
        assert_eq!(t.total_reconnects(), 2);
    }
}
