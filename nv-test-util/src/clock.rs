//! Controllable test clock for deterministic timestamp generation.

use nv_core::MonotonicTs;

/// A test clock that produces monotonic timestamps under manual control.
///
/// Useful for tests that need deterministic, reproducible timing.
///
/// # Example
///
/// ```
/// use nv_test_util::clock::TestClock;
///
/// let mut clock = TestClock::new();
/// assert_eq!(clock.now().as_nanos(), 0);
///
/// clock.advance_ms(33); // advance ~1 frame at 30fps
/// assert_eq!(clock.now().as_nanos(), 33_000_000);
/// ```
pub struct TestClock {
    current_ns: u64,
}

impl TestClock {
    /// Create a new test clock starting at zero.
    #[must_use]
    pub fn new() -> Self {
        Self { current_ns: 0 }
    }

    /// Create a test clock starting at the given nanosecond value.
    #[must_use]
    pub fn starting_at(nanos: u64) -> Self {
        Self { current_ns: nanos }
    }

    /// Current timestamp.
    #[must_use]
    pub fn now(&self) -> MonotonicTs {
        MonotonicTs::from_nanos(self.current_ns)
    }

    /// Advance the clock by the given number of nanoseconds.
    pub fn advance_nanos(&mut self, nanos: u64) {
        self.current_ns += nanos;
    }

    /// Advance the clock by the given number of milliseconds.
    pub fn advance_ms(&mut self, ms: u64) {
        self.current_ns += ms * 1_000_000;
    }

    /// Advance the clock by the given number of seconds.
    pub fn advance_secs(&mut self, secs: u64) {
        self.current_ns += secs * 1_000_000_000;
    }

    /// Tick the clock by one frame interval at the given FPS.
    ///
    /// Returns the new timestamp after advancing.
    pub fn tick_at_fps(&mut self, fps: f64) -> MonotonicTs {
        let interval_ns = (1_000_000_000.0 / fps) as u64;
        self.current_ns += interval_ns;
        self.now()
    }
}

impl Default for TestClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_zero() {
        let clock = TestClock::new();
        assert_eq!(clock.now().as_nanos(), 0);
    }

    #[test]
    fn advance_and_tick() {
        let mut clock = TestClock::new();
        clock.advance_ms(100);
        assert_eq!(clock.now().as_nanos(), 100_000_000);

        let ts = clock.tick_at_fps(30.0);
        assert!(ts.as_nanos() > 100_000_000);
    }
}
