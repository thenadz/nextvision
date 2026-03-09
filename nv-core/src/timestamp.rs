//! Timestamp types for the NextVision runtime.
//!
//! Two timestamp types serve distinct purposes:
//!
//! - [`MonotonicTs`] — monotonic nanoseconds since feed start. Used for all
//!   internal ordering and duration calculations. Never wall-clock.
//! - [`WallTs`] — wall-clock microseconds since Unix epoch. Used only in
//!   output and provenance for correlation with external systems.

use std::fmt;
use std::ops;
use std::time;

/// Monotonic timestamp — nanoseconds since feed start.
///
/// Used for all internal ordering, duration calculations, and temporal logic.
/// Never derived from wall-clock time. Compare freely for ordering.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicTs(u64);

impl MonotonicTs {
    /// The zero timestamp (feed start).
    pub const ZERO: Self = Self(0);

    /// Create a monotonic timestamp from nanoseconds since feed start.
    #[must_use]
    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Returns the timestamp as nanoseconds.
    #[must_use]
    pub fn as_nanos(self) -> u64 {
        self.0
    }

    /// Returns the timestamp as fractional seconds.
    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    /// Compute the duration between two timestamps.
    ///
    /// Returns `None` if `other` is later than `self`.
    #[must_use]
    pub fn checked_duration_since(self, other: Self) -> Option<Duration> {
        self.0.checked_sub(other.0).map(Duration::from_nanos)
    }

    /// Saturating subtraction — returns zero duration if `other > self`.
    #[must_use]
    pub fn saturating_duration_since(self, other: Self) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(other.0))
    }
}

impl fmt::Display for MonotonicTs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ns", self.0)
    }
}

impl fmt::Debug for MonotonicTs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MonotonicTs({}ns)", self.0)
    }
}

impl ops::Add<Duration> for MonotonicTs {
    type Output = Self;
    fn add(self, rhs: Duration) -> Self {
        Self(self.0 + rhs.as_nanos())
    }
}

/// Wall-clock timestamp — microseconds since Unix epoch.
///
/// Used only in output and provenance for external correlation.
/// **Never use for ordering or duration calculations within the pipeline.**
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct WallTs(i64);

impl WallTs {
    /// Create a wall-clock timestamp from microseconds since Unix epoch.
    #[must_use]
    pub fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Capture the current wall-clock time.
    #[must_use]
    pub fn now() -> Self {
        let d = time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap_or_default();
        Self(d.as_micros() as i64)
    }

    /// Returns the timestamp as microseconds since Unix epoch.
    #[must_use]
    pub fn as_micros(self) -> i64 {
        self.0
    }
}

impl fmt::Display for WallTs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}μs", self.0)
    }
}

impl fmt::Debug for WallTs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WallTs({}μs)", self.0)
    }
}

/// A duration in nanoseconds.
///
/// Library-defined duration type for consistency across the crate boundary.
/// Convertible to/from [`std::time::Duration`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(u64);

impl Duration {
    /// Zero duration.
    pub const ZERO: Self = Self(0);

    /// Create a duration from nanoseconds.
    #[must_use]
    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Create a duration from microseconds.
    #[must_use]
    pub fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }

    /// Create a duration from milliseconds.
    #[must_use]
    pub fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }

    /// Create a duration from seconds.
    #[must_use]
    pub fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }

    /// Returns the duration in nanoseconds.
    #[must_use]
    pub fn as_nanos(self) -> u64 {
        self.0
    }

    /// Returns the duration as fractional seconds.
    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1_000_000_000 {
            write!(f, "{:.3}s", self.as_secs_f64())
        } else if self.0 >= 1_000_000 {
            write!(f, "{:.3}ms", self.0 as f64 / 1_000_000.0)
        } else if self.0 >= 1_000 {
            write!(f, "{:.1}μs", self.0 as f64 / 1_000.0)
        } else {
            write!(f, "{}ns", self.0)
        }
    }
}

impl fmt::Debug for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Duration({}ns)", self.0)
    }
}

impl From<std::time::Duration> for Duration {
    fn from(d: std::time::Duration) -> Self {
        Self(d.as_nanos() as u64)
    }
}

impl From<Duration> for std::time::Duration {
    fn from(d: Duration) -> Self {
        std::time::Duration::from_nanos(d.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_ordering() {
        let a = MonotonicTs::from_nanos(100);
        let b = MonotonicTs::from_nanos(200);
        assert!(a < b);
    }

    #[test]
    fn duration_conversion() {
        let d = Duration::from_millis(42);
        let std_d: std::time::Duration = d.into();
        assert_eq!(std_d.as_millis(), 42);
    }

    #[test]
    fn checked_duration_since() {
        let a = MonotonicTs::from_nanos(300);
        let b = MonotonicTs::from_nanos(100);
        let d = a.checked_duration_since(b).unwrap();
        assert_eq!(d.as_nanos(), 200);
        assert!(b.checked_duration_since(a).is_none());
    }
}
