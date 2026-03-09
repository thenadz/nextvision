//! Strongly-typed identifiers for feeds, tracks, detections, and stages.
//!
//! All IDs are lightweight, `Copy`, and suitable for use as hash-map keys.

use std::fmt;

/// Unique identifier for a video feed within the runtime.
///
/// Backed by a `u64`. Two feeds in the same runtime always have distinct IDs.
/// IDs are never reused within a single runtime session.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeedId(u64);

impl FeedId {
    /// Create a new `FeedId` from a raw value.
    ///
    /// Intended for internal use by the runtime. External callers should
    /// receive `FeedId` values from [`FeedHandle`](crate) rather than constructing them.
    #[must_use]
    pub fn new(val: u64) -> Self {
        Self(val)
    }

    /// Returns the underlying `u64` value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for FeedId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "feed-{}", self.0)
    }
}

impl fmt::Debug for FeedId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeedId({})", self.0)
    }
}

/// Unique identifier for a tracked object across frames.
///
/// Backed by a `u64`. Track IDs are unique within a single feed session.
/// After a feed restart, track IDs may be reused (the temporal store is cleared).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(u64);

impl TrackId {
    /// Create a new `TrackId`.
    #[must_use]
    pub fn new(val: u64) -> Self {
        Self(val)
    }

    /// Returns the underlying `u64` value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "track-{}", self.0)
    }
}

impl fmt::Debug for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TrackId({})", self.0)
    }
}

/// Unique identifier for a single detection within a frame.
///
/// Backed by a `u64`. Detection IDs are unique within a single frame's
/// [`DetectionSet`](crate).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetectionId(u64);

impl DetectionId {
    /// Create a new `DetectionId`.
    #[must_use]
    pub fn new(val: u64) -> Self {
        Self(val)
    }

    /// Returns the underlying `u64` value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for DetectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "det-{}", self.0)
    }
}

impl fmt::Debug for DetectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DetectionId({})", self.0)
    }
}

/// Identifier for a perception stage.
///
/// A compile-time `&'static str` name, not a runtime-generated value.
/// Stage names are fixed in the stage implementation (e.g., `"yolov8_detector"`).
///
/// `StageId` is `Copy` and zero-allocation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StageId(pub &'static str);

impl StageId {
    /// Returns the stage name as a string slice.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl fmt::Debug for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StageId(\"{}\")", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_id_display() {
        let id = FeedId::new(42);
        assert_eq!(id.to_string(), "feed-42");
    }

    #[test]
    fn stage_id_copy() {
        let a = StageId("detector");
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn ids_are_copy_and_eq() {
        let t = TrackId::new(1);
        let d = DetectionId::new(2);
        assert_eq!(t, TrackId::new(1));
        assert_eq!(d, DetectionId::new(2));
    }
}
