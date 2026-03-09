//! Trajectory types: paths, segments, points, and motion features.

use nv_core::{AffineTransform2D, BBox, MonotonicTs, Point2};
use nv_view::ViewEpoch;

use crate::continuity::SegmentBoundary;

/// A tracked object's spatial path over time.
///
/// Composed of segments, each covering a contiguous run of observations
/// within a single [`ViewEpoch`].
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    /// Ordered segments. The last segment is the active (open) segment.
    pub segments: Vec<TrajectorySegment>,
}

impl Trajectory {
    /// Create an empty trajectory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Returns the currently active (last) segment, if any.
    #[must_use]
    pub fn active_segment(&self) -> Option<&TrajectorySegment> {
        self.segments.last().filter(|s| s.closed_by.is_none())
    }

    /// Returns the total number of points across all segments.
    #[must_use]
    pub fn total_points(&self) -> usize {
        self.segments.iter().map(|s| s.points.len()).sum()
    }
}

/// A contiguous run of trajectory observations within a single [`ViewEpoch`].
///
/// Segments are opened and closed by the temporal store in response to
/// epoch changes, track creation, and track loss. Each boundary is
/// recorded as a [`SegmentBoundary`] for auditability.
#[derive(Clone, Debug)]
pub struct TrajectorySegment {
    /// The view epoch for this segment.
    pub view_epoch: ViewEpoch,
    /// Ordered trajectory points within the segment.
    pub points: Vec<TrajectoryPoint>,
    /// Computed motion features for this segment.
    pub motion: MotionFeatures,
    /// Why this segment was opened.
    pub opened_by: SegmentBoundary,
    /// Why this segment was closed. `None` if still active.
    pub closed_by: Option<SegmentBoundary>,
    /// Cumulative compensation transform applied within this segment.
    ///
    /// `None` if no compensation has been applied.
    pub compensation: Option<AffineTransform2D>,
    /// Number of times compensation was applied within this segment.
    pub compensation_count: u32,
}

impl TrajectorySegment {
    /// Whether this segment is still active (not closed).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.closed_by.is_none()
    }

    /// Duration of this segment (first point to last point).
    #[must_use]
    pub fn duration(&self) -> Option<nv_core::Duration> {
        if self.points.len() < 2 {
            return None;
        }
        let first = self.points.first()?.ts;
        let last = self.points.last()?.ts;
        last.checked_duration_since(first)
    }
}

/// A single point in a trajectory.
#[derive(Clone, Debug)]
pub struct TrajectoryPoint {
    /// Timestamp of this point.
    pub ts: MonotonicTs,
    /// Centroid position (center of bbox) in normalized coordinates.
    pub position: Point2,
    /// Full bounding box at this point.
    pub bbox: BBox,
}

/// Computed motion features for a trajectory segment.
///
/// All spatial values are in normalized `[0, 1]` coordinates.
/// Speed values are in normalized-coordinates per second.
#[derive(Clone, Debug)]
pub struct MotionFeatures {
    /// Total path length along the trajectory (sum of point-to-point distances).
    pub displacement: f32,
    /// Straight-line distance from first to last point.
    pub net_displacement: f32,
    /// Mean speed: `displacement / elapsed_time`.
    pub mean_speed: f32,
    /// Maximum instantaneous speed observed.
    pub max_speed: f32,
    /// Dominant direction in radians. `None` if stationary.
    pub direction: Option<f32>,
    /// Whether the object is stationary (below a configurable speed threshold).
    pub is_stationary: bool,
}

impl Default for MotionFeatures {
    fn default() -> Self {
        Self {
            displacement: 0.0,
            net_displacement: 0.0,
            mean_speed: 0.0,
            max_speed: 0.0,
            direction: None,
            is_stationary: true,
        }
    }
}

impl MotionFeatures {
    /// Recompute motion features from a sequence of trajectory points.
    ///
    /// Returns default (stationary) features if fewer than 2 points.
    #[must_use]
    pub fn compute(points: &[TrajectoryPoint]) -> Self {
        if points.len() < 2 {
            return Self::default();
        }

        let mut displacement = 0.0f32;
        let mut max_speed = 0.0f32;

        for window in points.windows(2) {
            let d = window[0].position.distance_to(&window[1].position);
            displacement += d;

            let dt = window[1]
                .ts
                .saturating_duration_since(window[0].ts)
                .as_secs_f64() as f32;
            if dt > 0.0 {
                let speed = d / dt;
                max_speed = max_speed.max(speed);
            }
        }

        let first = &points[0];
        let last = &points[points.len() - 1];
        let net_displacement = first.position.distance_to(&last.position);
        let elapsed = last
            .ts
            .saturating_duration_since(first.ts)
            .as_secs_f64() as f32;
        let mean_speed = if elapsed > 0.0 {
            displacement / elapsed
        } else {
            0.0
        };

        let direction = if net_displacement > 1e-6 {
            let dx = last.position.x - first.position.x;
            let dy = last.position.y - first.position.y;
            Some(dy.atan2(dx))
        } else {
            None
        };

        // Stationary threshold: ~0.5% of frame per second
        let is_stationary = mean_speed < 0.005;

        Self {
            displacement,
            net_displacement,
            mean_speed,
            max_speed,
            direction,
            is_stationary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_features_stationary() {
        let f = MotionFeatures::compute(&[]);
        assert!(f.is_stationary);
        assert_eq!(f.displacement, 0.0);
    }

    #[test]
    fn motion_features_moving() {
        let points = vec![
            TrajectoryPoint {
                ts: MonotonicTs::from_nanos(0),
                position: Point2::new(0.0, 0.0),
                bbox: BBox::new(0.0, 0.0, 0.1, 0.1),
            },
            TrajectoryPoint {
                ts: MonotonicTs::from_nanos(1_000_000_000),
                position: Point2::new(0.5, 0.0),
                bbox: BBox::new(0.45, 0.0, 0.55, 0.1),
            },
        ];
        let f = MotionFeatures::compute(&points);
        assert!(!f.is_stationary);
        assert!((f.displacement - 0.5).abs() < 1e-5);
        assert!((f.net_displacement - 0.5).abs() < 1e-5);
        assert!((f.mean_speed - 0.5).abs() < 1e-5);
    }
}
