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

    /// Returns a mutable reference to the currently active (last) segment, if any.
    pub fn active_segment_mut(&mut self) -> Option<&mut TrajectorySegment> {
        self.segments.last_mut().filter(|s| s.closed_by.is_none())
    }

    /// Returns the total number of points across all segments.
    #[must_use]
    pub fn total_points(&self) -> usize {
        self.segments.iter().map(|s| s.points.len()).sum()
    }

    /// Number of segments (both open and closed).
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Close the currently active segment with the given boundary reason.
    ///
    /// Returns `true` if an active segment was closed, `false` if there
    /// was no active segment.
    pub fn close_active_segment(&mut self, reason: SegmentBoundary) -> bool {
        if let Some(seg) = self.segments.last_mut().filter(|s| s.is_active()) {
            seg.closed_by = Some(reason);
            true
        } else {
            false
        }
    }

    /// Open a new segment. If there is a currently active segment, it is
    /// closed with `close_reason` first.
    ///
    /// The new segment starts with an empty point list and default motion
    /// features.
    pub fn open_segment(
        &mut self,
        epoch: ViewEpoch,
        opened_by: SegmentBoundary,
        close_reason: Option<SegmentBoundary>,
    ) {
        if let Some(reason) = close_reason {
            self.close_active_segment(reason);
        }
        self.segments.push(TrajectorySegment {
            view_epoch: epoch,
            points: Vec::new(),
            motion: MotionFeatures::default(),
            opened_by,
            closed_by: None,
            compensation: None,
            compensation_count: 0,
        });
    }

    /// Push a point to the active segment and recompute its motion features.
    ///
    /// Returns `true` if the point was added, `false` if there is no active
    /// segment.
    pub fn push_point(&mut self, point: TrajectoryPoint) -> bool {
        if let Some(seg) = self.active_segment_mut() {
            seg.points.push(point);
            seg.motion = MotionFeatures::compute(&seg.points);
            true
        } else {
            false
        }
    }

    /// Prune the oldest points to bring total points at or below `max_points`.
    ///
    /// Removes points from the oldest closed segments first. Empty closed
    /// segments are discarded entirely. The active segment is pruned from
    /// the front as a last resort.
    ///
    /// Uses a two-pass approach: first counts how many complete closed
    /// segments can be removed, then drains them in a single `Vec::drain`
    /// call, avoiding repeated O(n) front-removals.
    pub fn prune_oldest_points(&mut self, max_points: usize) {
        let total = self.total_points();
        if total <= max_points {
            return;
        }
        let mut to_remove = total - max_points;

        // Pass 1: count complete closed segments that can be removed entirely.
        let mut segments_to_drain = 0;
        for seg in &self.segments {
            if to_remove == 0 {
                break;
            }
            if seg.is_active() {
                // Active segment is always last — handled below.
                break;
            }
            if seg.points.len() <= to_remove {
                to_remove -= seg.points.len();
                segments_to_drain += 1;
            } else {
                break;
            }
        }

        // Remove complete closed segments in one operation — O(n) total.
        if segments_to_drain > 0 {
            self.segments.drain(..segments_to_drain);
        }

        // Pass 2: handle remaining deficit from the first surviving segment.
        if to_remove > 0 {
            if let Some(seg) = self.segments.first_mut() {
                if seg.is_active() && seg.points.len() <= to_remove {
                    seg.points.clear();
                    seg.motion = MotionFeatures::default();
                } else {
                    let drain_count = to_remove.min(seg.points.len());
                    seg.points.drain(..drain_count);
                    seg.motion = MotionFeatures::compute(&seg.points);
                }
            }
        }
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

    /// Apply a compensation transform to all points in this segment.
    ///
    /// Each point's `position` and `bbox` are transformed in place.
    /// The segment's motion features are recomputed from the updated
    /// positions, and the cumulative compensation transform is updated.
    pub fn apply_compensation(&mut self, transform: &AffineTransform2D) {
        for point in &mut self.points {
            point.position = transform.apply(point.position);
            point.bbox = transform.apply_bbox(point.bbox);
        }
        self.motion = MotionFeatures::compute(&self.points);
        self.compensation = Some(match self.compensation.take() {
            Some(existing) => existing.then(transform),
            None => *transform,
        });
        self.compensation_count += 1;
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
        let elapsed = last.ts.saturating_duration_since(first.ts).as_secs_f64() as f32;
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

    fn make_point(ts_ns: u64, x: f32, y: f32) -> TrajectoryPoint {
        TrajectoryPoint {
            ts: MonotonicTs::from_nanos(ts_ns),
            position: Point2::new(x, y),
            bbox: BBox::new(x - 0.05, y - 0.05, x + 0.05, y + 0.05),
        }
    }

    // ------------------------------------------------------------------
    // Trajectory open/close/push tests
    // ------------------------------------------------------------------

    #[test]
    fn open_segment_creates_empty_segment() {
        let mut traj = Trajectory::new();
        traj.open_segment(
            ViewEpoch::INITIAL,
            SegmentBoundary::TrackCreated,
            None,
        );
        assert_eq!(traj.segment_count(), 1);
        assert!(traj.active_segment().is_some());
        assert_eq!(traj.total_points(), 0);
    }

    #[test]
    fn push_point_adds_to_active_segment() {
        let mut traj = Trajectory::new();
        traj.open_segment(
            ViewEpoch::INITIAL,
            SegmentBoundary::TrackCreated,
            None,
        );

        assert!(traj.push_point(make_point(0, 0.1, 0.1)));
        assert!(traj.push_point(make_point(33_333_333, 0.2, 0.1)));
        assert_eq!(traj.total_points(), 2);

        let seg = traj.active_segment().unwrap();
        assert!(!seg.motion.is_stationary, "two distant points should show motion");
    }

    #[test]
    fn push_point_returns_false_without_active_segment() {
        let mut traj = Trajectory::new();
        assert!(!traj.push_point(make_point(0, 0.1, 0.1)));
    }

    #[test]
    fn close_active_segment_marks_boundary() {
        let mut traj = Trajectory::new();
        traj.open_segment(
            ViewEpoch::INITIAL,
            SegmentBoundary::TrackCreated,
            None,
        );
        traj.push_point(make_point(0, 0.1, 0.1));

        assert!(traj.close_active_segment(SegmentBoundary::TrackLost));
        assert!(traj.active_segment().is_none());
        assert_eq!(traj.segments[0].closed_by, Some(SegmentBoundary::TrackLost));
    }

    #[test]
    fn close_returns_false_when_no_active_segment() {
        let mut traj = Trajectory::new();
        assert!(!traj.close_active_segment(SegmentBoundary::TrackLost));
    }

    // ------------------------------------------------------------------
    // Trajectory segmentation (epoch-based)
    // ------------------------------------------------------------------

    #[test]
    fn open_new_segment_closes_previous() {
        let epoch_0 = ViewEpoch::INITIAL;
        let epoch_1 = epoch_0.next();

        let mut traj = Trajectory::new();
        traj.open_segment(epoch_0, SegmentBoundary::TrackCreated, None);
        traj.push_point(make_point(0, 0.1, 0.1));
        traj.push_point(make_point(33_333_333, 0.2, 0.1));

        // Open a new segment due to epoch change, closing the previous one.
        traj.open_segment(
            epoch_1,
            SegmentBoundary::EpochChange {
                from_epoch: epoch_0,
                to_epoch: epoch_1,
            },
            Some(SegmentBoundary::EpochChange {
                from_epoch: epoch_0,
                to_epoch: epoch_1,
            }),
        );

        assert_eq!(traj.segment_count(), 2);

        // First segment should be closed.
        assert!(!traj.segments[0].is_active());
        assert_eq!(traj.segments[0].view_epoch, epoch_0);
        assert_eq!(traj.segments[0].points.len(), 2);

        // Second segment should be active and empty.
        assert!(traj.segments[1].is_active());
        assert_eq!(traj.segments[1].view_epoch, epoch_1);
        assert_eq!(traj.segments[1].points.len(), 0);
    }

    #[test]
    fn multi_segment_trajectory_total_points() {
        let epoch_0 = ViewEpoch::INITIAL;
        let epoch_1 = epoch_0.next();

        let mut traj = Trajectory::new();
        traj.open_segment(epoch_0, SegmentBoundary::TrackCreated, None);
        traj.push_point(make_point(0, 0.1, 0.1));
        traj.push_point(make_point(33_333_333, 0.2, 0.1));
        traj.push_point(make_point(66_666_666, 0.3, 0.1));

        traj.open_segment(
            epoch_1,
            SegmentBoundary::EpochChange {
                from_epoch: epoch_0,
                to_epoch: epoch_1,
            },
            Some(SegmentBoundary::EpochChange {
                from_epoch: epoch_0,
                to_epoch: epoch_1,
            }),
        );
        traj.push_point(make_point(100_000_000, 0.4, 0.1));
        traj.push_point(make_point(133_333_333, 0.5, 0.1));

        assert_eq!(traj.segment_count(), 2);
        assert_eq!(traj.total_points(), 5);
        assert_eq!(traj.segments[0].points.len(), 3);
        assert_eq!(traj.segments[1].points.len(), 2);
    }

    #[test]
    fn segment_duration() {
        let mut seg = TrajectorySegment {
            view_epoch: ViewEpoch::INITIAL,
            points: Vec::new(),
            motion: MotionFeatures::default(),
            opened_by: SegmentBoundary::TrackCreated,
            closed_by: None,
            compensation: None,
            compensation_count: 0,
        };

        // No points → no duration.
        assert!(seg.duration().is_none());

        // One point → no duration.
        seg.points.push(make_point(1_000_000_000, 0.1, 0.1));
        assert!(seg.duration().is_none());

        // Two points → computable duration.
        seg.points.push(make_point(3_000_000_000, 0.2, 0.2));
        let dur = seg.duration().unwrap();
        assert_eq!(dur.as_nanos(), 2_000_000_000);
    }

    // ------------------------------------------------------------------
    // Motion features edge cases
    // ------------------------------------------------------------------

    #[test]
    fn motion_features_single_point() {
        let f = MotionFeatures::compute(&[make_point(0, 0.5, 0.5)]);
        assert!(f.is_stationary);
        assert_eq!(f.displacement, 0.0);
    }

    #[test]
    fn motion_features_direction() {
        let points = vec![
            make_point(0, 0.0, 0.0),
            make_point(1_000_000_000, 0.0, 0.5),
        ];
        let f = MotionFeatures::compute(&points);
        // Moving purely in +Y → direction should be ~π/2.
        let dir = f.direction.unwrap();
        assert!(
            (dir - std::f32::consts::FRAC_PI_2).abs() < 0.01,
            "direction should be approx π/2 for +Y motion, got {dir}"
        );
    }

    #[test]
    fn motion_features_max_speed() {
        // Slow → fast → slow.
        let points = vec![
            make_point(0, 0.0, 0.0),
            make_point(1_000_000_000, 0.01, 0.0),  // slow: 0.01/s
            make_point(2_000_000_000, 0.51, 0.0),  // fast: 0.50/s
            make_point(3_000_000_000, 0.52, 0.0),  // slow: 0.01/s
        ];
        let f = MotionFeatures::compute(&points);
        assert!(f.max_speed > 0.4, "max speed should reflect the fast segment");
    }

    // ------------------------------------------------------------------
    // Trajectory point pruning tests
    // ------------------------------------------------------------------

    #[test]
    fn prune_oldest_points_no_op_when_under_limit() {
        let mut traj = Trajectory::new();
        traj.open_segment(
            ViewEpoch::INITIAL,
            SegmentBoundary::TrackCreated,
            None,
        );
        for i in 0..5u64 {
            traj.push_point(make_point(i * 1_000_000, 0.0, 0.0));
        }
        traj.prune_oldest_points(10);
        assert_eq!(traj.total_points(), 5);
    }

    #[test]
    fn prune_oldest_points_removes_from_single_segment() {
        let mut traj = Trajectory::new();
        traj.open_segment(
            ViewEpoch::INITIAL,
            SegmentBoundary::TrackCreated,
            None,
        );
        for i in 0..10u64 {
            traj.push_point(make_point(i * 1_000_000, 0.0, 0.0));
        }
        traj.prune_oldest_points(3);
        assert_eq!(traj.total_points(), 3);
        // Remaining points should be the most recent.
        let seg = traj.active_segment().unwrap();
        assert_eq!(seg.points[0].ts, MonotonicTs::from_nanos(7_000_000));
    }

    #[test]
    fn prune_oldest_points_removes_closed_segments_first() {
        let epoch0 = ViewEpoch::INITIAL;
        let epoch1 = epoch0.next();
        let epoch2 = epoch1.next();

        let mut traj = Trajectory::new();
        // Segment 0: 3 points, closed.
        traj.open_segment(epoch0, SegmentBoundary::TrackCreated, None);
        for i in 0..3u64 {
            traj.push_point(make_point(i * 1_000_000, 0.0, 0.0));
        }
        // Segment 1: 3 points, closed.
        traj.open_segment(
            epoch1,
            SegmentBoundary::EpochChange { from_epoch: epoch0, to_epoch: epoch1 },
            Some(SegmentBoundary::EpochChange { from_epoch: epoch0, to_epoch: epoch1 }),
        );
        for i in 3..6u64 {
            traj.push_point(make_point(i * 1_000_000, 0.0, 0.0));
        }
        // Segment 2: 3 points, active.
        traj.open_segment(
            epoch2,
            SegmentBoundary::EpochChange { from_epoch: epoch1, to_epoch: epoch2 },
            Some(SegmentBoundary::EpochChange { from_epoch: epoch1, to_epoch: epoch2 }),
        );
        for i in 6..9u64 {
            traj.push_point(make_point(i * 1_000_000, 0.0, 0.0));
        }

        assert_eq!(traj.total_points(), 9);
        assert_eq!(traj.segment_count(), 3);

        // Prune to 4 points: should remove segment 0 entirely (3 pts),
        // then remove 2 pts from segment 1 (3 → 1).
        traj.prune_oldest_points(4);
        assert_eq!(traj.total_points(), 4);
        // Segment 0 should be gone entirely.
        assert_eq!(traj.segment_count(), 2);
        // First remaining segment (formerly seg1) should have 1 point.
        assert_eq!(traj.segments[0].points.len(), 1);
        // Active segment intact.
        assert_eq!(traj.segments[1].points.len(), 3);
    }

    #[test]
    fn prune_to_zero_keeps_active_segment_shell() {
        let mut traj = Trajectory::new();
        traj.open_segment(
            ViewEpoch::INITIAL,
            SegmentBoundary::TrackCreated,
            None,
        );
        for i in 0..5u64 {
            traj.push_point(make_point(i * 1_000_000, 0.0, 0.0));
        }
        traj.prune_oldest_points(0);
        // Active segment is kept (but empty).
        assert_eq!(traj.segment_count(), 1);
        assert_eq!(traj.total_points(), 0);
        assert!(traj.active_segment().is_some());
    }

    // ------------------------------------------------------------------
    // Segment compensation tests
    // ------------------------------------------------------------------

    #[test]
    fn segment_apply_compensation_transforms_points() {
        let mut seg = TrajectorySegment {
            view_epoch: ViewEpoch::INITIAL,
            points: vec![
                make_point(0, 0.1, 0.2),
                make_point(1_000_000, 0.3, 0.4),
            ],
            motion: MotionFeatures::default(),
            opened_by: SegmentBoundary::TrackCreated,
            closed_by: None,
            compensation: None,
            compensation_count: 0,
        };

        let t = AffineTransform2D::new(1.0, 0.0, 0.5, 0.0, 1.0, -0.1);
        seg.apply_compensation(&t);

        assert!((seg.points[0].position.x - 0.6).abs() < 1e-5);
        assert!((seg.points[0].position.y - 0.1).abs() < 1e-5);
        assert!((seg.points[1].position.x - 0.8).abs() < 1e-5);
        assert!((seg.points[1].position.y - 0.3).abs() < 1e-5);
        assert_eq!(seg.compensation_count, 1);
        assert!(seg.compensation.is_some());
    }

    #[test]
    fn segment_apply_compensation_composes_transforms() {
        let mut seg = TrajectorySegment {
            view_epoch: ViewEpoch::INITIAL,
            points: vec![make_point(0, 0.0, 0.0)],
            motion: MotionFeatures::default(),
            opened_by: SegmentBoundary::TrackCreated,
            closed_by: None,
            compensation: None,
            compensation_count: 0,
        };

        let t1 = AffineTransform2D::new(1.0, 0.0, 0.1, 0.0, 1.0, 0.0);
        let t2 = AffineTransform2D::new(1.0, 0.0, 0.2, 0.0, 1.0, 0.0);
        seg.apply_compensation(&t1);
        seg.apply_compensation(&t2);

        assert_eq!(seg.compensation_count, 2);
        // Point should have moved by 0.1 + 0.2 = 0.3.
        assert!((seg.points[0].position.x - 0.3).abs() < 1e-5);
        // Cumulative transform tx should be ~0.3.
        let comp = seg.compensation.unwrap();
        assert!((comp.m[2] - 0.3).abs() < 1e-9);
    }
}
