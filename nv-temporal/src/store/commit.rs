use std::sync::Arc;

use nv_core::{MonotonicTs, TrackId};
use nv_perception::Track;
use nv_view::ViewEpoch;

use crate::continuity::SegmentBoundary;
use crate::trajectory::{Trajectory, TrajectoryPoint};

use super::{TemporalStore, TrackHistory};

impl TemporalStore {
    /// Commit a track update from the latest frame.
    ///
    /// This is the canonical way to upsert track history after stage
    /// execution. For each track produced by the pipeline:
    ///
    /// 1. If the track is new and the store is at the hard cap, the
    ///    store pre-evicts a victim (Lost → Coasted → Tentative,
    ///    oldest first) before inserting. If no evictable candidate
    ///    exists (only Confirmed tracks remain), the new track is
    ///    rejected and `false` is returned. Updates to existing
    ///    tracks are always accepted regardless of cap.
    /// 2. The observation is pushed into the ring buffer.
    /// 3. If the view epoch has changed since the last point, the active
    ///    trajectory segment is closed with [`SegmentBoundary::EpochChange`]
    ///    and a new segment is opened.
    /// 4. A trajectory point is appended from the observation's bbox center.
    /// 5. If the track state is `Lost`, the active segment is closed with
    ///    [`SegmentBoundary::TrackLost`].
    /// 6. Per-track observation and trajectory-point caps are enforced.
    ///
    /// The store size never exceeds `max_concurrent_tracks` after a
    /// successful commit. Call [`enforce_retention`](Self::enforce_retention)
    /// once per frame for age-based and trajectory-point pruning.
    ///
    /// Returns `true` if the track was committed, `false` if the track
    /// was a new admission rejected due to the hard cap.
    pub fn commit_track(
        &mut self,
        track: &Track,
        now_ts: MonotonicTs,
        current_epoch: ViewEpoch,
    ) -> bool {
        let track_id = track.id;
        let track_arc = Arc::new(track.clone());
        let max_obs = self.retention.max_observations_per_track;
        let max_traj = self.retention.max_trajectory_points_per_track;

        if self.get_track(&track_id).is_none() {
            // New track — enforce strict cap via pre-eviction.
            if self.tracks.len() >= self.retention.max_concurrent_tracks {
                match self.find_eviction_victim() {
                    Some(id) => {
                        self.tracks.remove(&id);
                    }
                    None => {
                        // Only Confirmed tracks remain — reject.
                        return false;
                    }
                }
            }

            let mut trajectory = Trajectory::new();
            trajectory.open_segment(current_epoch, SegmentBoundary::TrackCreated, None);
            let history = TrackHistory::new(
                Arc::clone(&track_arc),
                Arc::new(trajectory),
                now_ts,
                now_ts,
                current_epoch,
            );
            self.insert_track(track_id, history);
        }

        if let Some(history) = self.get_track_mut(&track_id) {
            history.track = Arc::clone(&track_arc);
            history.last_seen = now_ts;

            // Push observation.
            history.push_observation(track.current.clone());

            // Enforce per-track observation cap.
            while history.observation_count() > max_obs {
                history.pop_observation_front();
            }

            let traj = Arc::make_mut(&mut history.trajectory);

            // Epoch drift detection: if the active segment belongs to a
            // different epoch, close it and open a new segment.
            let needs_epoch_segment = traj
                .active_segment()
                .is_some_and(|seg| seg.view_epoch != current_epoch);

            if needs_epoch_segment {
                let old_epoch = traj.active_segment().unwrap().view_epoch;
                let boundary = SegmentBoundary::EpochChange {
                    from_epoch: old_epoch,
                    to_epoch: current_epoch,
                };
                traj.open_segment(current_epoch, boundary.clone(), Some(boundary));
            }

            // Append trajectory point from observation bbox center.
            let center = track.current.bbox.center();
            let point = TrajectoryPoint {
                ts: now_ts,
                position: nv_core::Point2::new(center.x, center.y),
                bbox: track.current.bbox,
            };
            traj.push_point(point);

            // Close the active segment when the track transitions to Lost.
            if track.state == nv_perception::TrackState::Lost {
                traj.close_active_segment(SegmentBoundary::TrackLost);
            }

            // Enforce per-track trajectory point cap.
            traj.prune_oldest_points(max_traj);
        }

        true
    }

    /// Enforce the retention policy: evict stale tracks, enforce the
    /// concurrent-track hard cap, and prune trajectory points.
    ///
    /// Should be called once per frame, after all track commits.
    ///
    /// Phase 1: evict `Lost` tracks older than
    /// [`RetentionPolicy::max_track_age`].
    ///
    /// Phase 2: if the track count exceeds
    /// [`RetentionPolicy::max_concurrent_tracks`], evict tracks by
    /// priority until under the limit:
    ///   1. `Lost` (oldest first)
    ///   2. `Coasted` (oldest first)
    ///   3. `Tentative` (oldest first)
    ///
    /// `Confirmed` tracks are never evicted by the hard cap.
    ///
    /// Note: in normal production flow the cap is already enforced by
    /// [`commit_track`]'s pre-eviction, so Phase 2 is a
    /// safety-net that fires only when tracks are inserted via test
    /// helpers or other non-`commit_track` paths.
    ///
    /// Phase 3: prune trajectory points for all surviving tracks to
    /// stay within [`RetentionPolicy::max_trajectory_points_per_track`].
    pub fn enforce_retention(&mut self, now_ts: MonotonicTs) {
        let max_age = self.retention.max_track_age;
        let max_tracks = self.retention.max_concurrent_tracks;
        let max_traj = self.retention.max_trajectory_points_per_track;

        // Phase 1: age-based eviction of Lost tracks.
        let mut to_evict: Vec<TrackId> = Vec::new();
        for (id, history) in self.tracks.iter() {
            if history.track.state == nv_perception::TrackState::Lost {
                if let Some(age) = now_ts.checked_duration_since(history.last_seen) {
                    if age >= max_age {
                        to_evict.push(*id);
                    }
                }
            }
        }
        for id in &to_evict {
            self.tracks.remove(id);
        }

        // Phase 2: hard cap on concurrent tracks.
        // Eviction priority: Lost (oldest) → Coasted (oldest) → Tentative (oldest).
        // Confirmed tracks are never evicted.
        while self.tracks.len() > max_tracks {
            match self.find_eviction_victim() {
                Some(id) => {
                    self.tracks.remove(&id);
                }
                None => break, // only Confirmed remain — tolerate excess
            }
        }

        // Phase 3: prune trajectory points per track.
        for history in self.tracks.values_mut() {
            Arc::make_mut(&mut history.trajectory).prune_oldest_points(max_traj);
        }
    }

    /// Apply a compensation transform to the active trajectory segments
    /// of all tracks in the given epoch.
    ///
    /// Each active segment's points are transformed in place and the
    /// segment's cumulative compensation transform is updated. Motion
    /// features are recomputed from the transformed points.
    ///
    /// Called by the executor when `EpochDecision::Compensate` is issued.
    pub fn apply_compensation(&mut self, transform: &nv_core::AffineTransform2D, epoch: ViewEpoch) {
        for history in self.tracks.values_mut() {
            let traj = Arc::make_mut(&mut history.trajectory);
            if let Some(seg) = traj.active_segment_mut() {
                if seg.view_epoch != epoch {
                    continue;
                }
                seg.apply_compensation(transform);
            }
        }
    }

    /// End a track normally (e.g., object left the scene).
    ///
    /// Closes the active trajectory segment with
    /// [`SegmentBoundary::TrackEnded`] and removes the track from the
    /// store. Returns the removed [`TrackHistory`] if the track existed.
    ///
    /// Called by the pipeline executor when a track that was previously
    /// in the temporal store is absent from an authoritative track set.
    /// Tracks explicitly reported as `Lost` by the tracker are handled
    /// by [`commit_track`](Self::commit_track) with
    /// [`SegmentBoundary::TrackLost`] instead.
    pub fn end_track(&mut self, id: &TrackId) -> Option<TrackHistory> {
        if let Some(history) = self.tracks.get_mut(id) {
            Arc::make_mut(&mut history.trajectory)
                .close_active_segment(SegmentBoundary::TrackEnded);
        }
        self.tracks.remove(id)
    }

    /// Close all active trajectory segments with the given boundary reason.
    ///
    /// Called before clearing state (e.g., on feed restart) to ensure all
    /// trajectory segments are properly bounded before disposal.
    pub fn close_all_segments(&mut self, boundary: SegmentBoundary) {
        for history in self.tracks.values_mut() {
            Arc::make_mut(&mut history.trajectory).close_active_segment(boundary.clone());
        }
    }
}
