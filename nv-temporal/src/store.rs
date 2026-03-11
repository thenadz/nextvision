//! The [`TemporalStore`] — per-feed temporal state manager.
//!
//! # Hot-path performance
//!
//! [`TemporalStore::snapshot()`] is called once per frame and must be fast.
//! All heavy fields inside [`TrackHistory`] are `Arc`-wrapped:
//!
//! - **`Track`** and **`Trajectory`** — shared via `Arc::clone` (pointer bump).
//! - **Observations** — wrapped in `Arc<[TrackObservation]>` and rebuilt only
//!   when new observations arrive (tracked via a generation counter). When
//!   unchanged, the snapshot shares the same allocation as the previous one.
//!
//! This makes `snapshot()` **O(n)** in the number of *active* tracks (those
//! that received new observations since the last snapshot) rather than
//! **O(n × k)** where *k* is the observation window depth.
//!
//! # Observation mutation
//!
//! The observation buffer inside [`TrackHistory`] is private. All mutations
//! go through [`push_observation()`](TrackHistory::push_observation),
//! [`pop_observation_front()`](TrackHistory::pop_observation_front), and
//! [`clear_observations()`](TrackHistory::clear_observations), which
//! automatically bump the generation counter. This prevents stale-cache
//! bugs that would arise if callers mutated the buffer directly and forgot
//! to invalidate the snapshot cache.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use nv_core::{MonotonicTs, TrackId};
use nv_perception::{Track, TrackObservation};
use nv_view::ViewEpoch;

use crate::continuity::SegmentBoundary;
use crate::retention::RetentionPolicy;
use crate::trajectory::{Trajectory, TrajectoryPoint};

/// Per-feed temporal state manager.
///
/// Owns all track histories, trajectories, and motion features for a
/// single feed. Not shared between feeds.
///
/// The store is mutated only by the pipeline executor between frames
/// (after all stages have run), never during stage execution.
pub struct TemporalStore {
    tracks: HashMap<TrackId, TrackHistory>,
    retention: RetentionPolicy,
    view_epoch: ViewEpoch,
}

impl TemporalStore {
    /// Create a new temporal store with the given retention policy.
    #[must_use]
    pub fn new(retention: RetentionPolicy) -> Self {
        Self {
            tracks: HashMap::new(),
            retention,
            view_epoch: ViewEpoch::INITIAL,
        }
    }

    /// Current view epoch.
    #[must_use]
    pub fn view_epoch(&self) -> ViewEpoch {
        self.view_epoch
    }

    /// Set the view epoch (called by the runtime on epoch change).
    pub fn set_view_epoch(&mut self, epoch: ViewEpoch) {
        self.view_epoch = epoch;
    }

    /// Number of active tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get a track history by ID.
    #[must_use]
    pub fn get_track(&self, id: &TrackId) -> Option<&TrackHistory> {
        self.tracks.get(id)
    }

    /// Get a mutable reference to a track history by ID.
    pub fn get_track_mut(&mut self, id: &TrackId) -> Option<&mut TrackHistory> {
        self.tracks.get_mut(id)
    }

    /// Insert or replace a track history.
    pub fn insert_track(&mut self, id: TrackId, history: TrackHistory) {
        self.tracks.insert(id, history);
    }

    /// Remove a track history by ID, returning it if present.
    pub fn remove_track(&mut self, id: &TrackId) -> Option<TrackHistory> {
        self.tracks.remove(id)
    }

    /// Iterate over all track histories.
    pub fn tracks(&self) -> impl Iterator<Item = (&TrackId, &TrackHistory)> {
        self.tracks.iter()
    }

    /// Take an immutable snapshot for stage consumption.
    ///
    /// The snapshot is `Arc`-wrapped and cheap to clone. Track and trajectory
    /// data is shared via `Arc`; observations are only re-materialized when
    /// the track's observation buffer has changed since the previous snapshot.
    ///
    /// # Complexity
    ///
    /// - **Unchanged tracks**: O(1) per track (Arc bumps + cached obs reuse).
    /// - **Changed tracks**: O(k) per track where *k* is the observation window.
    /// - **Total**: O(n) best-case, O(n × k) worst-case (all tracks mutated).
    #[must_use]
    pub fn snapshot(&mut self) -> TemporalStoreSnapshot {
        TemporalStoreSnapshot {
            inner: Arc::new(SnapshotInner {
                tracks: self
                    .tracks
                    .iter_mut()
                    .map(|(k, v)| (*k, SnapshotTrackHistory::from_history(v)))
                    .collect(),
                view_epoch: self.view_epoch,
            }),
        }
    }

    /// Clear all state (called on feed restart).
    pub fn clear(&mut self) {
        self.tracks.clear();
    }

    /// Access the retention policy.
    #[must_use]
    pub fn retention(&self) -> &RetentionPolicy {
        &self.retention
    }

    /// Iterate over all track IDs.
    pub fn track_ids(&self) -> impl Iterator<Item = &TrackId> {
        self.tracks.keys()
    }

    /// Find the best eviction victim by priority.
    ///
    /// Priority: Lost (oldest) → Coasted (oldest) → Tentative (oldest).
    /// Confirmed tracks are never selected. Returns `None` if only
    /// Confirmed tracks remain.
    fn find_eviction_victim(&self) -> Option<TrackId> {
        self.tracks
            .iter()
            .filter(|(_, h)| h.track.state == nv_perception::TrackState::Lost)
            .min_by_key(|(_, h)| h.last_seen)
            .map(|(id, _)| *id)
            .or_else(|| {
                self.tracks
                    .iter()
                    .filter(|(_, h)| h.track.state == nv_perception::TrackState::Coasted)
                    .min_by_key(|(_, h)| h.last_seen)
                    .map(|(id, _)| *id)
            })
            .or_else(|| {
                self.tracks
                    .iter()
                    .filter(|(_, h)| h.track.state == nv_perception::TrackState::Tentative)
                    .min_by_key(|(_, h)| h.last_seen)
                    .map(|(id, _)| *id)
            })
    }

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
    pub fn commit_track(&mut self, track: &Track, now_ts: MonotonicTs, current_epoch: ViewEpoch) -> bool {
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
            trajectory.open_segment(
                current_epoch,
                SegmentBoundary::TrackCreated,
                None,
            );
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
                traj.open_segment(
                    current_epoch,
                    boundary.clone(),
                    Some(boundary),
                );
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

/// Complete history of a single track within the temporal store.
#[derive(Clone, Debug)]
pub struct TrackHistory {
    /// Current track state (`Arc`-wrapped so snapshots share the allocation).
    pub track: Arc<Track>,
    /// Ring buffer of recent observations (private — mutate through methods
    /// so the generation counter stays consistent with the snapshot cache).
    observations: VecDeque<TrackObservation>,
    /// Spatial trajectory segmented by view epoch (`Arc`-wrapped for cheap snapshots).
    pub trajectory: Arc<Trajectory>,
    /// Timestamp when this track was first seen.
    pub first_seen: MonotonicTs,
    /// Timestamp of the most recent observation.
    pub last_seen: MonotonicTs,
    /// View epoch at the time this track was created.
    pub view_epoch_at_creation: ViewEpoch,
    /// Generation counter — incremented each time `observations` is mutated.
    /// Used by `snapshot()` to skip re-materializing the observation slice
    /// when nothing changed.
    observation_gen: u64,
    /// Cached `Arc<[TrackObservation]>` from the last snapshot. Reused when
    /// `observation_gen` has not advanced.
    cached_observations: Option<(u64, Arc<[TrackObservation]>)>,
}

impl TrackHistory {
    /// Create a new track history with no observations.
    #[must_use]
    pub fn new(
        track: Arc<Track>,
        trajectory: Arc<Trajectory>,
        first_seen: MonotonicTs,
        last_seen: MonotonicTs,
        view_epoch_at_creation: ViewEpoch,
    ) -> Self {
        Self {
            track,
            observations: VecDeque::new(),
            trajectory,
            first_seen,
            last_seen,
            view_epoch_at_creation,
            observation_gen: 0,
            cached_observations: None,
        }
    }

    /// Push an observation to the back of the ring buffer.
    ///
    /// Automatically invalidates the snapshot cache.
    pub fn push_observation(&mut self, obs: TrackObservation) {
        self.observations.push_back(obs);
        self.observation_gen = self.observation_gen.wrapping_add(1);
    }

    /// Remove and return the oldest observation, if any.
    ///
    /// Automatically invalidates the snapshot cache.
    pub fn pop_observation_front(&mut self) -> Option<TrackObservation> {
        let item = self.observations.pop_front();
        if item.is_some() {
            self.observation_gen = self.observation_gen.wrapping_add(1);
        }
        item
    }

    /// Number of observations currently buffered.
    #[must_use]
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Read-only access to the observation ring buffer.
    #[must_use]
    pub fn observations(&self) -> &VecDeque<TrackObservation> {
        &self.observations
    }

    /// Clear all observations.
    ///
    /// Automatically invalidates the snapshot cache.
    pub fn clear_observations(&mut self) {
        if !self.observations.is_empty() {
            self.observations.clear();
            self.observation_gen = self.observation_gen.wrapping_add(1);
        }
    }

    /// Materialize or reuse the observation snapshot slice.
    fn snapshot_observations(&mut self) -> Arc<[TrackObservation]> {
        if let Some((cached_gen, ref cached)) = self.cached_observations {
            if cached_gen == self.observation_gen {
                return Arc::clone(cached);
            }
        }
        let slice: Arc<[TrackObservation]> = self.observations.iter().cloned().collect();
        self.cached_observations = Some((self.observation_gen, Arc::clone(&slice)));
        slice
    }
}

/// Internal snapshot data.
#[derive(Clone, Debug)]
struct SnapshotInner {
    tracks: HashMap<TrackId, SnapshotTrackHistory>,
    view_epoch: ViewEpoch,
}

/// Track history inside a snapshot — shares `Arc`-wrapped data with the
/// original [`TrackHistory`].
///
/// Observations are stored as `Arc<[TrackObservation]>` so that successive
/// snapshots of the same (unchanged) track share the same allocation.
#[derive(Clone, Debug)]
pub struct SnapshotTrackHistory {
    /// Current track state (shared with the live store via `Arc`).
    pub track: Arc<Track>,
    /// Recent observations, oldest-first. Shared with the live store's cache
    /// when unchanged between snapshots.
    pub observations: Arc<[TrackObservation]>,
    /// Spatial trajectory segmented by view epoch (shared via `Arc`).
    pub trajectory: Arc<Trajectory>,
    /// Timestamp when this track was first seen.
    pub first_seen: MonotonicTs,
    /// Timestamp of the most recent observation.
    pub last_seen: MonotonicTs,
    /// View epoch at the time this track was created.
    pub view_epoch_at_creation: ViewEpoch,
}

impl SnapshotTrackHistory {
    fn from_history(h: &mut TrackHistory) -> Self {
        Self {
            track: Arc::clone(&h.track),
            observations: h.snapshot_observations(),
            trajectory: Arc::clone(&h.trajectory),
            first_seen: h.first_seen,
            last_seen: h.last_seen,
            view_epoch_at_creation: h.view_epoch_at_creation,
        }
    }
}

/// Read-only, cheaply-cloneable snapshot of the temporal store.
///
/// Captured once per frame before stage execution. All stages for that
/// frame see the same snapshot. `Clone` is an `Arc` bump.
///
/// Implements [`TemporalAccess`](nv_perception::TemporalAccess) for use
/// through `StageContext::temporal`.
#[derive(Clone, Debug)]
pub struct TemporalStoreSnapshot {
    inner: Arc<SnapshotInner>,
}

impl TemporalStoreSnapshot {
    /// Current view epoch in this snapshot.
    #[must_use]
    pub fn view_epoch(&self) -> ViewEpoch {
        self.inner.view_epoch
    }

    /// Number of tracks in this snapshot.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.inner.tracks.len()
    }

    /// Get a snapshot track history by ID.
    #[must_use]
    pub fn get_track(&self, id: &TrackId) -> Option<&SnapshotTrackHistory> {
        self.inner.tracks.get(id)
    }

    /// Iterate over all snapshot track histories.
    pub fn tracks(&self) -> impl Iterator<Item = (&TrackId, &SnapshotTrackHistory)> {
        self.inner.tracks.iter()
    }
}

// ---------------------------------------------------------------------------
// TrackStore / TrajectoryStoreAccess — focused sub-interfaces
// ---------------------------------------------------------------------------

/// Focused interface for track lifecycle queries.
///
/// Provides quick track-level checks without requiring the full
/// [`TemporalAccess`](nv_perception::TemporalAccess) surface. Methods
/// on this trait are intentionally non-overlapping with `TemporalAccess`
/// to avoid ambiguity when both traits are in scope.
///
/// Implemented by [`TemporalStore`] and [`TemporalStoreSnapshot`].
pub trait TrackStore {
    /// Whether a track with the given ID exists.
    fn has_track(&self, id: &TrackId) -> bool;

    /// Get the current lifecycle state of a track, if present.
    ///
    /// Cheaper than `get_track()` when only the state enum is needed.
    fn track_state(&self, id: &TrackId) -> Option<nv_perception::TrackState>;

    /// View epoch at which a track was created.
    fn track_creation_epoch(&self, id: &TrackId) -> Option<ViewEpoch>;
}

/// Focused interface for trajectory queries.
///
/// Provides direct access to [`Trajectory`] objects (with segments,
/// points, and motion features) — richer than the count-only accessors
/// available through [`TemporalAccess`](nv_perception::TemporalAccess).
///
/// Implemented by [`TemporalStore`] and [`TemporalStoreSnapshot`].
pub trait TrajectoryStoreAccess {
    /// Get the full trajectory for a track, if present.
    ///
    /// Returns the trajectory with all segments, points, motion features,
    /// and boundary reasons — unlike `TemporalAccess` which exposes only
    /// point and segment counts.
    fn trajectory(&self, id: &TrackId) -> Option<&Trajectory>;
}

impl TrackStore for TemporalStore {
    fn has_track(&self, id: &TrackId) -> bool {
        self.tracks.contains_key(id)
    }

    fn track_state(&self, id: &TrackId) -> Option<nv_perception::TrackState> {
        self.tracks.get(id).map(|h| h.track.state)
    }

    fn track_creation_epoch(&self, id: &TrackId) -> Option<ViewEpoch> {
        self.tracks.get(id).map(|h| h.view_epoch_at_creation)
    }
}

impl TrajectoryStoreAccess for TemporalStore {
    fn trajectory(&self, id: &TrackId) -> Option<&Trajectory> {
        self.tracks.get(id).map(|h| &*h.trajectory)
    }
}

impl TrackStore for TemporalStoreSnapshot {
    fn has_track(&self, id: &TrackId) -> bool {
        self.inner.tracks.contains_key(id)
    }

    fn track_state(&self, id: &TrackId) -> Option<nv_perception::TrackState> {
        self.inner.tracks.get(id).map(|h| h.track.state)
    }

    fn track_creation_epoch(&self, id: &TrackId) -> Option<ViewEpoch> {
        self.inner.tracks.get(id).map(|h| h.view_epoch_at_creation)
    }
}

impl TrajectoryStoreAccess for TemporalStoreSnapshot {
    fn trajectory(&self, id: &TrackId) -> Option<&Trajectory> {
        self.inner.tracks.get(id).map(|h| &*h.trajectory)
    }
}

/// Empty slice constant for returning from `TemporalAccess` when a track is absent.
static EMPTY_OBSERVATIONS: &[TrackObservation] = &[];

impl nv_perception::TemporalAccess for TemporalStoreSnapshot {
    fn view_epoch(&self) -> ViewEpoch {
        self.inner.view_epoch
    }

    fn track_count(&self) -> usize {
        self.inner.tracks.len()
    }

    fn track_ids(&self) -> Vec<TrackId> {
        self.inner.tracks.keys().copied().collect()
    }

    fn get_track(&self, id: &TrackId) -> Option<&Track> {
        self.inner.tracks.get(id).map(|h| h.track.as_ref())
    }

    fn recent_observations(&self, id: &TrackId) -> &[TrackObservation] {
        self.inner
            .tracks
            .get(id)
            .map(|h| &*h.observations)
            .unwrap_or(EMPTY_OBSERVATIONS)
    }

    fn first_seen(&self, id: &TrackId) -> Option<MonotonicTs> {
        self.inner.tracks.get(id).map(|h| h.first_seen)
    }

    fn last_seen(&self, id: &TrackId) -> Option<MonotonicTs> {
        self.inner.tracks.get(id).map(|h| h.last_seen)
    }

    fn trajectory_point_count(&self, id: &TrackId) -> usize {
        self.inner
            .tracks
            .get(id)
            .map(|h| h.trajectory.total_points())
            .unwrap_or(0)
    }

    fn trajectory_segment_count(&self, id: &TrackId) -> usize {
        self.inner
            .tracks
            .get(id)
            .map(|h| h.trajectory.segment_count())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::geom::BBox;
    use nv_perception::TrackState;

    fn make_observation(ts_ns: u64) -> TrackObservation {
        TrackObservation {
            ts: MonotonicTs::from_nanos(ts_ns),
            bbox: BBox::new(0.0, 0.0, 0.1, 0.1),
            confidence: 0.9,
            state: TrackState::Confirmed,
            detection_id: None,
        }
    }

    fn make_track(id: u64) -> Track {
        Track {
            id: TrackId::new(id),
            class_id: 0,
            state: TrackState::Confirmed,
            current: make_observation(0),
            metadata: nv_core::TypedMetadata::new(),
        }
    }

    /// Insert a track into the store for testing.
    fn insert_track(store: &mut TemporalStore, track_id: u64, obs_count: usize) {
        let id = TrackId::new(track_id);
        let mut history = TrackHistory::new(
            Arc::new(make_track(track_id)),
            Arc::new(Trajectory::new()),
            MonotonicTs::from_nanos(0),
            MonotonicTs::from_nanos((obs_count as u64).saturating_sub(1) * 1_000_000),
            ViewEpoch::INITIAL,
        );
        for i in 0..obs_count {
            history.push_observation(make_observation((i as u64) * 1_000_000));
        }
        store.tracks.insert(id, history);
    }

    /// Insert a track with a specific state directly (bypasses admission control).
    fn insert_track_with_state(
        store: &mut TemporalStore,
        track_id: u64,
        state: TrackState,
        last_seen_ns: u64,
    ) {
        let id = TrackId::new(track_id);
        let track = Arc::new(make_track_with_state(track_id, state));
        let history = TrackHistory::new(
            track,
            Arc::new(Trajectory::new()),
            MonotonicTs::from_nanos(0),
            MonotonicTs::from_nanos(last_seen_ns),
            ViewEpoch::INITIAL,
        );
        store.tracks.insert(id, history);
    }

    #[test]
    fn snapshot_shares_track_arc() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 1, 3);

        let snap = store.snapshot();
        let id = TrackId::new(1);
        let live = store.get_track(&id).unwrap();
        let snapped = snap.get_track(&id).unwrap();

        // Track Arc should be the same allocation.
        assert!(Arc::ptr_eq(&live.track, &snapped.track));
        // Trajectory Arc should be the same allocation.
        assert!(Arc::ptr_eq(&live.trajectory, &snapped.trajectory));
    }

    #[test]
    fn snapshot_reuses_observations_when_unchanged() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 1, 5);

        let snap1 = store.snapshot();
        let snap2 = store.snapshot();

        let id = TrackId::new(1);
        let obs1 = &snap1.get_track(&id).unwrap().observations;
        let obs2 = &snap2.get_track(&id).unwrap().observations;

        // The two snapshots should share the exact same observations Arc
        // since no mutations happened between them.
        assert!(
            Arc::ptr_eq(obs1, obs2),
            "unchanged observations should share Arc allocation"
        );
    }

    #[test]
    fn snapshot_rebuilds_observations_after_mutation() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 1, 3);

        let snap1 = store.snapshot();

        // Mutate observations through the encapsulated API — generation is
        // bumped automatically, no manual `mark_observations_dirty()` needed.
        let id = TrackId::new(1);
        let history = store.tracks.get_mut(&id).unwrap();
        history.push_observation(make_observation(99_000_000));

        let snap2 = store.snapshot();

        let obs1 = &snap1.get_track(&id).unwrap().observations;
        let obs2 = &snap2.get_track(&id).unwrap().observations;

        assert_ne!(
            obs1.len(),
            obs2.len(),
            "snap2 should have the new observation"
        );
        assert!(
            !Arc::ptr_eq(obs1, obs2),
            "changed observations should have new Arc"
        );
        assert_eq!(obs2.len(), 4);
    }

    #[test]
    fn snapshot_clone_is_arc_bump() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 1, 2);

        let snap = store.snapshot();
        let snap_clone = snap.clone();

        // Both should point to the same inner allocation.
        assert!(Arc::ptr_eq(&snap.inner, &snap_clone.inner));
    }

    #[test]
    fn empty_store_snapshot() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let snap = store.snapshot();
        assert_eq!(snap.track_count(), 0);
    }

    #[test]
    fn pop_front_invalidates_cache() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 1, 3);

        let snap1 = store.snapshot();
        let id = TrackId::new(1);

        let history = store.tracks.get_mut(&id).unwrap();
        let popped = history.pop_observation_front();
        assert!(popped.is_some());

        let snap2 = store.snapshot();
        let obs1 = &snap1.get_track(&id).unwrap().observations;
        let obs2 = &snap2.get_track(&id).unwrap().observations;
        assert!(!Arc::ptr_eq(obs1, obs2), "pop_front must invalidate cache");
        assert_eq!(obs2.len(), 2);
    }

    #[test]
    fn clear_observations_invalidates_cache() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 1, 3);

        let snap1 = store.snapshot();
        let id = TrackId::new(1);

        let history = store.tracks.get_mut(&id).unwrap();
        history.clear_observations();

        let snap2 = store.snapshot();
        let obs1 = &snap1.get_track(&id).unwrap().observations;
        let obs2 = &snap2.get_track(&id).unwrap().observations;
        assert!(!Arc::ptr_eq(obs1, obs2));
        assert_eq!(obs2.len(), 0);
    }

    #[test]
    fn clear_empty_observations_does_not_bump_gen() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 1, 0);

        let snap1 = store.snapshot();
        let id = TrackId::new(1);

        let history = store.tracks.get_mut(&id).unwrap();
        history.clear_observations(); // no-op on empty

        let snap2 = store.snapshot();
        let obs1 = &snap1.get_track(&id).unwrap().observations;
        let obs2 = &snap2.get_track(&id).unwrap().observations;
        // Already empty, so the gen should not have bumped → same Arc.
        assert!(Arc::ptr_eq(obs1, obs2));
    }

    #[test]
    fn observation_count_reflects_mutations() {
        let mut history = TrackHistory::new(
            Arc::new(make_track(1)),
            Arc::new(Trajectory::new()),
            MonotonicTs::from_nanos(0),
            MonotonicTs::from_nanos(0),
            ViewEpoch::INITIAL,
        );
        assert_eq!(history.observation_count(), 0);
        history.push_observation(make_observation(1_000_000));
        assert_eq!(history.observation_count(), 1);
        history.push_observation(make_observation(2_000_000));
        assert_eq!(history.observation_count(), 2);
        history.pop_observation_front();
        assert_eq!(history.observation_count(), 1);
        history.clear_observations();
        assert_eq!(history.observation_count(), 0);
    }

    #[test]
    fn remove_track_returns_history() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        insert_track(&mut store, 42, 3);
        assert_eq!(store.track_count(), 1);

        let removed = store.remove_track(&TrackId::new(42));
        assert!(removed.is_some());
        assert_eq!(store.track_count(), 0);

        // Removing again yields None.
        assert!(store.remove_track(&TrackId::new(42)).is_none());
    }

    // ------------------------------------------------------------------
    // Track lifecycle tests
    // ------------------------------------------------------------------

    fn make_track_with_state(id: u64, state: TrackState) -> Track {
        Track {
            id: TrackId::new(id),
            class_id: 0,
            state,
            current: make_observation(0),
            metadata: nv_core::TypedMetadata::new(),
        }
    }

    #[test]
    fn commit_track_creates_new_history() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let ts = MonotonicTs::from_nanos(1_000_000);
        let epoch = ViewEpoch::INITIAL;

        store.commit_track(&track, ts, epoch);

        assert_eq!(store.track_count(), 1);
        let history = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(history.first_seen, ts);
        assert_eq!(history.last_seen, ts);
        assert_eq!(history.observation_count(), 1);
        assert_eq!(history.trajectory.segment_count(), 1);
        assert_eq!(history.trajectory.total_points(), 1);
    }

    #[test]
    fn commit_track_updates_existing_history() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);
        store.commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch);
        store.commit_track(&track, MonotonicTs::from_nanos(3_000_000), epoch);

        let history = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(history.observation_count(), 3);
        assert_eq!(history.trajectory.total_points(), 3);
        assert_eq!(history.last_seen, MonotonicTs::from_nanos(3_000_000));
        // first_seen should remain unchanged.
        assert_eq!(history.first_seen, MonotonicTs::from_nanos(1_000_000));
    }

    #[test]
    fn commit_track_enforces_observation_cap() {
        let retention = RetentionPolicy {
            max_observations_per_track: 3,
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        for i in 0..10u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        let history = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(
            history.observation_count(),
            3,
            "observations should be capped at max_observations_per_track"
        );
    }

    #[test]
    fn track_lifecycle_tentative_to_confirmed_to_lost() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        // Tentative track.
        let tentative = make_track_with_state(1, TrackState::Tentative);
        store.commit_track(&tentative, MonotonicTs::from_nanos(1_000_000), epoch);
        assert_eq!(
            store.get_track(&TrackId::new(1)).unwrap().track.state,
            TrackState::Tentative
        );

        // Confirmed.
        let confirmed = make_track_with_state(1, TrackState::Confirmed);
        store.commit_track(&confirmed, MonotonicTs::from_nanos(2_000_000), epoch);
        assert_eq!(
            store.get_track(&TrackId::new(1)).unwrap().track.state,
            TrackState::Confirmed
        );

        // Coasted.
        let coasted = make_track_with_state(1, TrackState::Coasted);
        store.commit_track(&coasted, MonotonicTs::from_nanos(3_000_000), epoch);
        assert_eq!(
            store.get_track(&TrackId::new(1)).unwrap().track.state,
            TrackState::Coasted
        );

        // Lost.
        let lost = make_track_with_state(1, TrackState::Lost);
        store.commit_track(&lost, MonotonicTs::from_nanos(4_000_000), epoch);
        assert_eq!(
            store.get_track(&TrackId::new(1)).unwrap().track.state,
            TrackState::Lost
        );
    }

    // ------------------------------------------------------------------
    // Retention / state-pruning tests
    // ------------------------------------------------------------------

    #[test]
    fn enforce_retention_evicts_old_lost_tracks() {
        let retention = RetentionPolicy {
            max_track_age: nv_core::Duration::from_secs(10),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let epoch = ViewEpoch::INITIAL;

        let lost = make_track_with_state(1, TrackState::Lost);
        store.commit_track(&lost, MonotonicTs::from_nanos(0), epoch);
        assert_eq!(store.track_count(), 1);

        // Time = 11 seconds → beyond max_track_age.
        store.enforce_retention(MonotonicTs::from_nanos(11_000_000_000));
        assert_eq!(store.track_count(), 0, "old Lost track should be evicted");
    }

    #[test]
    fn enforce_retention_keeps_recently_lost_tracks() {
        let retention = RetentionPolicy {
            max_track_age: nv_core::Duration::from_secs(10),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let epoch = ViewEpoch::INITIAL;

        let lost = make_track_with_state(1, TrackState::Lost);
        store.commit_track(&lost, MonotonicTs::from_nanos(5_000_000_000), epoch);

        // Time = 8 seconds → within max_track_age of last_seen (5s).
        store.enforce_retention(MonotonicTs::from_nanos(8_000_000_000));
        assert_eq!(store.track_count(), 1, "recent Lost track should be kept");
    }

    #[test]
    fn enforce_retention_does_not_evict_confirmed_tracks() {
        let retention = RetentionPolicy {
            max_track_age: nv_core::Duration::from_secs(1),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let epoch = ViewEpoch::INITIAL;

        let confirmed = make_track_with_state(1, TrackState::Confirmed);
        store.commit_track(&confirmed, MonotonicTs::from_nanos(0), epoch);

        // Even at max age, Confirmed tracks are not evicted.
        store.enforce_retention(MonotonicTs::from_nanos(10_000_000_000));
        assert_eq!(
            store.track_count(),
            1,
            "Confirmed tracks should never be age-evicted"
        );
    }

    #[test]
    fn enforce_retention_hard_cap_evicts_oldest_lost_first() {
        let retention = RetentionPolicy {
            max_concurrent_tracks: 3,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);

        // Insert directly to bypass pre-eviction — testing enforce_retention.
        for i in 1..=5u64 {
            insert_track_with_state(&mut store, i, TrackState::Lost, i * 1_000_000);
        }
        assert_eq!(store.track_count(), 5);

        store.enforce_retention(MonotonicTs::from_nanos(10_000_000));
        assert_eq!(store.track_count(), 3);

        // Oldest (track 1, 2) should be gone; newest (3, 4, 5) remain.
        assert!(store.get_track(&TrackId::new(1)).is_none());
        assert!(store.get_track(&TrackId::new(2)).is_none());
        assert!(store.get_track(&TrackId::new(3)).is_some());
    }

    #[test]
    fn enforce_retention_hard_cap_spares_confirmed_and_tentative() {
        let retention = RetentionPolicy {
            max_concurrent_tracks: 2,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);

        // Insert directly to bypass admission control — we're testing
        // enforce_retention eviction, not commit_track admission.
        // 3 Confirmed tracks + 1 Lost.
        for i in 1..=3u64 {
            insert_track_with_state(&mut store, i, TrackState::Confirmed, i * 1_000_000);
        }
        insert_track_with_state(&mut store, 10, TrackState::Lost, 0);
        assert_eq!(store.track_count(), 4);

        store.enforce_retention(MonotonicTs::from_nanos(100_000_000));

        // Lost track should be evicted, but Confirmed tracks stay even though > cap.
        assert!(store.get_track(&TrackId::new(10)).is_none());
        assert_eq!(store.track_count(), 3, "Confirmed tracks cannot be evicted by hard cap");
    }

    #[test]
    fn enforce_retention_hard_cap_evicts_tentative_after_lost_and_coasted() {
        let retention = RetentionPolicy {
            max_concurrent_tracks: 2,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);

        // Insert directly to bypass admission control.
        // 2 Confirmed + 2 Tentative — no Lost or Coasted.
        for i in 1..=2u64 {
            insert_track_with_state(&mut store, i, TrackState::Confirmed, i * 1_000_000);
        }
        for i in 3..=4u64 {
            insert_track_with_state(&mut store, i, TrackState::Tentative, i * 1_000_000);
        }
        assert_eq!(store.track_count(), 4);

        store.enforce_retention(MonotonicTs::from_nanos(100_000_000));

        // Tentative tracks should be evicted (oldest first) to reach cap.
        assert_eq!(store.track_count(), 2);
        // Oldest tentative (3) evicted first, then (4).
        assert!(store.get_track(&TrackId::new(3)).is_none());
        assert!(store.get_track(&TrackId::new(4)).is_none());
        // Confirmed remain.
        assert!(store.get_track(&TrackId::new(1)).is_some());
        assert!(store.get_track(&TrackId::new(2)).is_some());
    }

    #[test]
    fn hard_cap_strictly_bounded_under_high_id_churn_no_evictable() {
        // Simulates rapid ID churn with only Confirmed tracks:
        // commit_track should reject new admission when at cap.
        let retention = RetentionPolicy {
            max_concurrent_tracks: 3,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let epoch = ViewEpoch::INITIAL;

        // Fill to cap with Confirmed tracks.
        for i in 1..=3u64 {
            let confirmed = make_track_with_state(i, TrackState::Confirmed);
            let accepted = store.commit_track(&confirmed, MonotonicTs::from_nanos(i * 1_000_000), epoch);
            assert!(accepted, "track {i} should be accepted (under cap)");
        }

        // Try to admit new Confirmed tracks beyond cap — should be rejected.
        for i in 100..=105u64 {
            let confirmed = make_track_with_state(i, TrackState::Confirmed);
            let accepted = store.commit_track(&confirmed, MonotonicTs::from_nanos(i * 1_000_000), epoch);
            assert!(!accepted, "track {i} should be rejected (at cap, no evictable)");
        }

        assert_eq!(store.track_count(), 3, "count must never exceed cap");

        // Updates to existing tracks are always accepted.
        let updated = make_track_with_state(1, TrackState::Confirmed);
        let accepted = store.commit_track(&updated, MonotonicTs::from_nanos(200_000_000), epoch);
        assert!(accepted, "updates to existing tracks must always be accepted");
        assert_eq!(store.track_count(), 3);
    }

    #[test]
    fn admission_allowed_when_evictable_candidate_exists_at_cap() {
        let retention = RetentionPolicy {
            max_concurrent_tracks: 2,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let epoch = ViewEpoch::INITIAL;

        // 1 Confirmed + 1 Lost = at cap.
        let confirmed = make_track_with_state(1, TrackState::Confirmed);
        store.commit_track(&confirmed, MonotonicTs::from_nanos(1_000_000), epoch);
        let lost = make_track_with_state(2, TrackState::Lost);
        store.commit_track(&lost, MonotonicTs::from_nanos(2_000_000), epoch);
        assert_eq!(store.track_count(), 2);

        // New track should be admitted (Lost is pre-evicted) at cap.
        let new_track = make_track_with_state(3, TrackState::Confirmed);
        let accepted = store.commit_track(&new_track, MonotonicTs::from_nanos(3_000_000), epoch);
        assert!(accepted, "new track should be admitted when evictable candidate exists");
        // Pre-eviction removes the Lost victim during commit — count stays at cap.
        assert_eq!(store.track_count(), 2, "strict cap: count stays at max");
        assert!(store.get_track(&TrackId::new(2)).is_none(), "Lost victim should be pre-evicted");
        assert!(store.get_track(&TrackId::new(3)).is_some(), "new track should be present");
    }

    #[test]
    fn strict_cap_mixed_states_burst_never_exceeds() {
        // Burst-admission of mixed-state tracks should never push
        // the store above max_concurrent_tracks.
        let cap = 4;
        let retention = RetentionPolicy {
            max_concurrent_tracks: cap,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let epoch = ViewEpoch::INITIAL;

        // Phase 1: fill to cap with 2 Confirmed + 1 Lost + 1 Tentative.
        let c1 = make_track_with_state(1, TrackState::Confirmed);
        store.commit_track(&c1, MonotonicTs::from_nanos(1_000_000), epoch);
        let c2 = make_track_with_state(2, TrackState::Confirmed);
        store.commit_track(&c2, MonotonicTs::from_nanos(2_000_000), epoch);
        let l3 = make_track_with_state(3, TrackState::Lost);
        store.commit_track(&l3, MonotonicTs::from_nanos(3_000_000), epoch);
        let t4 = make_track_with_state(4, TrackState::Tentative);
        store.commit_track(&t4, MonotonicTs::from_nanos(4_000_000), epoch);
        assert_eq!(store.track_count(), cap);

        // Phase 2: burst 10 new confirmed tracks.
        let mut admitted = 0u32;
        let mut rejected = 0u32;
        for i in 100..110u64 {
            let t = make_track_with_state(i, TrackState::Confirmed);
            if store.commit_track(&t, MonotonicTs::from_nanos(i * 1_000_000), epoch) {
                admitted += 1;
            } else {
                rejected += 1;
            }
            // Invariant: never exceeds cap.
            assert!(
                store.track_count() <= cap,
                "count {} exceeds cap {} after admitting track {i}",
                store.track_count(),
                cap,
            );
        }

        // Lost (3) + Tentative (4) = 2 evictable victims, so 2 admissions
        // succeed; remaining 8 are rejected (only Confirmed remain).
        assert_eq!(admitted, 2, "should admit exactly as many as victims available");
        assert_eq!(rejected, 8);
        assert_eq!(store.track_count(), cap);
    }

    // ------------------------------------------------------------------
    // Trajectory segmentation tests
    // ------------------------------------------------------------------

    #[test]
    fn commit_creates_initial_trajectory_segment() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(hist.trajectory.segment_count(), 1);

        let seg = hist.trajectory.active_segment().unwrap();
        assert_eq!(seg.view_epoch, epoch);
        assert_eq!(seg.opened_by, SegmentBoundary::TrackCreated);
        assert!(seg.is_active());
        assert_eq!(seg.points.len(), 1);
    }

    #[test]
    fn trajectory_accumulates_points_within_segment() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        for i in 0..5u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 33_333_333), epoch);
        }

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(hist.trajectory.segment_count(), 1);
        assert_eq!(hist.trajectory.total_points(), 5);

        // Motion features should be computed.
        let seg = hist.trajectory.active_segment().unwrap();
        // With all observations at the same bbox, displacement ≈ 0.
        assert!(seg.motion.is_stationary);
    }

    // ------------------------------------------------------------------
    // Snapshot / TemporalAccess propagation tests
    // ------------------------------------------------------------------

    #[test]
    fn snapshot_exposes_trajectory_info_via_temporal_access() {
        use nv_perception::TemporalAccess;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        for i in 0..3u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        let snap = store.snapshot();
        let id = TrackId::new(1);

        assert_eq!(snap.trajectory_point_count(&id), 3);
        assert_eq!(snap.trajectory_segment_count(&id), 1);

        // Missing track returns 0.
        let missing = TrackId::new(999);
        assert_eq!(snap.trajectory_point_count(&missing), 0);
        assert_eq!(snap.trajectory_segment_count(&missing), 0);
    }

    #[test]
    fn snapshot_track_ids_matches_committed_tracks() {
        use nv_perception::TemporalAccess;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        for i in 1..=4u64 {
            let track = make_track(i);
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        let snap = store.snapshot();
        let mut ids: Vec<u64> = snap.track_ids().iter().map(|id| id.as_u64()).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    // ------------------------------------------------------------------
    // Provenance propagation via TemporalAccess
    // ------------------------------------------------------------------

    #[test]
    fn snapshot_preserves_first_and_last_seen() {
        use nv_perception::TemporalAccess;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        store.commit_track(&track, MonotonicTs::from_nanos(100), epoch);
        store.commit_track(&track, MonotonicTs::from_nanos(200), epoch);
        store.commit_track(&track, MonotonicTs::from_nanos(300), epoch);

        let snap = store.snapshot();
        let id = TrackId::new(1);

        assert_eq!(snap.first_seen(&id), Some(MonotonicTs::from_nanos(100)));
        assert_eq!(snap.last_seen(&id), Some(MonotonicTs::from_nanos(300)));
    }

    #[test]
    fn snapshot_observations_reflect_commit_order() {
        use nv_perception::TemporalAccess;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        // Create tracks with distinct observation timestamps.
        let mut track1 = make_track(1);
        track1.current.ts = MonotonicTs::from_nanos(100);
        store.commit_track(&track1, MonotonicTs::from_nanos(100), epoch);

        let mut track2 = make_track(1);
        track2.current.ts = MonotonicTs::from_nanos(200);
        store.commit_track(&track2, MonotonicTs::from_nanos(200), epoch);

        let snap = store.snapshot();
        let obs = snap.recent_observations(&TrackId::new(1));
        assert_eq!(obs.len(), 2);
        // Oldest first.
        assert!(obs[0].ts < obs[1].ts);
    }

    #[test]
    fn clear_resets_all_state() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        for i in 1..=5u64 {
            let track = make_track(i);
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        assert_eq!(store.track_count(), 5);
        store.clear();
        assert_eq!(store.track_count(), 0);
    }

    // ------------------------------------------------------------------
    // Epoch-transition integration tests (P1 / P2)
    // ------------------------------------------------------------------

    #[test]
    fn commit_track_segments_trajectory_on_epoch_change() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let epoch0 = ViewEpoch::INITIAL;
        let epoch1 = epoch0.next();

        // 3 commits in epoch 0.
        for i in 0..3u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch0);
        }
        assert_eq!(
            store
                .get_track(&TrackId::new(1))
                .unwrap()
                .trajectory
                .segment_count(),
            1
        );

        // Epoch changes — next commit should segment.
        store.set_view_epoch(epoch1);
        store.commit_track(&track, MonotonicTs::from_nanos(10_000_000), epoch1);

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(hist.trajectory.segment_count(), 2);

        // First segment should be closed with EpochChange.
        let seg0 = &hist.trajectory.segments[0];
        assert!(!seg0.is_active());
        assert_eq!(seg0.points.len(), 3);
        assert_eq!(
            seg0.closed_by,
            Some(SegmentBoundary::EpochChange {
                from_epoch: epoch0,
                to_epoch: epoch1,
            })
        );

        // Second segment should be active with EpochChange as opened_by.
        let seg1 = &hist.trajectory.segments[1];
        assert!(seg1.is_active());
        assert_eq!(
            seg1.opened_by,
            SegmentBoundary::EpochChange {
                from_epoch: epoch0,
                to_epoch: epoch1,
            }
        );
        assert_eq!(seg1.points.len(), 1);
        assert_eq!(seg1.view_epoch, epoch1);
    }

    #[test]
    fn commit_track_segments_across_multiple_epochs() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let track = make_track(1);
        let epoch0 = ViewEpoch::INITIAL;
        let epoch1 = epoch0.next();
        let epoch2 = epoch1.next();

        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch0);
        store.commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch1);
        store.commit_track(&track, MonotonicTs::from_nanos(3_000_000), epoch2);

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(hist.trajectory.segment_count(), 3);
        assert!(!hist.trajectory.segments[0].is_active());
        assert!(!hist.trajectory.segments[1].is_active());
        assert!(hist.trajectory.segments[2].is_active());
        assert_eq!(hist.trajectory.total_points(), 3);
    }

    #[test]
    fn commit_track_closes_segment_on_lost_state() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

        // Transition to Lost.
        let lost = make_track_with_state(1, TrackState::Lost);
        store.commit_track(&lost, MonotonicTs::from_nanos(2_000_000), epoch);

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert_eq!(hist.trajectory.segment_count(), 1);
        let seg = &hist.trajectory.segments[0];
        assert!(!seg.is_active());
        assert_eq!(seg.closed_by, Some(SegmentBoundary::TrackLost));
        // Both points were appended before close.
        assert_eq!(seg.points.len(), 2);
    }

    #[test]
    fn close_all_segments_marks_feed_restart() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        for i in 1..=3u64 {
            let track = make_track(i);
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        // All 3 tracks should have active segments.
        for (_, hist) in store.tracks() {
            assert!(hist.trajectory.active_segment().is_some());
        }

        store.close_all_segments(SegmentBoundary::FeedRestart);

        // All segments should now be closed with FeedRestart.
        for (_, hist) in store.tracks() {
            assert!(hist.trajectory.active_segment().is_none());
            let seg = hist.trajectory.segments.last().unwrap();
            assert_eq!(seg.closed_by, Some(SegmentBoundary::FeedRestart));
        }
    }

    // ------------------------------------------------------------------
    // Trajectory retention (point pruning) tests (P1)
    // ------------------------------------------------------------------

    #[test]
    fn commit_track_prunes_trajectory_points_per_track() {
        let retention = RetentionPolicy {
            max_trajectory_points_per_track: 5,
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        for i in 0..10u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert!(
            hist.trajectory.total_points() <= 5,
            "trajectory should be pruned to max_trajectory_points_per_track"
        );
    }

    #[test]
    fn enforce_retention_prunes_trajectory_points() {
        let retention = RetentionPolicy {
            max_trajectory_points_per_track: 3,
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let track = make_track(1);
        let epoch = ViewEpoch::INITIAL;

        // commit_track also prunes, but enforce_retention does a second pass.
        // Build up points across epoch segments.
        let epoch1 = epoch.next();
        for i in 0..5u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }
        for i in 5..10u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch1);
        }

        store.enforce_retention(MonotonicTs::from_nanos(20_000_000));

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert!(
            hist.trajectory.total_points() <= 3,
            "trajectory points should be pruned by enforce_retention: got {}",
            hist.trajectory.total_points()
        );
    }

    #[test]
    fn trajectory_pruning_removes_oldest_closed_segments_first() {
        let retention = RetentionPolicy {
            max_trajectory_points_per_track: 4,
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let track = make_track(1);
        let epoch0 = ViewEpoch::INITIAL;
        let epoch1 = epoch0.next();

        // 3 points in epoch 0, then 3 points in epoch 1.
        for i in 0..3u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch0);
        }
        for i in 3..6u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch1);
        }

        let hist = store.get_track(&TrackId::new(1)).unwrap();
        assert!(hist.trajectory.total_points() <= 4);
        // The remaining points should come from the most recent segment(s).
        if let Some(seg) = hist.trajectory.segments.last() {
            assert_eq!(seg.view_epoch, epoch1);
        }
    }

    // ------------------------------------------------------------------
    // TrackStore / TrajectoryStoreAccess trait tests (P2)
    // ------------------------------------------------------------------

    #[test]
    fn track_store_trait_on_temporal_store() {
        use super::TrackStore;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;
        let id = TrackId::new(1);

        assert!(!store.has_track(&id));

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

        assert!(store.has_track(&id));
        assert_eq!(store.track_state(&id), Some(TrackState::Confirmed));
        assert_eq!(store.track_creation_epoch(&id), Some(epoch));
    }

    #[test]
    fn track_store_trait_on_snapshot() {
        use super::TrackStore;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

        let snap = store.snapshot();
        let id = TrackId::new(1);

        assert!(snap.has_track(&id));
        assert_eq!(snap.track_state(&id), Some(TrackState::Confirmed));
        assert_eq!(snap.track_creation_epoch(&id), Some(epoch));
        assert!(!snap.has_track(&TrackId::new(999)));
    }

    #[test]
    fn trajectory_store_trait_on_temporal_store() {
        use super::TrajectoryStoreAccess;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;
        let id = TrackId::new(1);

        assert!(store.trajectory(&id).is_none());

        let track = make_track(1);
        for i in 0..3u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        let traj = store.trajectory(&id).unwrap();
        assert_eq!(traj.total_points(), 3);
        assert_eq!(traj.segment_count(), 1);
        assert!(traj.active_segment().is_some());
    }

    #[test]
    fn trajectory_store_trait_on_snapshot() {
        use super::TrajectoryStoreAccess;

        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        let track = make_track(1);
        for i in 0..3u64 {
            store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        }

        let snap = store.snapshot();
        let id = TrackId::new(1);

        let traj = TrajectoryStoreAccess::trajectory(&snap, &id).unwrap();
        assert_eq!(traj.total_points(), 3);
        assert_eq!(traj.segment_count(), 1);
        assert!(TrajectoryStoreAccess::trajectory(&snap, &TrackId::new(999)).is_none());
    }

    // ------------------------------------------------------------------
    // P2: Hard-cap eviction priority (Lost before Coasted)
    // ------------------------------------------------------------------

    #[test]
    fn hard_cap_evicts_lost_before_coasted() {
        let retention = RetentionPolicy {
            max_concurrent_tracks: 2,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);

        // Insert directly to bypass pre-eviction — testing enforce_retention
        // priority: Lost should be evicted before Coasted.
        insert_track_with_state(&mut store, 1, TrackState::Coasted, 1_000_000);
        insert_track_with_state(&mut store, 2, TrackState::Confirmed, 2_000_000);
        insert_track_with_state(&mut store, 3, TrackState::Lost, 3_000_000);
        assert_eq!(store.track_count(), 3);

        // Enforce — should evict Lost (track 3) first, even though Coasted
        // (track 1) is older.
        store.enforce_retention(MonotonicTs::from_nanos(5_000_000));
        assert_eq!(store.track_count(), 2);
        assert!(
            store.get_track(&TrackId::new(3)).is_none(),
            "Lost track should be evicted before Coasted"
        );
        assert!(
            store.get_track(&TrackId::new(1)).is_some(),
            "Coasted track should survive when Lost tracks are available"
        );
    }

    #[test]
    fn hard_cap_falls_back_to_coasted_when_no_lost() {
        let retention = RetentionPolicy {
            max_concurrent_tracks: 2,
            max_track_age: nv_core::Duration::from_secs(600),
            ..RetentionPolicy::default()
        };
        let mut store = TemporalStore::new(retention);
        let epoch = ViewEpoch::INITIAL;

        // 2 Coasted + 1 Confirmed = 3 total.
        let coasted1 = make_track_with_state(1, TrackState::Coasted);
        store.commit_track(&coasted1, MonotonicTs::from_nanos(1_000_000), epoch);

        let coasted2 = make_track_with_state(2, TrackState::Coasted);
        store.commit_track(&coasted2, MonotonicTs::from_nanos(2_000_000), epoch);

        let confirmed = make_track_with_state(3, TrackState::Confirmed);
        store.commit_track(&confirmed, MonotonicTs::from_nanos(3_000_000), epoch);

        store.enforce_retention(MonotonicTs::from_nanos(5_000_000));
        assert_eq!(store.track_count(), 2);
        assert!(
            store.get_track(&TrackId::new(1)).is_none(),
            "oldest Coasted should be evicted when no Lost tracks exist"
        );
        assert!(store.get_track(&TrackId::new(2)).is_some());
        assert!(store.get_track(&TrackId::new(3)).is_some());
    }

    // ------------------------------------------------------------------
    // P3: end_track emits TrackEnded boundary
    // ------------------------------------------------------------------

    #[test]
    fn end_track_closes_segment_with_track_ended() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);
        store.commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch);

        let removed = store.end_track(&TrackId::new(1));
        assert!(removed.is_some());

        let history = removed.unwrap();
        let seg = history.trajectory.segments.last().unwrap();
        assert!(!seg.is_active());
        assert_eq!(seg.closed_by, Some(SegmentBoundary::TrackEnded));
        assert_eq!(seg.points.len(), 2);
    }

    #[test]
    fn end_track_removes_track_from_store() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);
        assert_eq!(store.track_count(), 1);

        store.end_track(&TrackId::new(1));
        assert_eq!(store.track_count(), 0);
    }

    #[test]
    fn end_track_returns_none_for_missing_track() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        assert!(store.end_track(&TrackId::new(999)).is_none());
    }

    // ------------------------------------------------------------------
    // P1: apply_compensation transforms trajectory points
    // ------------------------------------------------------------------

    #[test]
    fn apply_compensation_transforms_active_segment_points() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

        // All points start at bbox center (0.05, 0.05).
        let id = TrackId::new(1);
        let before = store.get_track(&id).unwrap().trajectory.segments[0]
            .points[0]
            .position;

        // Apply a translation of (+0.1, +0.2).
        let t = nv_core::AffineTransform2D::new(1.0, 0.0, 0.1, 0.0, 1.0, 0.2);
        store.apply_compensation(&t, epoch);

        let after = store.get_track(&id).unwrap().trajectory.segments[0]
            .points[0]
            .position;
        assert!(
            (after.x - (before.x + 0.1)).abs() < 1e-5,
            "x should be translated"
        );
        assert!(
            (after.y - (before.y + 0.2)).abs() < 1e-5,
            "y should be translated"
        );

        // Compensation metadata should be recorded.
        let seg = &store.get_track(&id).unwrap().trajectory.segments[0];
        assert!(seg.compensation.is_some());
        assert_eq!(seg.compensation_count, 1);
    }

    #[test]
    fn apply_compensation_ignores_wrong_epoch() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;
        let other_epoch = epoch.next();

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

        let id = TrackId::new(1);
        let before = store.get_track(&id).unwrap().trajectory.segments[0]
            .points[0]
            .position;

        // Apply compensation for a different epoch — should be skipped.
        let t = nv_core::AffineTransform2D::new(1.0, 0.0, 0.5, 0.0, 1.0, 0.5);
        store.apply_compensation(&t, other_epoch);

        let after = store.get_track(&id).unwrap().trajectory.segments[0]
            .points[0]
            .position;
        assert!(
            (after.x - before.x).abs() < 1e-6,
            "position should be unchanged for wrong-epoch compensation"
        );
    }

    #[test]
    fn apply_compensation_composes_multiple_transforms() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        let track = make_track(1);
        store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

        let t1 = nv_core::AffineTransform2D::new(1.0, 0.0, 0.1, 0.0, 1.0, 0.0);
        let t2 = nv_core::AffineTransform2D::new(1.0, 0.0, 0.2, 0.0, 1.0, 0.0);
        store.apply_compensation(&t1, epoch);
        store.apply_compensation(&t2, epoch);

        let seg = &store.get_track(&TrackId::new(1)).unwrap().trajectory.segments[0];
        assert_eq!(seg.compensation_count, 2);
        // Cumulative transform should be t1 then t2 = translate by 0.3.
        let comp = seg.compensation.unwrap();
        assert!((comp.m[2] - 0.3).abs() < 1e-9, "cumulative tx should be ~0.3");
    }

    // ------------------------------------------------------------------
    // end_track on already-closed segment (Lost → end_track)
    // ------------------------------------------------------------------

    #[test]
    fn end_track_on_already_closed_segment_preserves_track_lost() {
        let mut store = TemporalStore::new(RetentionPolicy::default());
        let epoch = ViewEpoch::INITIAL;

        // Commit a Lost track — segment is closed with TrackLost.
        let lost = make_track_with_state(1, TrackState::Lost);
        store.commit_track(&lost, MonotonicTs::from_nanos(1_000_000), epoch);
        assert!(
            store
                .get_track(&TrackId::new(1))
                .unwrap()
                .trajectory
                .active_segment()
                .is_none(),
            "Lost track's segment should already be closed"
        );

        // end_track should still remove it even though there's no active segment.
        let removed = store.end_track(&TrackId::new(1));
        assert!(removed.is_some());
        assert_eq!(store.track_count(), 0);

        // The last segment should still have TrackLost (not overwritten).
        let hist = removed.unwrap();
        let seg = hist.trajectory.segments.last().unwrap();
        assert_eq!(
            seg.closed_by,
            Some(SegmentBoundary::TrackLost),
            "pre-existing TrackLost boundary should be preserved"
        );
    }
}
