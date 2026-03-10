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

use crate::retention::RetentionPolicy;
use crate::trajectory::Trajectory;

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

/// Empty slice constant for returning from `TemporalAccess` when a track is absent.
static EMPTY_OBSERVATIONS: &[TrackObservation] = &[];

impl nv_perception::TemporalAccess for TemporalStoreSnapshot {
    fn view_epoch(&self) -> ViewEpoch {
        self.inner.view_epoch
    }

    fn track_count(&self) -> usize {
        self.inner.tracks.len()
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

        assert_ne!(obs1.len(), obs2.len(), "snap2 should have the new observation");
        assert!(!Arc::ptr_eq(obs1, obs2), "changed observations should have new Arc");
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
}
