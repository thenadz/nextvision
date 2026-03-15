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

mod commit;
mod snapshot;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use nv_core::{MonotonicTs, TrackId};
use nv_perception::{Track, TrackObservation};
use nv_view::ViewEpoch;

use crate::retention::RetentionPolicy;
use crate::trajectory::Trajectory;

// Re-export public types from submodules.
pub use snapshot::{
    SnapshotTrackHistory, TemporalStoreSnapshot, TrackStore, TrajectoryStoreAccess,
};

/// Per-feed temporal state manager.
///
/// Owns all track histories, trajectories, and motion features for a
/// single feed. Not shared between feeds.
///
/// The store is mutated only by the pipeline executor between frames
/// (after all stages have run), never during stage execution.
pub struct TemporalStore {
    pub(super) tracks: HashMap<TrackId, TrackHistory>,
    pub(super) retention: RetentionPolicy,
    pub(super) view_epoch: ViewEpoch,
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

    /// Low-level insert or replace of a track history.
    ///
    /// **Does not enforce [`RetentionPolicy::max_concurrent_tracks`].**
    /// Prefer [`commit_track`](Self::commit_track) for normal pipeline
    /// usage — it handles pre-eviction and cap enforcement.
    ///
    /// This is intentionally public for test harnesses and advanced
    /// integrations that manage track lifecycle externally.
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
        TemporalStoreSnapshot::from_store(self)
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
    pub(super) fn find_eviction_victim(&self) -> Option<TrackId> {
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
}

// ---------------------------------------------------------------------------
// TrackHistory
// ---------------------------------------------------------------------------

/// Complete history of a single track within the temporal store.
#[derive(Clone, Debug)]
pub struct TrackHistory {
    /// Current track state (`Arc`-wrapped so snapshots share the allocation).
    pub track: Arc<Track>,
    /// Ring buffer of recent observations (private — mutate through methods
    /// so the generation counter stays consistent with the snapshot cache).
    pub(super) observations: VecDeque<TrackObservation>,
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
    pub(super) observation_gen: u64,
    /// Cached `Arc<[TrackObservation]>` from the last snapshot. Reused when
    /// `observation_gen` has not advanced.
    pub(super) cached_observations: Option<(u64, Arc<[TrackObservation]>)>,
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
    pub(super) fn snapshot_observations(&mut self) -> Arc<[TrackObservation]> {
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

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
