use std::collections::HashMap;
use std::sync::Arc;

use nv_core::{MonotonicTs, TrackId};
use nv_perception::{Track, TrackObservation};
use nv_view::ViewEpoch;

use crate::trajectory::Trajectory;

use super::{TemporalStore, TrackHistory};

/// Internal snapshot data.
#[derive(Clone, Debug)]
pub(super) struct SnapshotInner {
    pub tracks: HashMap<TrackId, SnapshotTrackHistory>,
    pub view_epoch: ViewEpoch,
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
    pub(super) fn from_history(h: &mut TrackHistory) -> Self {
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
    pub(super) inner: Arc<SnapshotInner>,
}

impl TemporalStoreSnapshot {
    /// Build a snapshot from the live store.
    pub(super) fn from_store(store: &mut TemporalStore) -> Self {
        Self {
            inner: Arc::new(SnapshotInner {
                tracks: store
                    .tracks
                    .iter_mut()
                    .map(|(k, v)| (*k, SnapshotTrackHistory::from_history(v)))
                    .collect(),
                view_epoch: store.view_epoch,
            }),
        }
    }

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
