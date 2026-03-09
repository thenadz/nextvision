//! Typed temporal-access contract for stages.
//!
//! [`TemporalAccess`] defines the read-only interface that stages use to
//! query track history and temporal state. It lives in `nv-perception` (not
//! `nv-temporal`) so that stages can depend on it without a circular
//! dependency: `nv-temporal` implements `TemporalAccess` for its snapshot
//! type, and `nv-perception` never imports from `nv-temporal`.

use nv_core::{MonotonicTs, TrackId};
use nv_view::ViewEpoch;

use crate::track::{Track, TrackObservation};

/// Read-only access to temporal state for stages.
///
/// Implemented by `nv_temporal::TemporalStoreSnapshot`. The runtime passes
/// a `&dyn TemporalAccess` through [`StageContext::temporal`](super::StageContext).
///
/// Stages that need track history, observation windows, or view-epoch context
/// program against this trait, keeping them decoupled from the concrete
/// temporal store implementation.
pub trait TemporalAccess: Send + Sync {
    /// View epoch at the time this snapshot was taken.
    fn view_epoch(&self) -> ViewEpoch;

    /// Number of tracks in the snapshot.
    fn track_count(&self) -> usize;

    /// Look up a track's current state by ID.
    ///
    /// Returns `None` if the track is not present in the snapshot.
    fn get_track(&self, id: &TrackId) -> Option<&Track>;

    /// Return the most recent observations for a track, ordered oldest-first.
    ///
    /// Returns an empty slice if the track is not present or has no
    /// recorded observations.
    fn recent_observations(&self, id: &TrackId) -> &[TrackObservation];

    /// Timestamp when a track was first seen.
    ///
    /// Returns `None` if the track is not present.
    fn first_seen(&self, id: &TrackId) -> Option<MonotonicTs>;

    /// Timestamp of the most recent observation for a track.
    ///
    /// Returns `None` if the track is not present.
    fn last_seen(&self, id: &TrackId) -> Option<MonotonicTs>;
}
