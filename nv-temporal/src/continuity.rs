//! Segment boundary types — causal explanations for trajectory splits.

use nv_view::ViewEpoch;

/// Why a trajectory segment boundary (open or close) occurred.
///
/// Every segment boundary is causally explained, making trajectory
/// continuity fully auditable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentBoundary {
    /// First segment for this track (track was newly created).
    TrackCreated,

    /// The `EpochPolicy` returned `Segment`, causing a `ViewEpoch` change.
    EpochChange {
        from_epoch: ViewEpoch,
        to_epoch: ViewEpoch,
    },

    /// Feed was restarted; temporal state was cleared and rebuilt.
    FeedRestart,

    /// Track was lost (coasted too long, evicted by retention policy).
    TrackLost,

    /// Track ended normally (e.g., object left the scene).
    TrackEnded,
}
