//! Metrics helpers for per-feed instrumentation.
//!
//! The library uses the `tracing` crate for structured logging and depends
//! on user-configured subscribers for export. This module provides helpers
//! for per-feed metric recording and snapshot types.
//!
//! The actual metrics recording will be implemented when the runtime is built.
//! This module defines the metric snapshot types used in the public API.

use crate::id::FeedId;

/// A snapshot of per-feed metrics at a point in time.
///
/// Returned by `FeedHandle::metrics()`.
#[derive(Debug, Clone)]
pub struct FeedMetrics {
    /// Which feed these metrics belong to.
    pub feed_id: FeedId,
    /// Total frames received from the source.
    pub frames_received: u64,
    /// Frames dropped due to backpressure.
    pub frames_dropped: u64,
    /// Frames successfully processed through all stages.
    pub frames_processed: u64,
    /// Number of active tracks in the temporal store.
    pub tracks_active: u64,
    /// Current view epoch value.
    pub view_epoch: u64,
    /// Number of feed restarts.
    pub restarts: u32,
}

/// Per-stage metrics snapshot, included in [`StageContext`](crate) for
/// stages that want to adapt behavior based on their own performance.
#[derive(Debug, Clone)]
pub struct StageMetrics {
    /// Cumulative frame count processed by this stage.
    pub frames_processed: u64,
    /// Cumulative error count for this stage.
    pub errors: u64,
}
