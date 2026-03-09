//! Health events and stop reasons.
//!
//! [`HealthEvent`] is the primary mechanism for observing runtime behavior.
//! Events are broadcast to subscribers via a channel. They cover source
//! lifecycle, stage errors, feed restarts, backpressure, and view-state changes.

use crate::error::{MediaError, StageError};
use crate::id::{FeedId, StageId};

/// A health event emitted by the runtime.
///
/// Subscribe via `FeedHandle::health_recv()` (per-feed) or
/// `Runtime::health_recv()` (aggregate).
#[derive(Debug)]
pub enum HealthEvent {
    /// The video source connected successfully.
    SourceConnected { feed_id: FeedId },

    /// The video source disconnected.
    SourceDisconnected {
        feed_id: FeedId,
        reason: MediaError,
    },

    /// The video source is attempting to reconnect.
    SourceReconnecting { feed_id: FeedId, attempt: u32 },

    /// A stage returned an error for a single frame.
    /// The frame was dropped; the feed continues.
    StageError {
        feed_id: FeedId,
        stage_id: StageId,
        error: StageError,
    },

    /// A stage panicked. The feed will restart per its restart policy.
    StagePanic { feed_id: FeedId, stage_id: StageId },

    /// The feed is restarting.
    FeedRestarting {
        feed_id: FeedId,
        restart_count: u32,
    },

    /// The feed has stopped permanently.
    FeedStopped {
        feed_id: FeedId,
        reason: StopReason,
    },

    /// Frames were dropped due to backpressure.
    BackpressureDrop {
        feed_id: FeedId,
        frames_dropped: u64,
    },

    /// The view epoch changed (camera discontinuity detected).
    ViewEpochChanged {
        feed_id: FeedId,
        /// New epoch value (from `nv-view`, represented as u64 here to avoid
        /// circular dependency; re-exported with proper type in `nv-runtime`).
        epoch: u64,
    },

    /// The view validity has degraded due to camera motion.
    ViewDegraded {
        feed_id: FeedId,
        stability_score: f32,
    },

    /// A compensation transform was applied to active tracks.
    ViewCompensationApplied { feed_id: FeedId, epoch: u64 },
}

/// Reason a feed stopped permanently.
#[derive(Debug, Clone)]
pub enum StopReason {
    /// Normal shutdown requested by the user.
    UserRequested,

    /// The restart limit was exceeded after repeated failures.
    RestartLimitExceeded { restart_count: u32 },

    /// End of stream on a file source.
    EndOfStream,

    /// An unrecoverable error occurred.
    Fatal { detail: String },
}
