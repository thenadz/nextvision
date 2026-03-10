//! Health events and stop reasons.
//!
//! [`HealthEvent`] is the primary mechanism for observing runtime behavior.
//! Events are broadcast to subscribers via a channel. They cover source
//! lifecycle, stage errors, feed restarts, backpressure, and view-state changes.

use crate::error::{MediaError, StageError};
use crate::id::{FeedId, StageId};

/// A health event emitted by the runtime.
///
/// Subscribe via [`Runtime::health_subscribe()`](crate) (aggregate).
/// Per-feed filtering is the subscriber's responsibility.
#[derive(Debug, Clone)]
pub enum HealthEvent {
    /// The video source connected successfully.
    SourceConnected { feed_id: FeedId },

    /// The video source disconnected.
    SourceDisconnected { feed_id: FeedId, reason: MediaError },

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
    FeedRestarting { feed_id: FeedId, restart_count: u32 },

    /// The feed has stopped permanently.
    FeedStopped { feed_id: FeedId, reason: StopReason },

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

    /// The output broadcast channel is saturated — the internal sentinel
    /// receiver observed ring-buffer wrap.
    ///
    /// An internal sentinel receiver (the slowest possible consumer)
    /// monitors the aggregate output channel. When production outpaces
    /// the sentinel's drain interval, the ring buffer wraps past the
    /// sentinel's read position and `messages_lost` reports how many
    /// messages the sentinel missed.
    ///
    /// **What this means:** the channel is under backpressure. The
    /// sentinel observes worst-case wrap — it does **not** prove that
    /// any specific external subscriber lost messages. A subscriber
    /// consuming faster than the sentinel will experience less (or no)
    /// loss. Treat this as a saturation / capacity warning, not a
    /// per-subscriber loss report.
    ///
    /// **Attribution:** this is a runtime-global event. The output
    /// channel is shared across all feeds, so saturation is not
    /// attributable to any single feed.
    ///
    /// **`messages_lost` semantics:** per-event delta — the number of
    /// messages the sentinel missed since the previous `OutputLagged`
    /// event (or since runtime start for the first event). This is
    /// **not** a cumulative total and **not** a count of messages lost
    /// by external subscribers.
    ///
    /// **Throttling:** events are coalesced to prevent storms. The
    /// runtime emits one event on the initial saturation transition,
    /// then at most one per second during sustained saturation, each
    /// carrying the accumulated sentinel-observed delta for that
    /// interval.
    ///
    /// **Action:** consider increasing `output_capacity` or improving
    /// subscriber throughput.
    OutputLagged {
        /// Sentinel-observed ring-buffer wrap since the previous
        /// `OutputLagged` event (per-event delta, not cumulative).
        /// Reflects channel saturation, not guaranteed per-subscriber
        /// loss.
        messages_lost: u64,
    },

    /// The source reached end-of-stream (file sources).
    ///
    /// For non-looping file sources this is terminal: the feed stops
    /// with [`StopReason::EndOfStream`] rather than restarting.
    SourceEos { feed_id: FeedId },

    /// The per-feed [`OutputSink`] panicked during `emit()`.
    ///
    /// The output is dropped but the feed continues processing.
    /// The runtime wraps `OutputSink::emit()` in `catch_unwind` to
    /// prevent a misbehaving sink from tearing down the worker thread.
    SinkPanic { feed_id: FeedId },
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
