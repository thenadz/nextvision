//! Health events, stop reasons, and decode outcome classification.
//!
//! [`HealthEvent`] is the primary mechanism for observing runtime behavior.
//! Events are broadcast to subscribers via a channel. They cover source
//! lifecycle, stage errors, feed restarts, backpressure, and view-state changes.
//!
//! [`DecodeOutcome`] provides a backend-neutral classification of which
//! decoder class is in effect (hardware, software, or unknown).

use crate::error::{MediaError, StageError};
use crate::id::{FeedId, StageId};

// ---------------------------------------------------------------------------
// Decode outcome — backend-neutral decoder classification
// ---------------------------------------------------------------------------

/// Backend-neutral classification of the effective decoder.
///
/// After the media backend negotiates a decoder for a stream, this
/// categorises the result without exposing backend element names, GPU
/// memory modes, or inference-framework details.
///
/// Used in [`HealthEvent::DecodeDecision`] and the internal
/// `DecodeDecisionInfo` diagnostic report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeOutcome {
    /// A hardware-accelerated video decoder is in use.
    Hardware,
    /// A software (CPU-only) video decoder is in use.
    Software,
    /// The backend could not determine which decoder class is active.
    ///
    /// This can happen with custom pipeline fragments or when the
    /// backend does not expose decoder identity.
    Unknown,
}

impl std::fmt::Display for DecodeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hardware => f.write_str("Hardware"),
            Self::Software => f.write_str("Software"),
            Self::Unknown => f.write_str("Unknown"),
        }
    }
}

/// A health event emitted by the runtime.
///
/// Subscribe via [`Runtime::health_subscribe()`](crate) (aggregate).
/// Per-feed filtering is the subscriber's responsibility.
#[derive(Debug, Clone)]
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

    /// A decode decision was made for a feed's video stream.
    ///
    /// Emitted once per session start (not per frame) when the backend
    /// identifies which decoder was selected for the stream. Provides
    /// visibility into hardware vs. software decode selection without
    /// exposing backend-specific element names in required fields.
    ///
    /// `detail` is a backend-specific diagnostic string (e.g., the
    /// GStreamer element name). It is intended for logging — do not
    /// match on its contents.
    DecodeDecision {
        feed_id: FeedId,
        /// Hardware, Software, or Unknown.
        outcome: DecodeOutcome,
        /// The user-requested decode preference that was in effect.
        preference: String,
        /// Whether the adaptive fallback cache overrode the requested
        /// preference for this session.
        fallback_active: bool,
        /// Human-readable reason for fallback (populated when
        /// `fallback_active` is `true`).
        fallback_reason: Option<String>,
        /// Backend-specific diagnostic detail (element name etc.).
        detail: String,
    },

    /// The per-feed [`OutputSink`] panicked during `emit()`.
    ///
    /// The output is dropped but the feed continues processing.
    /// The runtime wraps `OutputSink::emit()` in `catch_unwind` to
    /// prevent a misbehaving sink from tearing down the worker thread.
    SinkPanic { feed_id: FeedId },

    /// The per-feed sink queue is full — output was dropped.
    ///
    /// The feed continues processing; only the output delivery is
    /// dropped to prevent slow downstream I/O from blocking the
    /// perception pipeline.
    ///
    /// `outputs_dropped` is the number of outputs dropped since the
    /// last `SinkBackpressure` event (per-event delta).
    SinkBackpressure {
        feed_id: FeedId,
        outputs_dropped: u64,
    },

    /// Tracks were rejected by the temporal store's admission
    /// control because the hard cap was reached and no evictable
    /// candidates (Lost/Coasted/Tentative) were available.
    ///
    /// The feed continues processing. This event indicates tracker
    /// saturation — the scene has more confirmed objects than
    /// `max_concurrent_tracks` allows.
    ///
    /// Events are coalesced per frame: a single event is emitted
    /// with the total number of tracks rejected in that frame.
    TrackAdmissionRejected {
        feed_id: FeedId,
        /// Number of tracks rejected in this frame.
        rejected_count: u32,
    },

    /// The batch processor returned an error or panicked.
    ///
    /// All frames in the affected batch are dropped. Each feed thread
    /// waiting on that batch receives the error and skips the frame.
    BatchError {
        processor_id: StageId,
        batch_size: u32,
        error: StageError,
    },

    /// A feed's batch submission was rejected because the coordinator's
    /// queue is full or the coordinator has shut down. The frame is
    /// dropped for this feed.
    ///
    /// Events are coalesced: under sustained overload, the executor
    /// emits one event per throttle window (1 second) with
    /// `dropped_count` reflecting the number of frames rejected in
    /// that window. On recovery (a subsequent successful submission),
    /// any remaining accumulated count is flushed immediately.
    ///
    /// Indicates the batch coordinator is overloaded — either the
    /// processor is too slow for the combined frame rate, or
    /// `max_batch_size` is too small.
    BatchSubmissionRejected {
        feed_id: FeedId,
        processor_id: StageId,
        /// Number of frames rejected in this throttle window.
        dropped_count: u64,
    },

    /// A feed's batch response timed out — the coordinator did not
    /// return a result within `max_latency + response_timeout`.
    ///
    /// Events are coalesced: under sustained timeout conditions, the
    /// executor emits one event per throttle window (1 second) with
    /// `timed_out_count` reflecting the number of timeouts in that
    /// window. On recovery (a subsequent successful submission), any
    /// remaining accumulated count is flushed immediately.
    ///
    /// Indicates the batch processor is slower than the configured
    /// safety margin. Consider increasing `response_timeout` or
    /// reducing `max_batch_size`.
    BatchTimeout {
        feed_id: FeedId,
        processor_id: StageId,
        /// Number of timeouts in this throttle window.
        timed_out_count: u64,
    },

    /// A feed's batch submission was rejected because the feed already
    /// has the maximum number of in-flight items in the coordinator
    /// pipeline (default: 1).
    ///
    /// This occurs when a prior submission timed out on the feed side
    /// but has not yet been processed (or drained) by the coordinator.
    /// The in-flight cap prevents one feed from accumulating orphaned
    /// items in the shared queue.
    ///
    /// Events are coalesced with the same 1-second throttle window as
    /// other batch rejection events. On recovery, any remaining
    /// accumulated count is flushed immediately.
    BatchInFlightExceeded {
        feed_id: FeedId,
        processor_id: StageId,
        /// Number of submissions rejected in this throttle window.
        rejected_count: u64,
    },
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
