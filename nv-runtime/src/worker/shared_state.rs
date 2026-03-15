use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_core::metrics::FeedMetrics;
use nv_media::ingress::HealthSink;
use tokio::sync::broadcast;

use crate::queue::FrameQueue;

// ---------------------------------------------------------------------------
// Shared state between FeedHandle and the worker thread
// ---------------------------------------------------------------------------

/// Shared state accessed atomically by both the `FeedHandle` (user thread)
/// and the feed worker thread.
pub(crate) struct FeedSharedState {
    pub id: FeedId,
    pub paused: AtomicBool,
    pub shutdown: Arc<AtomicBool>,
    pub frames_received: AtomicU64,
    pub frames_dropped: AtomicU64,
    pub frames_processed: AtomicU64,
    pub tracks_active: AtomicU64,
    pub view_epoch: AtomicU64,
    pub restarts: AtomicU32,
    /// Set to `false` when the worker thread exits.
    pub alive: AtomicBool,
    /// Condvar for pause/resume: the worker waits on this instead of spin-sleeping.
    /// The Mutex guards a `bool` that mirrors `paused` — the Condvar wakes
    /// the worker when the pause state changes or shutdown is requested.
    pub pause_condvar: (Mutex<bool>, Condvar),
    /// Frame queue for the current session — set by the worker when the queue
    /// is created, cleared when the session ends. Used by `request_shutdown`
    /// to wake the consumer without waiting for a poll interval.
    queue: Mutex<Option<Arc<FrameQueue>>>,
    /// Atomic occupancy counter for the per-feed output sink queue.
    /// Incremented on successful send, decremented on recv by the sink thread.
    /// Shared (via `Arc` clone) with the [`SinkWorker`] thread.
    pub sink_occupancy: Arc<AtomicUsize>,
    /// Configured capacity of the output sink queue. Written once when the
    /// feed starts, read by `FeedHandle::queue_telemetry()`.
    pub sink_capacity: AtomicUsize,
    /// Instant when the current processing session started (set on each
    /// start or restart). Used by `FeedHandle::uptime()` to report
    /// session-scoped uptime.
    pub session_started_at: Mutex<Instant>,
    /// Last confirmed decode status from the media backend.
    /// Written once per session when the stream starts; read by
    /// `FeedHandle::decode_status()`.
    pub decode_status: Mutex<Option<(nv_core::health::DecodeOutcome, String)>>,
    /// View-system stability score stored as `f32::to_bits()`. Updated
    /// per-frame by the worker. Read by `FeedHandle::diagnostics()`.
    pub view_stability_score: AtomicU32,
    /// View-system context validity ordinal. Updated per-frame by the
    /// worker. 0 = Stable, 1 = Degraded, 2 = Invalid.
    /// Read by `FeedHandle::diagnostics()`.
    pub view_context_validity: AtomicU8,
    /// The batch coordinator this feed submits to, if any.
    /// Set once at spawn time; read by `FeedHandle::diagnostics()`.
    pub batch_processor_id: Option<nv_core::id::StageId>,
}

impl FeedSharedState {
    pub fn new(id: FeedId, batch_processor_id: Option<nv_core::id::StageId>) -> Self {
        Self {
            id,
            paused: AtomicBool::new(false),
            shutdown: Arc::new(AtomicBool::new(false)),
            frames_received: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            frames_processed: AtomicU64::new(0),
            tracks_active: AtomicU64::new(0),
            view_epoch: AtomicU64::new(0),
            restarts: AtomicU32::new(0),
            alive: AtomicBool::new(true),
            pause_condvar: (Mutex::new(false), Condvar::new()),
            queue: Mutex::new(None),
            sink_occupancy: Arc::new(AtomicUsize::new(0)),
            sink_capacity: AtomicUsize::new(0),
            session_started_at: Mutex::new(Instant::now()),
            decode_status: Mutex::new(None),
            view_stability_score: AtomicU32::new(1.0_f32.to_bits()),
            view_context_validity: AtomicU8::new(0),
            batch_processor_id,
        }
    }

    /// Snapshot the current metrics.
    pub fn metrics(&self) -> FeedMetrics {
        FeedMetrics {
            feed_id: self.id,
            frames_received: self.frames_received.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            tracks_active: self.tracks_active.load(Ordering::Relaxed),
            view_epoch: self.view_epoch.load(Ordering::Relaxed),
            restarts: self.restarts.load(Ordering::Relaxed),
        }
    }

    /// Read the current source queue depth and capacity.
    ///
    /// Returns `(depth, capacity)`. If no session is active (between
    /// restarts or after shutdown), returns `(0, 0)`.
    pub fn source_queue_telemetry(&self) -> (usize, usize) {
        if let Ok(guard) = self.queue.lock() {
            if let Some(ref q) = *guard {
                return (q.depth(), q.capacity());
            }
        }
        (0, 0)
    }

    /// Current session uptime (time since last successful start/restart).
    ///
    /// Returns `Duration::ZERO` if the lock is poisoned.
    pub fn session_uptime(&self) -> Duration {
        self.session_started_at
            .lock()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// The last confirmed decode status, if known.
    pub fn decode_status(&self) -> Option<(nv_core::health::DecodeOutcome, String)> {
        self.decode_status
            .lock()
            .ok()
            .and_then(|g| g.clone())
    }

    /// Request shutdown and wake the worker if it is paused or waiting on
    /// the frame queue.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Wake the pause condvar.
        let (_lock, cvar) = &self.pause_condvar;
        cvar.notify_one();
        // Wake the frame queue consumer (event-driven — no poll delay).
        if let Ok(guard) = self.queue.lock() {
            if let Some(ref q) = *guard {
                q.wake_consumer();
            }
        }
    }

    /// Register the current session's queue for shutdown notification.
    pub(crate) fn set_queue(&self, queue: Option<Arc<FrameQueue>>) {
        if let Ok(mut guard) = self.queue.lock() {
            *guard = queue;
        }
    }
}

// ---------------------------------------------------------------------------
// HealthSink adapter — forwards to broadcast channel
// ---------------------------------------------------------------------------

/// Forwards [`HealthEvent`]s from a [`MediaSource`] to the runtime's
/// broadcast channel.
pub(crate) struct BroadcastHealthSink {
    tx: broadcast::Sender<HealthEvent>,
}

impl BroadcastHealthSink {
    pub fn new(tx: broadcast::Sender<HealthEvent>) -> Self {
        Self { tx }
    }
}

impl HealthSink for BroadcastHealthSink {
    fn emit(&self, event: HealthEvent) {
        let _ = self.tx.send(event);
    }
}
