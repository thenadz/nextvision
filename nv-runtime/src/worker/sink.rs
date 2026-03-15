use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nv_core::error::RuntimeError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use tokio::sync::broadcast;

use crate::output::{OutputSink, SharedOutput};

// ---------------------------------------------------------------------------
// SinkWorker — decoupled per-feed sink thread
// ---------------------------------------------------------------------------

/// Maximum time to wait for the sink worker thread to finish during
/// shutdown. If the sink's `emit()` is blocked (e.g., on network I/O),
/// we detach the thread rather than hang the feed/runtime shutdown.
const SINK_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Minimum interval between consecutive `SinkBackpressure` health events.
pub(super) const SINK_BP_THROTTLE_INTERVAL: Duration = Duration::from_secs(1);

/// Manages a dedicated thread that calls [`OutputSink::emit()`] asynchronously,
/// isolating the feed processing thread from slow downstream I/O.
pub(super) struct SinkWorker {
    tx: std::sync::mpsc::SyncSender<SharedOutput>,
    thread: Option<std::thread::JoinHandle<Box<dyn OutputSink>>>,
    occupancy: Arc<AtomicUsize>,
}

impl SinkWorker {
    /// Spawn a sink worker thread for the given feed.
    pub(super) fn spawn(
        feed_id: FeedId,
        sink: Box<dyn OutputSink>,
        health_tx: broadcast::Sender<HealthEvent>,
        capacity: usize,
        occupancy: Arc<AtomicUsize>,
    ) -> Result<Self, RuntimeError> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<SharedOutput>(capacity);
        let occ = Arc::clone(&occupancy);
        let thread = std::thread::Builder::new()
            .name(format!("nv-sink-{}", feed_id))
            .spawn(move || Self::run(feed_id, sink, rx, health_tx, occ))
            .map_err(|e| RuntimeError::ThreadSpawnFailed {
                detail: format!("sink worker for {feed_id}: {e}"),
            })?;
        Ok(Self {
            tx,
            thread: Some(thread),
            occupancy,
        })
    }

    /// Enqueue output for the sink. Returns `true` if accepted, `false`
    /// if the sink queue is full (output dropped, throttled health event).
    pub(super) fn send(
        &self,
        output: SharedOutput,
        sink_bp: &mut SinkBpThrottle,
        health_tx: &broadcast::Sender<HealthEvent>,
        feed_id: FeedId,
    ) -> bool {
        // Increment *before* try_send so the sink thread's
        // fetch_sub on recv cannot race ahead of the add and
        // transiently underflow, which would produce bogus
        // telemetry depth values.
        self.occupancy.fetch_add(1, Ordering::Relaxed);
        match self.tx.try_send(output) {
            Ok(()) => true,
            Err(std::sync::mpsc::TrySendError::Full(rejected)) => {
                self.occupancy.fetch_sub(1, Ordering::Relaxed);
                sink_bp.record_drop(health_tx, feed_id);
                let _ = rejected; // drop explicitly
                false
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                self.occupancy.fetch_sub(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Sink thread main loop. Drains the channel until the sender is
    /// dropped, then returns the sink so it can be reused across restart
    /// sessions.
    fn run(
        feed_id: FeedId,
        sink: Box<dyn OutputSink>,
        rx: std::sync::mpsc::Receiver<SharedOutput>,
        health_tx: broadcast::Sender<HealthEvent>,
        occupancy: Arc<AtomicUsize>,
    ) -> Box<dyn OutputSink> {
        while let Ok(output) = rx.recv() {
            occupancy.fetch_sub(1, Ordering::Relaxed);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                sink.emit(output);
            }));
            if result.is_err() {
                tracing::error!(
                    feed_id = %feed_id,
                    "OutputSink::emit() panicked — output dropped",
                );
                let _ = health_tx.send(HealthEvent::SinkPanic { feed_id });
            }
        }
        sink
    }

    /// Shut down the sink worker: drop the sender (closes the channel),
    /// join the thread with a bounded timeout, and return the recovered
    /// sink for reuse.
    ///
    /// If the sink thread does not finish within [`SINK_SHUTDOWN_TIMEOUT`],
    /// it is detached and a [`NullSink`] placeholder is returned. This
    /// prevents a blocked `OutputSink::emit()` from hanging feed
    /// restart or runtime shutdown.
    ///
    /// If the sink thread panicked, returns a [`NullSink`] placeholder.
    pub(super) fn shutdown(mut self, health_tx: &broadcast::Sender<HealthEvent>, feed_id: FeedId) -> Box<dyn OutputSink> {
        // Drop the sender to signal the sink thread to exit.
        drop(self.tx);
        let Some(handle) = self.thread.take() else {
            return Box::new(NullSink);
        };

        // Wait for the thread with a bounded timeout via a rendezvous channel.
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        // We cannot join with a timeout directly, so park the JoinHandle
        // on a small helper thread that sends the result back.
        let _detached = std::thread::Builder::new()
            .name(format!("nv-sink-join-{feed_id}"))
            .spawn(move || {
                let result = handle.join();
                let _ = done_tx.send(result);
            });
        match done_rx.recv_timeout(SINK_SHUTDOWN_TIMEOUT) {
            Ok(Ok(sink)) => sink,
            Ok(Err(_)) => {
                tracing::error!(
                    feed_id = %feed_id,
                    "sink worker thread panicked during shutdown",
                );
                let _ = health_tx.send(HealthEvent::SinkPanic { feed_id });
                Box::new(NullSink)
            }
            Err(_) => {
                // Timed out — the sink thread is blocked in emit().
                // Detach it (the helper thread will eventually join it
                // when emit() returns or the process exits).
                tracing::warn!(
                    feed_id = %feed_id,
                    timeout_secs = SINK_SHUTDOWN_TIMEOUT.as_secs(),
                    "sink worker thread did not finish within timeout — detaching",
                );
                let _ = health_tx.send(HealthEvent::SinkPanic { feed_id });
                Box::new(NullSink)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NullSink — placeholder while the real sink is lent to SinkWorker
// ---------------------------------------------------------------------------

/// Sink placeholder used while the real `OutputSink` is owned by the
/// `SinkWorker` thread during a processing session.
pub(super) struct NullSink;

impl OutputSink for NullSink {
    fn emit(&self, _output: SharedOutput) {}
}

// ---------------------------------------------------------------------------
// SinkBpThrottle — coalesces per-output SinkBackpressure events
// ---------------------------------------------------------------------------

/// Coalesces `SinkBackpressure` events to prevent per-drop storms.
///
/// Same strategy as `BackpressureThrottle` for frame drops:
/// - First drop → emit immediately.
/// - During sustained backpressure → emit at most once per
///   [`SINK_BP_THROTTLE_INTERVAL`], carrying the accumulated delta.
///
/// Lives on the feed worker thread (single-threaded access, no Mutex).
pub(super) struct SinkBpThrottle {
    in_backpressure: bool,
    accumulated: u64,
    last_event: Instant,
}

impl SinkBpThrottle {
    pub(super) fn new() -> Self {
        Self {
            in_backpressure: false,
            accumulated: 0,
            last_event: Instant::now(),
        }
    }

    pub(super) fn record_drop(&mut self, health_tx: &broadcast::Sender<HealthEvent>, feed_id: FeedId) {
        self.accumulated += 1;

        if !self.in_backpressure {
            // Transition into backpressure — emit immediately.
            self.in_backpressure = true;
            let delta = self.accumulated;
            self.accumulated = 0;
            self.last_event = Instant::now();
            let _ = health_tx.send(HealthEvent::SinkBackpressure {
                feed_id,
                outputs_dropped: delta,
            });
            return;
        }

        // Sustained — emit at most once per throttle interval.
        if self.last_event.elapsed() >= SINK_BP_THROTTLE_INTERVAL {
            let delta = self.accumulated;
            self.accumulated = 0;
            self.last_event = Instant::now();
            let _ = health_tx.send(HealthEvent::SinkBackpressure {
                feed_id,
                outputs_dropped: delta,
            });
        }
    }

    /// Flush any accumulated but un-emitted drop count.
    pub(super) fn flush(&mut self, health_tx: &broadcast::Sender<HealthEvent>, feed_id: FeedId) {
        if self.accumulated > 0 {
            let delta = self.accumulated;
            self.accumulated = 0;
            self.in_backpressure = false;
            let _ = health_tx.send(HealthEvent::SinkBackpressure {
                feed_id,
                outputs_dropped: delta,
            });
        }
    }
}
