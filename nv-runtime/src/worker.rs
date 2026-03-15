//! Per-feed worker thread — owns the source, executor, and processing loop.
//!
//! Each feed runs on a dedicated OS thread. This gives perfect isolation:
//! a stage that blocks or panics affects only its own feed.
//!
//! # Thread model
//!
//! ```text
//! ┌──────────────┐    FrameQueue     ┌─────────────────┐   SinkQueue   ┌───────────┐
//! │ GStreamer     │──── push() ──────▶│ Feed Worker      │── push() ───▶│ Sink      │
//! │ streaming     │                   │ (OS thread)      │              │ Thread    │
//! │ thread        │                   │                  │              │           │
//! │               │                   │  pop() → stages  │              │ emit()    │
//! │  on_error()   │                   │  → broadcast     │              │ (user     │
//! │  on_eos()  ───┼── close() ──────▶│  → health events │              │  sink)    │
//! └──────────────┘                   └─────────────────┘              └───────────┘
//! ```
//!
//! The worker thread owns:
//! - `PipelineExecutor` (stages, temporal store, view state)
//! - Source handle (via `MediaIngressFactory`)
//! - `FrameQueue` (shared with `FeedFrameSink`)
//!
//! Output is decoupled from the feed thread via a bounded
//! per-feed sink queue. The sink thread calls `OutputSink::emit()`
//! asynchronously, preventing slow sinks from blocking perception.
//!
//! Shutdown is coordinated via `FeedSharedState.shutdown` (`AtomicBool`)
//! and the queue's `close()` / `wake_consumer()` methods.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::{MediaError, NvError, RuntimeError};
use nv_core::health::{HealthEvent, StopReason};
use nv_core::id::{FeedId, StageId};
use nv_core::metrics::FeedMetrics;
use nv_frame::FrameEnvelope;
use nv_media::ingress::{FrameSink, HealthSink, MediaIngress, MediaIngressFactory, IngressOptions, SourceStatus};
use tokio::sync::broadcast;

use crate::backpressure::BackpressurePolicy;
use crate::executor::PipelineExecutor;
use crate::feed::FeedConfig;
use crate::output::{LagDetector, OutputSink, SharedOutput};
use crate::queue::{FrameQueue, PopResult, PushOutcome};
use crate::shutdown::{RestartPolicy, RestartTrigger};

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
    pub batch_processor_id: Option<StageId>,
}

impl FeedSharedState {
    pub fn new(id: FeedId, batch_processor_id: Option<StageId>) -> Self {
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
// FrameSink adapter — bridges media ingress → FrameQueue
// ---------------------------------------------------------------------------

/// Adapter that implements [`FrameSink`] by pushing into a [`FrameQueue`].
///
/// Created per feed session. The `Arc<FrameQueue>` is shared with the
/// feed worker thread's pop loop.
struct FeedFrameSink {
    queue: Arc<FrameQueue>,
    shared: Arc<FeedSharedState>,
    health_tx: broadcast::Sender<HealthEvent>,
    feed_id: FeedId,
    bp_throttle: BackpressureThrottle,
}

impl FrameSink for FeedFrameSink {
    fn on_frame(&self, frame: FrameEnvelope) {
        self.shared.frames_received.fetch_add(1, Ordering::Relaxed);
        let outcome = self.queue.push(frame);
        match outcome {
            PushOutcome::Accepted => {}
            PushOutcome::DroppedOldest | PushOutcome::Rejected => {
                self.shared.frames_dropped.fetch_add(1, Ordering::Relaxed);
                self.bp_throttle
                    .record_drop(&self.health_tx, self.feed_id);
            }
        }
    }

    fn on_error(&self, _error: MediaError) {
        // SourceDisconnected is emitted by the source FSM via its
        // HealthSink — no duplicate emission here.
        //
        // Wake the consumer so the worker thread ticks the source and
        // advances the reconnection FSM promptly. Without this, an
        // indefinite-deadline pop() would never return.
        self.queue.wake_consumer();
    }

    fn on_eos(&self) {
        self.queue.close();
    }

    fn wake(&self) {
        self.queue.wake_consumer();
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

// ---------------------------------------------------------------------------
// BackpressureThrottle — coalesces per-frame drop events
// ---------------------------------------------------------------------------

/// Minimum interval between consecutive `BackpressureDrop` events.
const BP_THROTTLE_INTERVAL: Duration = Duration::from_secs(1);

/// Coalesces `BackpressureDrop` events to avoid per-frame storms.
///
/// - On transition into backpressure (first drop), emits immediately.
/// - During sustained backpressure, emits at most once per second
///   carrying the accumulated delta.
///
/// Thread-safe: accessed from the GStreamer streaming thread. Uses
/// `try_lock` on the event-emission path to avoid blocking the hot path.
struct BackpressureThrottle {
    inner: Mutex<BpThrottleInner>,
}

struct BpThrottleInner {
    in_backpressure: bool,
    accumulated: u64,
    last_event: Instant,
}

impl BackpressureThrottle {
    fn new() -> Self {
        Self {
            inner: Mutex::new(BpThrottleInner {
                in_backpressure: false,
                accumulated: 0,
                last_event: Instant::now(),
            }),
        }
    }

    /// Record a frame drop and possibly emit a throttled health event.
    fn record_drop(&self, health_tx: &broadcast::Sender<HealthEvent>, feed_id: FeedId) {
        let Ok(mut inner) = self.inner.try_lock() else {
            // Contention — skip event emission this frame. The drop is
            // still counted via the atomic counter in FeedSharedState.
            return;
        };

        inner.accumulated += 1;

        if !inner.in_backpressure {
            // Transition into backpressure — emit immediately.
            inner.in_backpressure = true;
            let delta = inner.accumulated;
            inner.accumulated = 0;
            inner.last_event = Instant::now();
            let _ = health_tx.send(HealthEvent::BackpressureDrop {
                feed_id,
                frames_dropped: delta,
            });
            return;
        }

        // Sustained — emit at most once per throttle interval.
        if inner.last_event.elapsed() >= BP_THROTTLE_INTERVAL {
            let delta = inner.accumulated;
            inner.accumulated = 0;
            inner.last_event = Instant::now();
            let _ = health_tx.send(HealthEvent::BackpressureDrop {
                feed_id,
                frames_dropped: delta,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// SinkWorker — decoupled per-feed sink thread
// ---------------------------------------------------------------------------

/// Maximum time to wait for the sink worker thread to finish during
/// shutdown. If the sink's `emit()` is blocked (e.g., on network I/O),
/// we detach the thread rather than hang the feed/runtime shutdown.
const SINK_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Minimum interval between consecutive `SinkBackpressure` health events.
const SINK_BP_THROTTLE_INTERVAL: Duration = Duration::from_secs(1);

/// Manages a dedicated thread that calls [`OutputSink::emit()`] asynchronously,
/// isolating the feed processing thread from slow downstream I/O.
struct SinkWorker {
    tx: std::sync::mpsc::SyncSender<SharedOutput>,
    thread: Option<std::thread::JoinHandle<Box<dyn OutputSink>>>,
    occupancy: Arc<AtomicUsize>,
}

impl SinkWorker {
    /// Spawn a sink worker thread for the given feed.
    fn spawn(
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
    fn send(
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
    fn shutdown(mut self, health_tx: &broadcast::Sender<HealthEvent>, feed_id: FeedId) -> Box<dyn OutputSink> {
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
// FeedWorker — the per-feed thread entry point
// ---------------------------------------------------------------------------

/// Why the processing loop exited.
enum ExitReason {
    /// Graceful shutdown was requested.
    Shutdown,
    /// Source ended (EOS/queue closed) — may be eligible for restart.
    SourceEnded,
    /// Non-looping file source reached end-of-stream — terminal, no restart.
    FileEos,
    /// A stage panicked — may trigger restart if policy allows.
    StagePanic,
    /// Source reported permanently stopped (reconnection budget exhausted).
    SourceStopped,
    /// The sink worker thread could not be spawned. A terminal health
    /// event has already been emitted by `processing_loop`.
    SinkSpawnFailed,
}



/// Spawns and runs the per-feed worker thread.
///
/// Returns the `Arc<FeedSharedState>` and the thread `JoinHandle`.
///
/// # Errors
///
/// Returns `RuntimeError::ThreadSpawnFailed` if the OS thread cannot be created.
pub(crate) fn spawn_feed_worker(
    feed_id: FeedId,
    config: FeedConfig,
    factory: Arc<dyn MediaIngressFactory>,
    health_tx: broadcast::Sender<HealthEvent>,
    output_tx: broadcast::Sender<SharedOutput>,
    lag_detector: Arc<LagDetector>,
) -> Result<(Arc<FeedSharedState>, std::thread::JoinHandle<()>), NvError> {
    let batch_processor_id = config.batch.as_ref().map(|b| b.processor_id());
    let shared = Arc::new(FeedSharedState::new(feed_id, batch_processor_id));
    let shared_clone = Arc::clone(&shared);
    let is_file_nonloop = config.source.is_file_nonloop();

    let handle = std::thread::Builder::new()
        .name(format!("nv-feed-{}", feed_id))
        .spawn(move || {
            let mut worker = FeedWorker {
                feed_id,
                factory,
                source_spec: config.source,
                reconnect_policy: config.reconnect,
                backpressure: config.backpressure,
                restart_policy: config.restart,
                executor: PipelineExecutor::new(
                    feed_id,
                    config.stages,
                    config.batch,
                    config.post_batch_stages,
                    config.temporal,
                    config.camera_mode,
                    config.view_state_provider,
                    config.epoch_policy,
                    config.frame_inclusion,
                    Arc::clone(&shared_clone.shutdown),
                ),
                output_sink: config.output_sink,
                ptz_provider: config.ptz_provider,
                health_tx,
                output_tx,
                shared: shared_clone,
                is_file_nonloop,
                lag_detector,
                had_external_subscribers: false,
                sink_queue_capacity: config.sink_queue_capacity,
                decode_preference: config.decode_preference,
            };
            worker.run();
        })
        .map_err(|e| {
            NvError::Runtime(RuntimeError::ThreadSpawnFailed {
                detail: e.to_string(),
            })
        })?;

    Ok((shared, handle))
}

/// Per-feed worker: owns all feed-local state and runs the processing loop.
struct FeedWorker {
    feed_id: FeedId,
    factory: Arc<dyn MediaIngressFactory>,
    source_spec: SourceSpec,
    reconnect_policy: ReconnectPolicy,
    backpressure: BackpressurePolicy,
    restart_policy: RestartPolicy,
    executor: PipelineExecutor,
    output_sink: Box<dyn OutputSink>,
    ptz_provider: Option<Arc<dyn nv_media::PtzProvider>>,
    health_tx: broadcast::Sender<HealthEvent>,
    output_tx: broadcast::Sender<SharedOutput>,
    shared: Arc<FeedSharedState>,
    /// Whether this is a non-looping file source (EOS is terminal).
    is_file_nonloop: bool,
    /// Shared lag detector — sentinel-based, runtime-global.
    lag_detector: Arc<LagDetector>,
    /// Whether external subscribers existed when we last emitted output.
    /// Used to detect the transition to no-subscribers so we can realign
    /// the lag detector even when the send is skipped.
    had_external_subscribers: bool,
    /// Per-feed output sink queue capacity (bounded channel to sink thread).
    sink_queue_capacity: usize,
    /// Decode preference — plumbed through to the media ingress factory.
    decode_preference: nv_media::DecodePreference,
}

/// Drop guard that ensures `FeedSharedState::alive` is set to `false`
/// when the worker exits — even on panic.
struct AliveGuard(Arc<FeedSharedState>);

impl Drop for AliveGuard {
    fn drop(&mut self) {
        self.0.alive.store(false, Ordering::Relaxed);
    }
}

impl FeedWorker {
    /// Main entry point — runs until shutdown or restart budget exhausted.
    fn run(&mut self) {
        // Guard ensures alive is set false even on unexpected panic.
        let _guard = AliveGuard(Arc::clone(&self.shared));
        let mut restart_count: u32 = 0;
        let mut session_start = Instant::now();

        loop {
            if self.shared.shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Reset temporal state on restart.
            if restart_count > 0 {
                self.executor.clear_temporal();
            }

            // Start stages.
            if let Err(e) = self.executor.start_stages() {
                tracing::error!(
                    feed_id = %self.feed_id,
                    error = %e,
                    "stage on_start failed"
                );
                if !self.try_restart(
                    &mut restart_count,
                    &mut session_start,
                    &ExitReason::SourceEnded,
                    format!("stage startup failed: {e}"),
                ) {
                    break;
                }
                continue;
            }

            // Mark session start (for restart_window tracking).
            session_start = Instant::now();

            // Create queue + source.
            let queue = Arc::new(FrameQueue::new(self.backpressure.clone()));
            self.shared.set_queue(Some(Arc::clone(&queue)));

            let bp_throttle = BackpressureThrottle::new();
            let sink_adapter = FeedFrameSink {
                queue: Arc::clone(&queue),
                shared: Arc::clone(&self.shared),
                health_tx: self.health_tx.clone(),
                feed_id: self.feed_id,
                bp_throttle,
            };

            let mut options = IngressOptions::new(
                self.feed_id,
                self.source_spec.clone(),
                self.reconnect_policy.clone(),
            ).with_decode_preference(self.decode_preference);
            if let Some(ref ptz) = self.ptz_provider {
                options = options.with_ptz_provider(Arc::clone(ptz));
            }
            let source = self.factory.create(options);

            let mut source = match source {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        error = %e,
                        "failed to create media source"
                    );
                    if !self.cleanup_and_try_restart(
                        &mut restart_count,
                        &mut session_start,
                        format!("source creation failed: {e}"),
                    ) {
                        break;
                    }
                    continue;
                }
            };

            // Start the source — frames begin flowing to the queue.
            match source.start(Box::new(sink_adapter)) {
                Ok(()) => {
                    tracing::info!(feed_id = %self.feed_id, "feed started");
                }
                Err(e) => {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        error = %e,
                        "source start failed"
                    );
                    if !self.cleanup_and_try_restart(
                        &mut restart_count,
                        &mut session_start,
                        format!("source start failed: {e}"),
                    ) {
                        break;
                    }
                    continue;
                }
            }

            // Processing loop.
            let exit_reason = self.processing_loop(&queue, &mut *source);

            // Cleanup.
            queue.close();
            self.shared.set_queue(None);
            let _ = source.stop();
            self.executor.stop_stages();
            if let Some(evt) = self.executor.flush_batch_rejections() {
                let _ = self.health_tx.send(evt);
            }
            if let Some(evt) = self.executor.flush_batch_timeouts() {
                let _ = self.health_tx.send(evt);
            }
            if let Some(evt) = self.executor.flush_batch_in_flight_rejections() {
                let _ = self.health_tx.send(evt);
            }

            match exit_reason {
                ExitReason::Shutdown => {
                    self.emit_feed_stopped(StopReason::UserRequested);
                    break;
                }
                ExitReason::FileEos => {
                    // Non-looping file source — terminal, no restart.
                    let _ = self.health_tx.send(HealthEvent::SourceEos {
                        feed_id: self.feed_id,
                    });
                    self.emit_feed_stopped(StopReason::EndOfStream);
                    break;
                }
                ExitReason::SourceStopped => {
                    // Source FSM reported permanently stopped.
                    // For non-looping files this is terminal EndOfStream.
                    // For everything else it means reconnect budget was
                    // exhausted — also terminal.
                    if self.is_file_nonloop {
                        let _ = self.health_tx.send(HealthEvent::SourceEos {
                            feed_id: self.feed_id,
                        });
                        self.emit_feed_stopped(StopReason::EndOfStream);
                    } else {
                        self.emit_feed_stopped(StopReason::Fatal {
                            detail: "source stopped (reconnection budget exhausted)".into(),
                        });
                    }
                    break;
                }
                ExitReason::SinkSpawnFailed => {
                    // Terminal health event already emitted by processing_loop.
                    break;
                }
                ExitReason::SourceEnded | ExitReason::StagePanic => {
                    let detail = match exit_reason {
                        ExitReason::StagePanic => format!(
                            "stage panic (trigger {:?} does not allow restart, or budget exhausted after {} restarts)",
                            self.restart_policy.restart_on, restart_count,
                        ),
                        _ => format!(
                            "restart budget exhausted after {} restarts",
                            restart_count,
                        ),
                    };
                    if !self.try_restart(
                        &mut restart_count,
                        &mut session_start,
                        &exit_reason,
                        detail,
                    ) {
                        break;
                    }
                }
            }
        }

        // `alive` is set to `false` by the AliveGuard drop.
    }

    /// Frame processing loop: pop → execute stages → emit output.
    ///
    /// The source is ticked on every frame and on queue pop timeouts. The
    /// timeout deadline is driven entirely by the source's
    /// [`TickOutcome::next_tick`] hint (e.g., reconnect backoff). When the
    /// source has no specific deadline (`next_tick: None`), the queue pop
    /// waits indefinitely — woken by incoming frames, source errors
    /// (via `wake_consumer()`), shutdown, or EOS.
    ///
    /// Returns the reason the loop exited.
    fn processing_loop(
        &mut self,
        queue: &Arc<FrameQueue>,
        source: &mut dyn MediaIngress,
    ) -> ExitReason {
        // Record session start time and sink capacity for telemetry.
        if let Ok(mut t) = self.shared.session_started_at.lock() {
            *t = Instant::now();
        }
        self.shared.sink_capacity.store(self.sink_queue_capacity, Ordering::Relaxed);
        self.shared.sink_occupancy.store(0, Ordering::Relaxed);

        // Spawn sink worker — output is decoupled from this thread.
        let sink = std::mem::replace(
            &mut self.output_sink,
            Box::new(NullSink),
        );
        let sink_worker = match SinkWorker::spawn(
            self.feed_id,
            sink,
            self.health_tx.clone(),
            self.sink_queue_capacity,
            Arc::clone(&self.shared.sink_occupancy),
        ) {
            Ok(w) => w,
            Err(e) => {
                tracing::error!(
                    feed_id = %self.feed_id,
                    error = %e,
                    "failed to spawn sink worker thread",
                );
                self.emit_feed_stopped(StopReason::Fatal {
                    detail: format!("sink worker spawn failed: {e}"),
                });
                // The real sink was already moved out; self.output_sink
                // holds NullSink. We cannot recover the original sink,
                // so exit immediately.
                return ExitReason::SinkSpawnFailed;
            }
        };

        let result = self.run_processing_loop(queue, source, &sink_worker);

        // Recover the sink from the worker thread so it can be reused
        // if the feed restarts.
        self.output_sink = sink_worker.shutdown(&self.health_tx, self.feed_id);

        result
    }

    fn run_processing_loop(
        &mut self,
        queue: &Arc<FrameQueue>,
        source: &mut dyn MediaIngress,
        sink_worker: &SinkWorker,
    ) -> ExitReason {
        // Seed the initial tick so that any deadline armed during source
        // start (e.g., liveness timeout, reconnect backoff) is honoured
        // from the very first queue pop. Without this, a source that
        // arms a deadline but emits no frames would cause the worker to
        // wait indefinitely.
        //
        // We intentionally do NOT short-circuit on SourceStatus::Stopped
        // here: the producer thread may have raced ahead and signalled
        // EOS before we enter the loop. The correct exit for that case
        // is through queue.pop() → Closed → SourceEnded (restartable),
        // not SourceStopped (terminal).
        let initial = source.tick();
        let mut next_tick_hint: Option<Duration> = initial.next_tick;

        // Sync initial decode status.
        Self::sync_decode_status(source, &self.shared);

        let mut sink_bp = SinkBpThrottle::new();

        let reason = self.run_loop_inner(queue, source, sink_worker, &mut sink_bp, &mut next_tick_hint);

        // Flush any accumulated tail from the sink backpressure
        // coalescer so the final delta is not silently lost.
        sink_bp.flush(&self.health_tx, self.feed_id);

        reason
    }

    fn run_loop_inner(
        &mut self,
        queue: &Arc<FrameQueue>,
        source: &mut dyn MediaIngress,
        sink_worker: &SinkWorker,
        sink_bp: &mut SinkBpThrottle,
        next_tick_hint: &mut Option<Duration>,
    ) -> ExitReason {
        loop {
            // Check shutdown first.
            if self.shared.shutdown.load(Ordering::Relaxed) {
                return ExitReason::Shutdown;
            }

            // Handle pause: pause the source, wait on condvar, then resume.
            if self.shared.paused.load(Ordering::Relaxed) {
                // Pause the media source to stop decoding/network I/O.
                if let Err(e) = source.pause() {
                    tracing::warn!(
                        feed_id = %self.feed_id,
                        error = %e,
                        "source pause failed (continuing paused state)"
                    );
                }
                let (lock, cvar) = &self.shared.pause_condvar;
                let mut paused = lock.lock().unwrap();
                while *paused && !self.shared.shutdown.load(Ordering::Relaxed) {
                    paused = cvar.wait(paused).unwrap();
                }
                // Resume the source when leaving paused state.
                if !self.shared.shutdown.load(Ordering::Relaxed) {
                    if let Err(e) = source.resume() {
                        tracing::warn!(
                            feed_id = %self.feed_id,
                            error = %e,
                            "source resume failed"
                        );
                    }
                }
                continue;
            }

            // Pop the next frame. The deadline is driven entirely by
            // the source's tick hint. None → wait indefinitely.
            let deadline = next_tick_hint.map(|d| Instant::now() + d);
            let pop_result = queue.pop(&self.shared.shutdown, deadline);
            let frame = match pop_result {
                PopResult::Frame(f) => f,
                PopResult::Closed => {
                    // Queue closed (EOS) or shutdown.
                    if self.shared.shutdown.load(Ordering::Relaxed) {
                        return ExitReason::Shutdown;
                    }
                    // File EOS is terminal for non-looping file sources.
                    if self.is_file_nonloop {
                        return ExitReason::FileEos;
                    }
                    return ExitReason::SourceEnded;
                }
                PopResult::Timeout | PopResult::Wake => {
                    // No frame available — tick the source to drive
                    // bus polling and reconnection.
                    let outcome = source.tick();
                    *next_tick_hint = outcome.next_tick;
                    Self::sync_decode_status(source, &self.shared);
                    if outcome.status == SourceStatus::Stopped {
                        return ExitReason::SourceStopped;
                    }
                    continue;
                }
            };

            // Run the pipeline.
            let (maybe_output, health_events) = self.executor.process_frame(&frame);

            // Check if any stage panicked.
            let had_panic = health_events
                .iter()
                .any(|e| matches!(e, HealthEvent::StagePanic { .. }));

            // Broadcast health events.
            for event in health_events {
                let _ = self.health_tx.send(event);
            }

            // Update shared metrics.
            self.shared
                .frames_processed
                .fetch_add(1, Ordering::Relaxed);
            self.shared
                .tracks_active
                .store(self.executor.track_count() as u64, Ordering::Relaxed);
            self.shared
                .view_epoch
                .store(self.executor.view_epoch(), Ordering::Relaxed);
            self.shared
                .view_stability_score
                .store(self.executor.stability_score().to_bits(), Ordering::Relaxed);
            self.shared
                .view_context_validity
                .store(self.executor.context_validity_ordinal(), Ordering::Relaxed);

            // Only emit output if the frame was not dropped.
            if let Some(output) = maybe_output {
                // Broadcast to external subscribers. receiver_count()
                // includes the internal sentinel receiver; only send when
                // there is at least one external subscriber (count > 1).
                let has_external = self.output_tx.receiver_count() > 1;

                // Arc-wrap first so broadcast and sink share the same
                // allocation — no deep clone needed.
                let shared_out: SharedOutput = Arc::new(output);

                if has_external {
                    self.had_external_subscribers = true;
                    let _ = self.output_tx.send(Arc::clone(&shared_out));
                    self.lag_detector.check_after_send(&self.health_tx);
                }

                // Enqueue for the sink worker (non-blocking).
                sink_worker.send(shared_out, sink_bp, &self.health_tx, self.feed_id);
            }

            // Tick the source after processing to drain any pending bus
            // events and update the next_tick_hint for the next pop
            // deadline.  Do NOT exit immediately when the source reports
            // Stopped — buffered frames in the queue must be drained
            // first.  The next pop() will return either another Frame
            // (continue processing) or Closed/Timeout/Wake, at which
            // point the stopped status is picked up and the loop exits
            // cleanly.
            let outcome = source.tick();
            *next_tick_hint = outcome.next_tick;
            Self::sync_decode_status(source, &self.shared);

            // Subscriber transition: external subscribers were present on
            // a prior frame but are gone now. Realign the lag detector to
            // flush any pending accumulated loss and drain stale sentinel
            // backlog.
            if self.had_external_subscribers && self.output_tx.receiver_count() <= 1 {
                self.had_external_subscribers = false;
                self.lag_detector.realign(&self.health_tx);
            }

            // If a stage panicked, exit the loop so the worker can
            // decide whether to restart based on the restart policy.
            if had_panic {
                return ExitReason::StagePanic;
            }
        }
    }

    /// Copy the latest decode status from the media source into shared state.
    ///
    /// This is a cheap check: the source returns a cached `Option` and
    /// the write only happens when the value actually changes.
    fn sync_decode_status(source: &dyn MediaIngress, shared: &FeedSharedState) {
        if let Some(status) = source.decode_status() {
            if let Ok(mut guard) = shared.decode_status.lock() {
                *guard = Some(status);
            }
        }
    }

    /// Common cleanup for source-create or source-start failures:
    /// clear queue, stop stages, flush batch rejections, then try restart.
    ///
    /// Returns `true` if restart was accepted (caller should `continue`),
    /// `false` if denied (caller should `break`).
    fn cleanup_and_try_restart(
        &mut self,
        restart_count: &mut u32,
        session_start: &mut Instant,
        detail: String,
    ) -> bool {
        self.shared.set_queue(None);
        self.executor.stop_stages();
        if let Some(evt) = self.executor.flush_batch_rejections() {
            let _ = self.health_tx.send(evt);
        }
        if let Some(evt) = self.executor.flush_batch_timeouts() {
            let _ = self.health_tx.send(evt);
        }
        if let Some(evt) = self.executor.flush_batch_in_flight_rejections() {
            let _ = self.health_tx.send(evt);
        }
        self.try_restart(restart_count, session_start, &ExitReason::SourceEnded, detail)
    }

    /// Attempt a restart. Returns `true` if the restart was accepted
    /// (caller should `continue`), `false` if restart was denied
    /// (caller should `break`).
    fn try_restart(
        &self,
        restart_count: &mut u32,
        session_start: &mut Instant,
        reason: &ExitReason,
        fatal_detail: String,
    ) -> bool {
        if !self.can_restart(*restart_count, session_start, reason) {
            self.emit_feed_stopped(StopReason::Fatal {
                detail: fatal_detail,
            });
            return false;
        }
        *restart_count = self.bump_restart(session_start, *restart_count);
        self.sleep_restart_delay();
        true
    }

    /// Check whether another restart is allowed given the exit reason and policy.
    fn can_restart(
        &self,
        current_count: u32,
        session_start: &Instant,
        reason: &ExitReason,
    ) -> bool {
        if self.restart_policy.restart_on == RestartTrigger::Never {
            return false;
        }
        if self.restart_policy.max_restarts == 0 {
            return false;
        }
        match (&self.restart_policy.restart_on, reason) {
            (RestartTrigger::SourceFailure, ExitReason::StagePanic) => return false,
            (RestartTrigger::SourceOrStagePanic, _) => {}
            (RestartTrigger::SourceFailure, _) => {}
            (RestartTrigger::Never, _) => return false,
        }
        let window: std::time::Duration = self.restart_policy.restart_window.into();
        if session_start.elapsed() >= window {
            return true;
        }
        current_count < self.restart_policy.max_restarts
    }

    /// Increment restart counter and emit FeedRestarting health event.
    fn bump_restart(
        &self,
        session_start: &mut Instant,
        current_count: u32,
    ) -> u32 {
        let window: std::time::Duration = self.restart_policy.restart_window.into();
        let new_count = if session_start.elapsed() >= window {
            1
        } else {
            current_count + 1
        };
        *session_start = Instant::now();

        let total = self.shared.restarts.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.health_tx.send(HealthEvent::FeedRestarting {
            feed_id: self.feed_id,
            restart_count: total,
        });
        new_count
    }

    fn emit_feed_stopped(&self, reason: StopReason) {
        let _ = self.health_tx.send(HealthEvent::FeedStopped {
            feed_id: self.feed_id,
            reason,
        });
    }

    fn sleep_restart_delay(&self) {
        let delay: std::time::Duration = self.restart_policy.restart_delay.into();
        let step = std::time::Duration::from_millis(50);
        let mut remaining = delay;
        while remaining > std::time::Duration::ZERO {
            if self.shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            let sleep_for = remaining.min(step);
            std::thread::sleep(sleep_for);
            remaining = remaining.saturating_sub(sleep_for);
        }
    }
}

// ---------------------------------------------------------------------------
// NullSink — placeholder while the real sink is lent to SinkWorker
// ---------------------------------------------------------------------------

/// Sink placeholder used while the real `OutputSink` is owned by the
/// `SinkWorker` thread during a processing session.
struct NullSink;

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
struct SinkBpThrottle {
    in_backpressure: bool,
    accumulated: u64,
    last_event: Instant,
}

impl SinkBpThrottle {
    fn new() -> Self {
        Self {
            in_backpressure: false,
            accumulated: 0,
            last_event: Instant::now(),
        }
    }

    fn record_drop(&mut self, health_tx: &broadcast::Sender<HealthEvent>, feed_id: FeedId) {
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
    fn flush(&mut self, health_tx: &broadcast::Sender<HealthEvent>, feed_id: FeedId) {
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
