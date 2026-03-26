use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::StageId;
use nv_perception::StageOutput;
use nv_perception::batch::BatchProcessor;
use tokio::sync::broadcast;

use super::config::BatchConfig;
use super::handle::{BatchHandle, BatchHandleInner, PendingEntry};
use super::metrics::BatchMetricsInner;

/// Default time to wait for `on_start()` to complete on the coordinator
/// thread. Overridden by [`BatchConfig::startup_timeout`] when set.
const DEFAULT_ON_START_TIMEOUT: Duration = Duration::from_secs(30);

/// Interval at which the coordinator thread checks the shutdown flag
/// during batch formation (Phase 1 and Phase 2). Keeps shutdown
/// responsive even when `max_latency` is large.
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Minimum interval between `BatchError` health events emitted by
/// the coordinator loop. Under persistent processor failure (every
/// batch returns `Err` or panics), events are coalesced to avoid
/// flooding the health channel.
const BATCH_ERROR_THROTTLE: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// BatchCoordinator
// ---------------------------------------------------------------------------

/// Coordinator state, owned by the runtime.
///
/// Manages a dedicated thread that collects submissions from feed threads,
/// forms batches, and dispatches them to a [`BatchProcessor`].
pub(crate) struct BatchCoordinator {
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    handle: BatchHandle,
}

impl std::fmt::Debug for BatchCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchCoordinator").finish_non_exhaustive()
    }
}

impl BatchCoordinator {
    /// Create and start a batch coordinator.
    pub fn start(
        mut processor: Box<dyn BatchProcessor>,
        config: BatchConfig,
        health_tx: broadcast::Sender<HealthEvent>,
    ) -> Result<Self, nv_core::error::NvError> {
        use nv_core::error::{NvError, RuntimeError};

        config.validate().map_err(NvError::Config)?;

        let queue_depth = match config.queue_capacity {
            Some(cap) => cap,
            None => config.max_batch_size.saturating_mul(4).max(4),
        };
        let (submit_tx, submit_rx) = std::sync::mpsc::sync_channel(queue_depth);
        let metrics = Arc::new(BatchMetricsInner::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let processor_id = processor.id();
        let capabilities = processor.capabilities();

        let handle = BatchHandle {
            inner: Arc::new(BatchHandleInner {
                submit_tx,
                metrics: Arc::clone(&metrics),
                processor_id,
                config: config.clone(),
                capabilities,
            }),
        };

        // on_start is called on the coordinator thread (not here) to
        // honour the thread-affinity contract documented above. A
        // sync_channel carries the result back so this call stays
        // synchronous.
        let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);

        let shutdown_clone = Arc::clone(&shutdown);
        let on_start_timeout = config.startup_timeout.unwrap_or(DEFAULT_ON_START_TIMEOUT);
        let thread = std::thread::Builder::new()
            .name(format!("nv-batch-{}", processor_id))
            .spawn(move || {
                // Panic-safe startup on the coordinator thread.
                let start_result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| processor.on_start()));
                match start_result {
                    Ok(Ok(())) => {
                        let _ = startup_tx.send(Ok(()));
                    }
                    Ok(Err(e)) => {
                        let _ =
                            startup_tx.send(Err(format!("batch processor on_start failed: {e}")));
                        return;
                    }
                    Err(_) => {
                        let _ = startup_tx.send(Err(format!(
                            "batch processor '{}' panicked in on_start",
                            processor_id
                        )));
                        return;
                    }
                }
                coordinator_loop(
                    submit_rx,
                    processor,
                    config,
                    shutdown_clone,
                    metrics,
                    health_tx,
                );
            })
            .map_err(|e| {
                NvError::Runtime(RuntimeError::ThreadSpawnFailed {
                    detail: format!("batch coordinator thread: {e}"),
                })
            })?;

        // Block until on_start completes on the coordinator thread,
        // bounded by the configured startup timeout to prevent hanging
        // on a processor that blocks indefinitely in on_start().
        match startup_rx.recv_timeout(on_start_timeout) {
            Ok(Ok(())) => {}
            Ok(Err(detail)) => {
                let _ = thread.join();
                return Err(NvError::Runtime(RuntimeError::ThreadSpawnFailed { detail }));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                shutdown.store(true, Ordering::Relaxed);

                // Attempt a short bounded join. The coordinator thread may
                // observe the shutdown flag (e.g. if on_start polls or
                // finishes shortly after the timeout). If it doesn't finish
                // promptly, we must detach — safe Rust cannot forcibly
                // terminate a blocked thread.
                //
                // LIMITATION: if on_start truly blocks forever (e.g. a
                // third-party SDK that never returns), the OS thread is
                // leaked. This is inherent to safe Rust's thread model.
                // The error detail explicitly flags this scenario so
                // operators can investigate the blocked processor.
                const STARTUP_JOIN_GRACE: Duration = Duration::from_secs(2);
                let detached = {
                    let (done_tx, done_rx) = std::sync::mpsc::channel();
                    let _ = std::thread::Builder::new()
                        .name(format!("nv-join-startup-{processor_id}"))
                        .spawn(move || {
                            let _ = thread.join();
                            let _ = done_tx.send(());
                        });
                    done_rx.recv_timeout(STARTUP_JOIN_GRACE).is_err()
                };

                let detail = if detached {
                    tracing::warn!(
                        processor = %processor_id,
                        timeout_secs = on_start_timeout.as_secs(),
                        "batch processor on_start timed out — coordinator thread detached \
                         (safe Rust cannot force-stop a blocked thread)"
                    );
                    format!(
                        "batch processor '{}' on_start did not complete within {}s; \
                         coordinator thread detached (cannot force-stop blocked on_start)",
                        processor_id,
                        on_start_timeout.as_secs(),
                    )
                } else {
                    format!(
                        "batch processor '{}' on_start did not complete within {}s",
                        processor_id,
                        on_start_timeout.as_secs(),
                    )
                };

                return Err(NvError::Runtime(RuntimeError::ThreadSpawnFailed { detail }));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(NvError::Runtime(RuntimeError::ThreadSpawnFailed {
                    detail: "batch coordinator thread exited during startup".into(),
                }));
            }
        }

        Ok(Self {
            shutdown,
            thread: Some(thread),
            handle,
        })
    }

    /// Get a clonable handle for use in `FeedConfig`.
    pub fn handle(&self) -> BatchHandle {
        self.handle.clone()
    }

    /// Signal the coordinator to shut down without joining the thread.
    ///
    /// Unblocks feed threads waiting on batch responses: once the
    /// coordinator exits, their response channels disconnect and
    /// return `CoordinatorShutdown`.
    ///
    /// Call [`shutdown()`](Self::shutdown) for a full signal-then-join.
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Shut down the coordinator: signal, bounded join, cleanup.
    ///
    /// The coordinator thread checks the shutdown flag every 100 ms in
    /// its recv loop, so termination is prompt under normal conditions.
    /// If it does not finish within `timeout`, the thread is detached
    /// with a warning and the detached join handle is returned.
    pub fn shutdown(mut self, timeout: Duration) -> Option<crate::runtime::DetachedJoin> {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            bounded_coordinator_join(thread, self.handle.processor_id(), timeout)
        } else {
            None
        }
    }
}

impl Drop for BatchCoordinator {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Thread will exit on next shutdown check. We don't join on drop
        // to avoid blocking.
    }
}

// ---------------------------------------------------------------------------
// Coordinator thread
// ---------------------------------------------------------------------------

/// Coordinator main loop.
///
/// 1. Block on first item (with periodic shutdown checks).
/// 2. Collect more items until `max_batch_size` or `max_latency` deadline.
/// 3. Dispatch the batch to the processor.
/// 4. Route results back to feed threads via per-item response channels.
///
/// # Shutdown
///
/// On exit, the receiver is dropped **before** calling `on_stop()`.
/// This is load-bearing: feed threads blocked in `submit_and_wait`
/// unblock immediately (their response channel disconnects) rather
/// than waiting for a potentially slow `on_stop()` (GPU teardown,
/// model unload) to complete.
fn coordinator_loop(
    rx: std::sync::mpsc::Receiver<PendingEntry>,
    mut processor: Box<dyn BatchProcessor>,
    config: BatchConfig,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<BatchMetricsInner>,
    health_tx: broadcast::Sender<HealthEvent>,
) {
    let mut batch: Vec<PendingEntry> = Vec::with_capacity(config.max_batch_size);
    let mut entries: Vec<nv_perception::batch::BatchEntry> =
        Vec::with_capacity(config.max_batch_size);
    let mut responses: Vec<std::sync::mpsc::SyncSender<Result<StageOutput, StageError>>> =
        Vec::with_capacity(config.max_batch_size);
    let mut in_flight_guards: Vec<Option<Arc<AtomicUsize>>> =
        Vec::with_capacity(config.max_batch_size);

    // Throttle state for BatchError health events.
    let mut last_batch_error_event: Option<Instant> = None;

    'outer: loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        batch.clear();

        // Phase 1: Wait for first item.
        let first = loop {
            if shutdown.load(Ordering::Relaxed) {
                break 'outer;
            }
            match rx.recv_timeout(SHUTDOWN_POLL_INTERVAL) {
                Ok(item) => break item,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break 'outer;
                }
            }
        };

        let formation_start = Instant::now();
        batch.push(first);

        // Phase 2: Collect more items until full, deadline, or shutdown.
        //
        // recv_timeout is capped at SHUTDOWN_POLL_INTERVAL so the
        // coordinator remains responsive to shutdown during long
        // max_latency windows.
        let deadline = Instant::now() + config.max_latency;
        while batch.len() < config.max_batch_size {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let wait = (deadline - now).min(SHUTDOWN_POLL_INTERVAL);
            match rx.recv_timeout(wait) {
                Ok(item) => batch.push(item),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        let formation_ns = formation_start.elapsed().as_nanos() as u64;

        // Phase 3: Dispatch.
        let batch_size = batch.len();
        entries.clear();
        responses.clear();
        in_flight_guards.clear();
        for pending in batch.drain(..) {
            entries.push(pending.entry);
            responses.push(pending.response_tx);
            in_flight_guards.push(pending.in_flight_guard);
        }

        let process_start = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            processor.process(&mut entries)
        }));
        let processing_ns = process_start.elapsed().as_nanos() as u64;

        metrics.record_dispatch(batch_size, formation_ns, processing_ns);

        tracing::debug!(
            processor = %processor.id(),
            batch_size,
            formation_ms = formation_ns / 1_000_000,
            processing_ms = processing_ns / 1_000_000,
            "batch dispatched"
        );

        // Phase 4: Route results.
        //
        // In-flight guards are decremented BEFORE sending the result
        // so that by the time the feed thread receives the response
        // (via the mpsc channel happens-before), the counter is
        // already updated. This prevents a spurious InFlightCapReached
        // on the next submission.
        match result {
            Ok(Ok(())) => {
                metrics.record_batch_success();
                for ((entry, tx), guard) in entries
                    .drain(..)
                    .zip(responses.drain(..))
                    .zip(in_flight_guards.drain(..))
                {
                    if let Some(g) = guard {
                        g.fetch_sub(1, Ordering::Release);
                    }
                    let output = entry.output.unwrap_or_default();
                    let _ = tx.send(Ok(output));
                }
            }
            Ok(Err(stage_err)) => {
                metrics.record_batch_error();
                tracing::error!(
                    processor = %processor.id(),
                    error = %stage_err,
                    batch_size,
                    "batch processor error"
                );
                let now = Instant::now();
                let should_emit = last_batch_error_event
                    .is_none_or(|t| now.duration_since(t) >= BATCH_ERROR_THROTTLE);
                if should_emit {
                    last_batch_error_event = Some(now);
                    let _ = health_tx.send(HealthEvent::BatchError {
                        processor_id: processor.id(),
                        batch_size: batch_size as u32,
                        error: stage_err.clone(),
                    });
                }
                for (tx, guard) in responses.drain(..).zip(in_flight_guards.drain(..)) {
                    if let Some(g) = guard {
                        g.fetch_sub(1, Ordering::Release);
                    }
                    let _ = tx.send(Err(stage_err.clone()));
                }
            }
            Err(_panic) => {
                metrics.record_batch_error();
                tracing::error!(
                    processor = %processor.id(),
                    batch_size,
                    "batch processor panicked"
                );
                let err = StageError::ProcessingFailed {
                    stage_id: processor.id(),
                    detail: "batch processor panicked".into(),
                };
                let now = Instant::now();
                let should_emit = last_batch_error_event
                    .is_none_or(|t| now.duration_since(t) >= BATCH_ERROR_THROTTLE);
                if should_emit {
                    last_batch_error_event = Some(now);
                    let _ = health_tx.send(HealthEvent::BatchError {
                        processor_id: processor.id(),
                        batch_size: batch_size as u32,
                        error: err.clone(),
                    });
                }
                for (tx, guard) in responses.drain(..).zip(in_flight_guards.drain(..)) {
                    if let Some(g) = guard {
                        g.fetch_sub(1, Ordering::Release);
                    }
                    let _ = tx.send(Err(err.clone()));
                }
            }
        }
    }

    // Drain any items remaining in the submission channel. Dropping
    // their response_tx channels unblocks feed threads immediately
    // with `CoordinatorShutdown` rather than letting them wait until
    // response_timeout fires, which would misclassify a normal
    // shutdown as a timeout.
    let drained = drain_pending(&rx);
    if drained > 0 {
        tracing::debug!(
            processor = %processor.id(),
            drained,
            "drained pending items on coordinator shutdown"
        );
    }

    // Drop the receiver *before* calling on_stop(). This is
    // load-bearing: on_stop may be slow (GPU teardown, model unload),
    // and any items submitted after the drain would block in
    // recv_timeout until the channel disconnects. Dropping rx now
    // disconnects the channel so new try_send() calls fail
    // immediately and existing response channels from drained items
    // are already dropped.
    drop(rx);

    call_on_stop(&mut *processor);
}

/// Drain all pending items from the submission channel, dropping
/// their response senders so feed threads unblock with
/// `RecvTimeoutError::Disconnected` → `CoordinatorShutdown`.
///
/// In-flight guards are decremented for each drained entry so that
/// feed threads blocked by `InFlightCapReached` are unblocked.
///
/// Returns the number of items drained.
fn drain_pending(rx: &std::sync::mpsc::Receiver<PendingEntry>) -> u64 {
    let mut count = 0u64;
    while let Ok(pe) = rx.try_recv() {
        if let Some(ref g) = pe.in_flight_guard {
            g.fetch_sub(1, Ordering::Release);
        }
        count += 1;
    }
    count
}

/// Best-effort call to `on_stop`. Errors and panics are logged but
/// do not propagate.
fn call_on_stop(processor: &mut dyn BatchProcessor) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| processor.on_stop()));
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::warn!(
                processor = %processor.id(),
                error = %e,
                "batch processor on_stop error (ignored)"
            );
        }
        Err(_) => {
            tracing::error!(
                processor = %processor.id(),
                "batch processor on_stop panicked (ignored)"
            );
        }
    }
}

/// Join the coordinator thread with a bounded timeout, mirroring the
/// [`bounded_join`](crate::runtime) pattern used for feed workers.
///
/// Returns `Some(DetachedJoin)` when the thread did not finish in time.
fn bounded_coordinator_join(
    thread: std::thread::JoinHandle<()>,
    processor_id: StageId,
    timeout: Duration,
) -> Option<crate::runtime::DetachedJoin> {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let label = format!("nv-join-batch-{processor_id}");
    let joiner = std::thread::Builder::new()
        .name(label.clone())
        .spawn(move || {
            let result = thread.join();
            let _ = done_tx.send(result);
        });
    match done_rx.recv_timeout(timeout) {
        Ok(Ok(())) => None,
        Ok(Err(_)) => {
            tracing::error!(
                processor = %processor_id,
                "batch coordinator thread panicked during join"
            );
            None
        }
        Err(_) => {
            tracing::warn!(
                processor = %processor_id,
                timeout_secs = timeout.as_secs(),
                "batch coordinator thread did not finish within timeout — detaching"
            );
            joiner.ok().map(|j| crate::runtime::DetachedJoin {
                label,
                done_rx,
                joiner: j,
            })
        }
    }
}
