//! Shared batch coordination infrastructure.
//!
//! This module provides the runtime-side machinery for cross-feed batch
//! inference: a [`BatchCoordinator`] that collects frames from feed
//! threads, forms bounded batches, dispatches to a user-supplied
//! [`BatchProcessor`](nv_perception::BatchProcessor), and routes results
//! back to feed-local pipeline continuations.
//!
//! # Ownership
//!
//! The coordinator takes **sole ownership** of the processor via
//! `Box<dyn BatchProcessor>`. The processor is never shared — `Sync` is
//! not required. All lifecycle methods (`on_start`, `process`, `on_stop`)
//! are called exclusively from the coordinator thread.
//!
//! # Thread model
//!
//! ```text
//! Feed-1 ──submit──┐                              ┌── response ──→ Feed-1
//! Feed-2 ──submit──┤  BatchCoordinator thread      ├── response ──→ Feed-2
//! Feed-3 ──submit──┘  (collects → dispatches)      └── response ──→ Feed-3
//!                           │
//!                    BatchProcessor::process(&mut self, &mut [BatchEntry])
//! ```
//!
//! Each feed thread submits a [`BatchEntry`] (frame + view snapshot) via
//! a bounded channel and blocks on a per-item response channel. The
//! coordinator thread collects items until `max_batch_size` or
//! `max_latency`, calls the processor, and routes results back.
//!
//! # Backpressure
//!
//! Submission uses `try_send` (non-blocking). If the coordinator queue is
//! full, the feed thread receives [`BatchSubmitError::QueueFull`] and the
//! frame is dropped with a [`HealthEvent::BatchSubmissionRejected`] event.
//! Rejection counts are coalesced per-feed with a 1-second throttle
//! window, and flushed on recovery or lifecycle boundaries.
//!
//! # Shutdown
//!
//! The coordinator checks its shutdown flag every 100 ms during both
//! idle waiting (Phase 1) and batch formation (Phase 2). During
//! runtime shutdown, coordinators are signaled *before* feed threads
//! are joined, so feed threads blocked in `submit_and_wait` unblock
//! promptly when the coordinator exits and the response channel
//! disconnects.
//!
//! ## Startup timeout
//!
//! `BatchProcessor::on_start()` is given [`ON_START_TIMEOUT`] (30 s).
//! If it exceeds that, the coordinator attempts a 2-second bounded join
//! before returning an error. If the thread is still alive after the
//! grace period it is detached (inherent safe-Rust limitation).
//!
//! ## Response timeout
//!
//! `submit_and_wait` blocks for at most `max_latency + response_timeout`
//! before returning [`BatchSubmitError::Timeout`]. The response timeout
//! defaults to 5 s ([`DEFAULT_RESPONSE_TIMEOUT`]) and can be configured
//! via [`BatchConfig::response_timeout`].
//!
//! ## Expected vs unexpected coordinator loss
//!
//! [`PipelineExecutor`](crate::executor::PipelineExecutor) distinguishes
//! expected shutdown (feed/runtime is shutting down) from unexpected
//! coordinator death by checking the feed's shutdown flag. Unexpected
//! loss emits exactly one `HealthEvent::StageError` per feed.
//!
//! # Observability
//!
//! [`BatchHandle::metrics()`] returns a [`BatchMetrics`] snapshot with:
//! batches dispatched, items processed, items rejected, processing
//! latency, formation latency, and batch-size distribution.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::StageId;
use nv_perception::batch::{BatchEntry, BatchProcessor};
use nv_perception::StageOutput;
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// BatchConfig
// ---------------------------------------------------------------------------

/// Configuration for a batch coordinator.
///
/// Controls batch formation: how many items accumulate before dispatch
/// and how long to wait for a full batch.
///
/// # Tradeoffs
///
/// - **`max_batch_size`**: Larger batches improve throughput (better GPU
///   utilization) but increase per-frame latency because each frame waits
///   for the batch to fill.
/// - **`max_latency`**: Lower values reduce worst-case latency for partial
///   batches but may dispatch smaller, less efficient batches.
///
/// Reasonable starting points for multi-feed inference:
/// - `max_batch_size`: 4–16 (depends on GPU memory / model size)
/// - `max_latency`: 20–100ms (depends on frame rate / latency tolerance)
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum items in a single batch.
    ///
    /// When this many items accumulate, the batch is dispatched
    /// immediately without waiting for `max_latency`.
    ///
    /// Must be ≥ 1.
    pub max_batch_size: usize,
    /// Maximum time to wait for a full batch before dispatching a
    /// partial one.
    ///
    /// After the first item arrives, the coordinator waits up to this
    /// duration for more items. If the batch is still not full when the
    /// deadline expires, it is dispatched as-is.
    ///
    /// Must be > 0.
    pub max_latency: Duration,
    /// Submission queue capacity.
    ///
    /// Controls how many pending items can be buffered before
    /// [`submit_and_wait`](BatchHandle::submit_and_wait) returns
    /// [`BatchSubmitError::QueueFull`].
    ///
    /// Defaults to `max_batch_size * 4` (minimum 4) when `None`.
    /// When specified, must be ≥ `max_batch_size`.
    pub queue_capacity: Option<usize>,
    /// Safety timeout added beyond `max_latency` when a feed thread waits
    /// for a batch response.
    ///
    /// The total wait is `max_latency + response_timeout`. This bounds
    /// how long a feed thread can block if the coordinator is wedged or
    /// processing is severely delayed.
    ///
    /// In practice, responses arrive within `max_latency + processing_time`.
    /// This safety margin exists only to guarantee eventual unblocking.
    ///
    /// Defaults to 5 seconds when `None`. Must be > 0 when specified.
    pub response_timeout: Option<Duration>,
}

impl BatchConfig {
    /// Create a validated batch configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidPolicy`](nv_core::error::ConfigError::InvalidPolicy)
    /// if `max_batch_size` is 0 or `max_latency` is zero.
    pub fn new(
        max_batch_size: usize,
        max_latency: Duration,
    ) -> Result<Self, nv_core::error::ConfigError> {
        if max_batch_size == 0 {
            return Err(nv_core::error::ConfigError::InvalidPolicy {
                detail: "batch max_batch_size must be >= 1".into(),
            });
        }
        if max_latency.is_zero() {
            return Err(nv_core::error::ConfigError::InvalidPolicy {
                detail: "batch max_latency must be > 0".into(),
            });
        }
        Ok(Self {
            max_batch_size,
            max_latency,
            queue_capacity: None,
            response_timeout: None,
        })
    }

    /// Set the submission queue capacity.
    ///
    /// When specified, must be ≥ `max_batch_size`. Pass `None` for the
    /// default (`max_batch_size * 4`, minimum 4).
    #[must_use]
    pub fn with_queue_capacity(mut self, capacity: Option<usize>) -> Self {
        self.queue_capacity = capacity;
        self
    }

    /// Set the response safety timeout.
    ///
    /// This is the safety margin added beyond `max_latency` when blocking
    /// for a batch response. Pass `None` for the default (5 seconds).
    /// Must be > 0 when specified.
    #[must_use]
    pub fn with_response_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.response_timeout = timeout;
        self
    }
}

impl Default for BatchConfig {
    /// Sensible defaults: batch size 4, 50 ms latency, auto queue capacity,
    /// 5-second response safety timeout.
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            max_latency: Duration::from_millis(50),
            queue_capacity: None,
            response_timeout: None,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchMetrics
// ---------------------------------------------------------------------------

/// Live metrics snapshot for a batch coordinator.
///
/// All counters are monotonically increasing. Compute rates by taking
/// deltas between snapshots.
///
/// # Counter semantics
///
/// - **`items_submitted`** — incremented when a feed thread calls
///   `submit_and_wait`, *before* the channel send. Every call
///   increments this exactly once, regardless of outcome.
/// - **`items_rejected`** — incremented when `try_send` fails
///   (`QueueFull` or `Disconnected`). Rejected items never reach
///   the coordinator.
/// - **`items_processed`** — incremented by the coordinator after
///   the batch processor returns (success or error). Each item in
///   the dispatched batch is counted.
///
/// **Invariant** (approximate under concurrent reads):
/// `items_submitted >= items_processed + items_rejected`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchMetrics {
    /// Total batches dispatched to the processor.
    pub batches_dispatched: u64,
    /// Total items successfully dispatched and processed (success or
    /// error) by the batch processor. Does not include rejected items.
    pub items_processed: u64,
    /// Total items submitted by feed threads (includes both accepted
    /// and rejected submissions).
    pub items_submitted: u64,
    /// Items rejected because the submission queue was full or the
    /// coordinator was shut down (`Disconnected`). These items never
    /// reached the coordinator thread.
    pub items_rejected: u64,
    /// Cumulative batch processing time (nanoseconds).
    pub total_processing_ns: u64,
    /// Cumulative batch formation wait time (nanoseconds).
    pub total_formation_ns: u64,
    /// Smallest batch size dispatched (0 if no batches yet).
    pub min_batch_size: u64,
    /// Largest batch size dispatched.
    pub max_batch_size_seen: u64,
    /// The configured `max_batch_size` for this coordinator.
    ///
    /// Included in the snapshot so callers can compute fill ratios
    /// without retaining a reference to the original config.
    pub configured_max_batch_size: u64,
}

impl BatchMetrics {
    /// Approximate number of items currently in-flight (submitted but not
    /// yet processed or rejected).
    ///
    /// Computed from atomic counters — may be briefly inconsistent under
    /// heavy contention, but sufficient for monitoring and dashboards.
    #[must_use]
    pub fn pending_items(&self) -> u64 {
        self.items_submitted
            .saturating_sub(self.items_processed)
            .saturating_sub(self.items_rejected)
    }

    /// Average batch size across all dispatched batches.
    ///
    /// Returns `None` if no batches have been dispatched yet.
    /// O(1), zero-allocation.
    #[must_use]
    pub fn avg_batch_size(&self) -> Option<f64> {
        if self.batches_dispatched == 0 {
            return None;
        }
        Some(self.items_processed as f64 / self.batches_dispatched as f64)
    }

    /// Average batch fill ratio: `avg_batch_size / configured_max_batch_size`.
    ///
    /// A value of `1.0` means batches are consistently full. Lower values
    /// indicate partial batches (dispatched on timeout rather than on size).
    ///
    /// Returns `None` if no batches have been dispatched yet.
    /// O(1), zero-allocation.
    #[must_use]
    pub fn avg_fill_ratio(&self) -> Option<f64> {
        let avg = self.avg_batch_size()?;
        if self.configured_max_batch_size == 0 {
            return None;
        }
        Some(avg / self.configured_max_batch_size as f64)
    }

    /// Average processing time per batch (nanoseconds).
    ///
    /// Returns `None` if no batches have been dispatched yet.
    /// O(1), zero-allocation.
    #[must_use]
    pub fn avg_processing_ns(&self) -> Option<f64> {
        if self.batches_dispatched == 0 {
            return None;
        }
        Some(self.total_processing_ns as f64 / self.batches_dispatched as f64)
    }

    /// Average formation wait time per batch (nanoseconds).
    ///
    /// Formation time is the interval from the first item arriving to
    /// the batch being dispatched. Lower values indicate faster batch
    /// filling (high submission rate or small batch size).
    ///
    /// Returns `None` if no batches have been dispatched yet.
    /// O(1), zero-allocation.
    #[must_use]
    pub fn avg_formation_ns(&self) -> Option<f64> {
        if self.batches_dispatched == 0 {
            return None;
        }
        Some(self.total_formation_ns as f64 / self.batches_dispatched as f64)
    }
}

/// Internal atomic counters for lock-free metrics recording.
struct BatchMetricsInner {
    batches_dispatched: AtomicU64,
    items_processed: AtomicU64,
    items_submitted: AtomicU64,
    items_rejected: AtomicU64,
    total_processing_ns: AtomicU64,
    total_formation_ns: AtomicU64,
    min_batch_size: AtomicU64,
    max_batch_size_seen: AtomicU64,
}

impl BatchMetricsInner {
    fn new() -> Self {
        Self {
            batches_dispatched: AtomicU64::new(0),
            items_processed: AtomicU64::new(0),
            items_submitted: AtomicU64::new(0),
            items_rejected: AtomicU64::new(0),
            total_processing_ns: AtomicU64::new(0),
            total_formation_ns: AtomicU64::new(0),
            min_batch_size: AtomicU64::new(u64::MAX),
            max_batch_size_seen: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, configured_max_batch_size: u64) -> BatchMetrics {
        let min = self.min_batch_size.load(Ordering::Relaxed);
        BatchMetrics {
            batches_dispatched: self.batches_dispatched.load(Ordering::Relaxed),
            items_processed: self.items_processed.load(Ordering::Relaxed),
            items_submitted: self.items_submitted.load(Ordering::Relaxed),
            items_rejected: self.items_rejected.load(Ordering::Relaxed),
            total_processing_ns: self.total_processing_ns.load(Ordering::Relaxed),
            total_formation_ns: self.total_formation_ns.load(Ordering::Relaxed),
            min_batch_size: if min == u64::MAX { 0 } else { min },
            max_batch_size_seen: self.max_batch_size_seen.load(Ordering::Relaxed),
            configured_max_batch_size,
        }
    }

    fn record_dispatch(&self, batch_size: usize, formation_ns: u64, processing_ns: u64) {
        let bs = batch_size as u64;
        self.batches_dispatched.fetch_add(1, Ordering::Relaxed);
        self.items_processed.fetch_add(bs, Ordering::Relaxed);
        self.total_processing_ns
            .fetch_add(processing_ns, Ordering::Relaxed);
        self.total_formation_ns
            .fetch_add(formation_ns, Ordering::Relaxed);
        self.min_batch_size.fetch_min(bs, Ordering::Relaxed);
        self.max_batch_size_seen.fetch_max(bs, Ordering::Relaxed);
    }

    fn record_submission(&self) {
        self.items_submitted.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rejection(&self) {
        self.items_rejected.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Internal submission types
// ---------------------------------------------------------------------------

/// A pending item: the entry to process plus the response channel.
struct PendingEntry {
    entry: BatchEntry,
    response_tx: std::sync::mpsc::SyncSender<Result<StageOutput, StageError>>,
}

// ---------------------------------------------------------------------------
// BatchHandle
// ---------------------------------------------------------------------------

/// Clonable handle to a batch coordinator.
///
/// Obtained from [`Runtime::create_batch`](crate::Runtime::create_batch).
/// Pass to [`FeedPipeline::builder().batch()`](crate::pipeline::FeedPipelineBuilder::batch)
/// to enable shared batch processing for a feed.
///
/// Multiple feeds can share the same handle — their frames are batched
/// together by the coordinator.
///
/// Clone is cheap (Arc bump).
#[derive(Clone)]
pub struct BatchHandle {
    pub(crate) inner: Arc<BatchHandleInner>,
}

pub(crate) struct BatchHandleInner {
    submit_tx: std::sync::mpsc::SyncSender<PendingEntry>,
    metrics: Arc<BatchMetricsInner>,
    processor_id: StageId,
    config: BatchConfig,
    capabilities: Option<nv_perception::stage::StageCapabilities>,
}

/// Default response safety timeout when [`BatchConfig::response_timeout`]
/// is `None`. See that field's documentation for semantics.
const DEFAULT_RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum time to wait for `on_start()` to complete on the coordinator
/// thread. If the processor's `on_start` blocks longer than this,
/// `BatchCoordinator::start()` returns an error and signals shutdown to
/// the coordinator thread.
const ON_START_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum time to wait for the coordinator thread to finish during
/// shutdown. The coordinator checks its shutdown flag every
/// [`SHUTDOWN_POLL_INTERVAL`], so this is generous. If exceeded, the
/// thread is detached.
const COORDINATOR_JOIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Interval at which the coordinator thread checks the shutdown flag
/// during batch formation (Phase 1 and Phase 2). Keeps shutdown
/// responsive even when `max_latency` is large.
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Minimum interval between `BatchError` health events emitted by
/// the coordinator loop. Under persistent processor failure (every
/// batch returns `Err` or panics), events are coalesced to avoid
/// flooding the health channel.
const BATCH_ERROR_THROTTLE: Duration = Duration::from_secs(1);

impl BatchHandle {
    /// The processor's stage ID.
    #[must_use]
    pub fn processor_id(&self) -> StageId {
        self.inner.processor_id
    }

    /// The processor's declared capabilities (if any).
    ///
    /// Used by the feed pipeline validation to verify that post-batch
    /// stages' input requirements are satisfied by the batch processor's
    /// declared outputs.
    #[must_use]
    pub fn capabilities(&self) -> Option<&nv_perception::stage::StageCapabilities> {
        self.inner.capabilities.as_ref()
    }

    /// Snapshot current batch metrics.
    #[must_use]
    pub fn metrics(&self) -> BatchMetrics {
        self.inner.metrics.snapshot(self.inner.config.max_batch_size as u64)
    }

    /// Submit an entry and block until the batch result is ready.
    ///
    /// Called by the feed executor on the feed's OS thread.
    ///
    /// # Per-submit channel allocation
    ///
    /// Each call allocates a `sync_channel(1)` for the response. This is
    /// intentional: the allocation is a single small heap pair, occurs at
    /// batch timescale (typically 20–100 ms), and is negligible relative to
    /// the total per-frame cost. Alternatives (pre-allocated slot pool,
    /// shared condvar) would add complexity and contention without
    /// measurable benefit at realistic batch rates.
    pub(crate) fn submit_and_wait(
        &self,
        entry: BatchEntry,
    ) -> Result<StageOutput, BatchSubmitError> {
        let (response_tx, response_rx) = std::sync::mpsc::sync_channel(1);
        self.inner.metrics.record_submission();

        // Non-blocking submit. If the queue is full, fail immediately
        // rather than blocking the feed thread.
        self.inner
            .submit_tx
            .try_send(PendingEntry { entry, response_tx })
            .map_err(|e| match e {
                std::sync::mpsc::TrySendError::Full(_) => {
                    self.inner.metrics.record_rejection();
                    BatchSubmitError::QueueFull
                }
                std::sync::mpsc::TrySendError::Disconnected(_) => {
                    self.inner.metrics.record_rejection();
                    BatchSubmitError::CoordinatorShutdown
                }
            })?;

        // Bounded wait for the response.
        let safety = self.inner.config.response_timeout
            .unwrap_or(DEFAULT_RESPONSE_TIMEOUT);
        let timeout = self.inner.config.max_latency + safety;
        match response_rx.recv_timeout(timeout) {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(stage_err)) => Err(BatchSubmitError::ProcessingFailed(stage_err)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(BatchSubmitError::Timeout),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                Err(BatchSubmitError::CoordinatorShutdown)
            }
        }
    }
}

/// Internal batch submission errors.
#[derive(Debug)]
pub(crate) enum BatchSubmitError {
    /// Submission queue is full — coordinator is overloaded.
    QueueFull,
    /// Coordinator thread has shut down.
    CoordinatorShutdown,
    /// Batch processor returned an error.
    ProcessingFailed(StageError),
    /// No response within safety timeout.
    Timeout,
}

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
        use nv_core::error::{ConfigError, NvError, RuntimeError};

        if config.max_batch_size == 0 {
            return Err(NvError::Config(ConfigError::InvalidPolicy {
                detail: "batch max_batch_size must be >= 1".into(),
            }));
        }
        if config.max_latency.is_zero() {
            return Err(NvError::Config(ConfigError::InvalidPolicy {
                detail: "batch max_latency must be > 0".into(),
            }));
        }
        if let Some(rt) = config.response_timeout {
            if rt.is_zero() {
                return Err(NvError::Config(ConfigError::InvalidPolicy {
                    detail: "batch response_timeout must be > 0".into(),
                }));
            }
        }

        let queue_depth = match config.queue_capacity {
            Some(cap) => {
                if cap < config.max_batch_size {
                    return Err(NvError::Config(ConfigError::InvalidPolicy {
                        detail: format!(
                            "batch queue_capacity ({cap}) must be >= max_batch_size ({})",
                            config.max_batch_size
                        ),
                    }));
                }
                cap
            }
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
        let thread = std::thread::Builder::new()
            .name(format!("nv-batch-{}", processor_id))
            .spawn(move || {
                // Panic-safe startup on the coordinator thread.
                let start_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    processor.on_start()
                }));
                match start_result {
                    Ok(Ok(())) => {
                        let _ = startup_tx.send(Ok(()));
                    }
                    Ok(Err(e)) => {
                        let _ = startup_tx.send(Err(
                            format!("batch processor on_start failed: {e}"),
                        ));
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
                coordinator_loop(submit_rx, processor, config, shutdown_clone, metrics, health_tx);
            })
            .map_err(|e| {
                NvError::Runtime(RuntimeError::ThreadSpawnFailed {
                    detail: format!("batch coordinator thread: {e}"),
                })
            })?;

        // Block until on_start completes on the coordinator thread,
        // bounded by ON_START_TIMEOUT to prevent hanging on a
        // processor that blocks indefinitely in on_start().
        match startup_rx.recv_timeout(ON_START_TIMEOUT) {
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
                        timeout_secs = ON_START_TIMEOUT.as_secs(),
                        "batch processor on_start timed out — coordinator thread detached \
                         (safe Rust cannot force-stop a blocked thread)"
                    );
                    format!(
                        "batch processor '{}' on_start did not complete within {}s; \
                         coordinator thread detached (cannot force-stop blocked on_start)",
                        processor_id,
                        ON_START_TIMEOUT.as_secs(),
                    )
                } else {
                    format!(
                        "batch processor '{}' on_start did not complete within {}s",
                        processor_id,
                        ON_START_TIMEOUT.as_secs(),
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
    /// If it does not finish within [`COORDINATOR_JOIN_TIMEOUT`], the
    /// thread is detached with a warning.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            bounded_coordinator_join(thread, self.handle.processor_id());
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
fn coordinator_loop(
    rx: std::sync::mpsc::Receiver<PendingEntry>,
    mut processor: Box<dyn BatchProcessor>,
    config: BatchConfig,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<BatchMetricsInner>,
    health_tx: broadcast::Sender<HealthEvent>,
) {
    let mut batch: Vec<PendingEntry> = Vec::with_capacity(config.max_batch_size);
    let mut entries: Vec<BatchEntry> = Vec::with_capacity(config.max_batch_size);
    let mut responses: Vec<std::sync::mpsc::SyncSender<Result<StageOutput, StageError>>> =
        Vec::with_capacity(config.max_batch_size);

    // Throttle state for BatchError health events.
    let mut last_batch_error_event: Option<Instant> = None;

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        batch.clear();

        // Phase 1: Wait for first item.
        let first = loop {
            if shutdown.load(Ordering::Relaxed) {
                call_on_stop(&mut *processor);
                return;
            }
            match rx.recv_timeout(SHUTDOWN_POLL_INTERVAL) {
                Ok(item) => break item,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    call_on_stop(&mut *processor);
                    return;
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
        for pending in batch.drain(..) {
            entries.push(pending.entry);
            responses.push(pending.response_tx);
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
        match result {
            Ok(Ok(())) => {
                for (entry, tx) in entries.drain(..).zip(responses.drain(..)) {
                    let output = entry.output.unwrap_or_default();
                    let _ = tx.send(Ok(output));
                }
            }
            Ok(Err(stage_err)) => {
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
                for tx in responses.drain(..) {
                    let _ = tx.send(Err(stage_err.clone()));
                }
            }
            Err(_panic) => {
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
                for tx in responses.drain(..) {
                    let _ = tx.send(Err(err.clone()));
                }
            }
        }
    }

    call_on_stop(&mut *processor);
}

/// Best-effort call to `on_stop`. Errors and panics are logged but
/// do not propagate.
fn call_on_stop(processor: &mut dyn BatchProcessor) {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| processor.on_stop()));
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
fn bounded_coordinator_join(
    thread: std::thread::JoinHandle<()>,
    processor_id: StageId,
) {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let _ = std::thread::Builder::new()
        .name(format!("nv-join-batch-{processor_id}"))
        .spawn(move || {
            let result = thread.join();
            let _ = done_tx.send(result);
        });
    match done_rx.recv_timeout(COORDINATOR_JOIN_TIMEOUT) {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            tracing::error!(
                processor = %processor_id,
                "batch coordinator thread panicked during join"
            );
        }
        Err(_) => {
            tracing::warn!(
                processor = %processor_id,
                timeout_secs = COORDINATOR_JOIN_TIMEOUT.as_secs(),
                "batch coordinator thread did not finish within timeout — detaching"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::id::{FeedId, StageId};
    use nv_perception::batch::{BatchEntry, BatchProcessor};
    use nv_perception::{DetectionSet, StageOutput};
    use nv_view::ViewSnapshot;
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;

    /// A trivial batch processor that sets each output to detections with
    /// N items where N = batch size.
    struct CountingProcessor {
        calls: Arc<AtomicU32>,
    }

    impl CountingProcessor {
        fn new() -> (Self, Arc<AtomicU32>) {
            let calls = Arc::new(AtomicU32::new(0));
            (Self { calls: Arc::clone(&calls) }, calls)
        }
    }

    impl BatchProcessor for CountingProcessor {
        fn id(&self) -> StageId {
            StageId("counting")
        }

        fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            for item in items.iter_mut() {
                item.output = Some(StageOutput::with_detections(DetectionSet::empty()));
            }
            Ok(())
        }
    }

    fn make_entry(feed_id: u64) -> BatchEntry {
        use nv_core::timestamp::MonotonicTs;
        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(feed_id),
            0,
            MonotonicTs::from_nanos(0),
            2,
            2,
            128,
        );
        let view = ViewSnapshot::new(nv_view::ViewState::fixed_initial());
        BatchEntry {
            feed_id: FeedId::new(feed_id),
            frame,
            view,
            output: None,
        }
    }

    fn start_coordinator(
        processor: Box<dyn BatchProcessor>,
        config: BatchConfig,
    ) -> (BatchCoordinator, broadcast::Receiver<HealthEvent>) {
        let (health_tx, health_rx) = broadcast::channel(64);
        let coord = BatchCoordinator::start(processor, config, health_tx).unwrap();
        (coord, health_rx)
    }

    #[test]
    fn single_item_dispatched_on_timeout() {
        let (proc, calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 8,
                max_latency: Duration::from_millis(20),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1));
        assert!(result.is_ok());
        assert!(result.unwrap().detections.is_some());
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        coord.shutdown();
    }

    #[test]
    fn full_batch_dispatched_immediately() {
        let (proc, calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_secs(10), // long timeout — should fire on size
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let mut handles = Vec::new();
        let start = Instant::now();
        for i in 0..4 {
            let h = handle.clone();
            handles.push(std::thread::spawn(move || h.submit_and_wait(make_entry(i))));
        }

        for h in handles {
            let result = h.join().unwrap();
            assert!(result.is_ok());
        }

        // Should have dispatched well before the 10s timeout.
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "full batch should dispatch immediately"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        coord.shutdown();
    }

    #[test]
    fn partial_batch_on_timeout() {
        let (proc, calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 8,
                max_latency: Duration::from_millis(30),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        // Submit 3 items (less than max_batch_size=8)
        let mut handles = Vec::new();
        for i in 0..3 {
            let h = handle.clone();
            handles.push(std::thread::spawn(move || h.submit_and_wait(make_entry(i))));
        }

        for h in handles {
            assert!(h.join().unwrap().is_ok());
        }

        // All 3 should be in one batch.
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let m = handle.metrics();
        assert_eq!(m.items_processed, 3);
        assert_eq!(m.batches_dispatched, 1);
        assert_eq!(m.min_batch_size, 3);
        assert_eq!(m.max_batch_size_seen, 3);

        coord.shutdown();
    }

    #[test]
    fn processor_error_propagated_to_all_feeds() {
        struct FailingProcessor;
        impl BatchProcessor for FailingProcessor {
            fn id(&self) -> StageId {
                StageId("fail")
            }
            fn process(&mut self, _items: &mut [BatchEntry]) -> Result<(), StageError> {
                Err(StageError::ProcessingFailed {
                    stage_id: StageId("fail"),
                    detail: "intentional".into(),
                })
            }
        }

        let (coord, mut health_rx) = start_coordinator(
            Box::new(FailingProcessor),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(20),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let mut join_handles = Vec::new();
        for i in 0..3 {
            let h = handle.clone();
            join_handles.push(std::thread::spawn(move || h.submit_and_wait(make_entry(i))));
        }

        for jh in join_handles {
            let result = jh.join().unwrap();
            assert!(
                matches!(result, Err(BatchSubmitError::ProcessingFailed(_))),
                "expected ProcessingFailed, got {result:?}"
            );
        }

        // Health event should have been emitted.
        let event = health_rx.try_recv();
        assert!(
            matches!(event, Ok(HealthEvent::BatchError { .. })),
            "expected BatchError health event"
        );

        coord.shutdown();
    }

    #[test]
    fn processor_panic_propagated_to_all_feeds() {
        struct PanicProcessor;
        impl BatchProcessor for PanicProcessor {
            fn id(&self) -> StageId {
                StageId("panicker")
            }
            fn process(&mut self, _items: &mut [BatchEntry]) -> Result<(), StageError> {
                panic!("intentional panic");
            }
        }

        let (coord, mut health_rx) = start_coordinator(
            Box::new(PanicProcessor),
            BatchConfig {
                max_batch_size: 2,
                max_latency: Duration::from_millis(20),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1));
        assert!(matches!(result, Err(BatchSubmitError::ProcessingFailed(_))));

        let event = health_rx.try_recv();
        assert!(matches!(event, Ok(HealthEvent::BatchError { .. })));

        coord.shutdown();
    }

    #[test]
    fn metrics_track_submissions_and_rejections() {
        let (proc, _calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 2,
                max_latency: Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let _ = handle.submit_and_wait(make_entry(1));
        let _ = handle.submit_and_wait(make_entry(2));

        let m = handle.metrics();
        assert_eq!(m.items_submitted, 2);
        assert!(m.batches_dispatched >= 1);

        coord.shutdown();
    }

    #[test]
    fn shutdown_while_waiting() {
        let (proc, _calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 100,
                // Long max_latency — the coordinator checks shutdown every
                // SHUTDOWN_POLL_INTERVAL (100 ms), so it should respond
                // well before this deadline.
                max_latency: Duration::from_secs(10),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        // Submit one item, then shut down — the coordinator's Phase 2
        // shutdown check should dispatch the partial batch promptly.
        let h = handle.clone();
        let jh = std::thread::spawn(move || h.submit_and_wait(make_entry(1)));

        // Give the coordinator time to receive the item.
        std::thread::sleep(Duration::from_millis(50));

        let start = Instant::now();
        coord.shutdown();
        // The response should arrive (either success from a partial batch
        // or CoordinatorShutdown). Both are acceptable.
        let _ = jh.join().unwrap();

        // Must complete promptly (within ~1s), not wait for the full
        // 10-second max_latency.
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "shutdown should complete promptly, took {:?}",
            start.elapsed(),
        );
    }

    #[test]
    fn coordinator_rejects_zero_batch_size() {
        let (proc, _) = CountingProcessor::new();
        let (health_tx, _) = broadcast::channel(16);
        let result = BatchCoordinator::start(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 0,
                max_latency: Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
            },
            health_tx,
        );
        assert!(result.is_err());
    }

    #[test]
    fn coordinator_rejects_zero_latency() {
        let (proc, _) = CountingProcessor::new();
        let (health_tx, _) = broadcast::channel(16);
        let result = BatchCoordinator::start(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::ZERO,
                queue_capacity: None,
                response_timeout: None,
            },
            health_tx,
        );
        assert!(result.is_err());
    }

    #[test]
    fn multi_feed_results_routed_correctly() {
        /// Processor that sets each output's detections to contain the
        /// feed_id as the detection count, so we can verify routing.
        struct RoutingProcessor;
        impl BatchProcessor for RoutingProcessor {
            fn id(&self) -> StageId {
                StageId("router")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                for item in items.iter_mut() {
                    // Encode the feed_id into a typed artifact so the
                    // caller can verify correct routing.
                    let mut out = StageOutput::empty();
                    out.artifacts.insert(item.feed_id.as_u64());
                    item.output = Some(out);
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(RoutingProcessor),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(50),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let mut join_handles = Vec::new();
        for feed_id in 1..=4u64 {
            let h = handle.clone();
            join_handles.push(std::thread::spawn(move || {
                let result = h.submit_and_wait(make_entry(feed_id));
                (feed_id, result)
            }));
        }

        for jh in join_handles {
            let (feed_id, result) = jh.join().unwrap();
            let output = result.expect("should succeed");
            let routed_id = output.artifacts.get::<u64>().copied();
            assert_eq!(
                routed_id,
                Some(feed_id),
                "result should be routed to the correct feed"
            );
        }

        coord.shutdown();
    }

    #[test]
    fn queue_capacity_too_small_rejected() {
        let (proc, _) = CountingProcessor::new();
        let (health_tx, _) = broadcast::channel(16);
        let result = BatchCoordinator::start(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 8,
                max_latency: Duration::from_millis(10),
                queue_capacity: Some(4), // less than max_batch_size=8
                response_timeout: None,
            },
            health_tx,
        );
        assert!(result.is_err(), "queue_capacity < max_batch_size should fail");
    }

    #[test]
    fn explicit_queue_capacity_accepted() {
        let (proc, _calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 2,
                max_latency: Duration::from_millis(20),
                queue_capacity: Some(32),
                response_timeout: None,
            },
        );
        let handle = coord.handle();
        let _ = handle.submit_and_wait(make_entry(1));
        coord.shutdown();
    }

    #[test]
    fn disconnected_submit_increments_rejected() {
        let (proc, _calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(20),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        // Shut down the coordinator so the channel becomes disconnected.
        coord.shutdown();
        // Give the thread time to finish.
        std::thread::sleep(Duration::from_millis(100));

        // Submit after coordinator is gone — should get CoordinatorShutdown.
        let result = handle.submit_and_wait(make_entry(1));
        assert!(matches!(result, Err(BatchSubmitError::CoordinatorShutdown)));

        // items_submitted was incremented, and items_rejected should also
        // be incremented so pending_items() stays accurate.
        let m = handle.metrics();
        assert_eq!(m.items_submitted, 1);
        assert_eq!(m.items_rejected, 1);
        assert_eq!(m.pending_items(), 0, "pending should be 0 after disconnect rejection");
    }

    #[test]
    fn avg_batch_size_returns_none_when_no_batches() {
        let m = BatchMetrics {
            batches_dispatched: 0,
            items_processed: 0,
            items_submitted: 0,
            items_rejected: 0,
            total_processing_ns: 0,
            total_formation_ns: 0,
            min_batch_size: 0,
            max_batch_size_seen: 0,
            configured_max_batch_size: 8,
        };
        assert!(m.avg_batch_size().is_none());
        assert!(m.avg_fill_ratio().is_none());
        assert!(m.avg_processing_ns().is_none());
        assert!(m.avg_formation_ns().is_none());
    }

    #[test]
    fn avg_batch_size_correct() {
        let m = BatchMetrics {
            batches_dispatched: 4,
            items_processed: 12,
            items_submitted: 12,
            items_rejected: 0,
            total_processing_ns: 400_000,
            total_formation_ns: 200_000,
            min_batch_size: 2,
            max_batch_size_seen: 4,
            configured_max_batch_size: 8,
        };
        let avg = m.avg_batch_size().unwrap();
        assert!((avg - 3.0).abs() < f64::EPSILON, "expected 12/4 = 3.0, got {avg}");
    }

    #[test]
    fn avg_fill_ratio_correct() {
        let m = BatchMetrics {
            batches_dispatched: 2,
            items_processed: 8,
            items_submitted: 8,
            items_rejected: 0,
            total_processing_ns: 0,
            total_formation_ns: 0,
            min_batch_size: 4,
            max_batch_size_seen: 4,
            configured_max_batch_size: 8,
        };
        let ratio = m.avg_fill_ratio().unwrap();
        assert!((ratio - 0.5).abs() < f64::EPSILON, "expected 4/8 = 0.5, got {ratio}");
    }

    #[test]
    fn avg_fill_ratio_full_batches() {
        let m = BatchMetrics {
            batches_dispatched: 3,
            items_processed: 24,
            items_submitted: 24,
            items_rejected: 0,
            total_processing_ns: 0,
            total_formation_ns: 0,
            min_batch_size: 8,
            max_batch_size_seen: 8,
            configured_max_batch_size: 8,
        };
        let ratio = m.avg_fill_ratio().unwrap();
        assert!((ratio - 1.0).abs() < f64::EPSILON, "expected 1.0 for full batches, got {ratio}");
    }

    #[test]
    fn avg_latency_helpers_correct() {
        let m = BatchMetrics {
            batches_dispatched: 5,
            items_processed: 20,
            items_submitted: 20,
            items_rejected: 0,
            total_processing_ns: 500_000,
            total_formation_ns: 250_000,
            min_batch_size: 4,
            max_batch_size_seen: 4,
            configured_max_batch_size: 8,
        };
        let avg_proc = m.avg_processing_ns().unwrap();
        assert!((avg_proc - 100_000.0).abs() < f64::EPSILON);
        let avg_form = m.avg_formation_ns().unwrap();
        assert!((avg_form - 50_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn configured_max_batch_size_in_metrics() {
        let (proc, _calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 16,
                max_latency: Duration::from_millis(20),
                queue_capacity: None,
                response_timeout: None,
            },
        );
        let m = coord.handle().metrics();
        assert_eq!(m.configured_max_batch_size, 16);
        coord.shutdown();
    }

    #[test]
    fn batch_config_new_validates() {
        // Valid config.
        let cfg = BatchConfig::new(4, Duration::from_millis(50));
        assert!(cfg.is_ok());
        let cfg = cfg.unwrap();
        assert_eq!(cfg.max_batch_size, 4);
        assert_eq!(cfg.max_latency, Duration::from_millis(50));
        assert!(cfg.queue_capacity.is_none());

        // Zero batch size.
        assert!(BatchConfig::new(0, Duration::from_millis(50)).is_err());

        // Zero latency.
        assert!(BatchConfig::new(4, Duration::ZERO).is_err());
    }

    #[test]
    fn batch_config_default_is_valid() {
        let cfg = BatchConfig::default();
        assert!(cfg.max_batch_size >= 1);
        assert!(!cfg.max_latency.is_zero());
    }

    #[test]
    fn batch_config_with_queue_capacity() {
        let cfg = BatchConfig::new(4, Duration::from_millis(50))
            .unwrap()
            .with_queue_capacity(Some(32));
        assert_eq!(cfg.queue_capacity, Some(32));
    }

    #[test]
    fn signal_shutdown_unblocks_coordinator() {
        let (proc, _calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 100,
                max_latency: Duration::from_secs(10),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let h = handle.clone();
        let jh = std::thread::spawn(move || h.submit_and_wait(make_entry(1)));

        std::thread::sleep(Duration::from_millis(50));

        // Signal shutdown without joining — the coordinator should still
        // process the pending item and exit.
        let start = Instant::now();
        coord.signal_shutdown();

        let result = jh.join().unwrap();
        // Either processed successfully or got CoordinatorShutdown.
        assert!(result.is_ok() || matches!(result, Err(BatchSubmitError::CoordinatorShutdown)));

        // Must not wait for the full 10s max_latency.
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "signal_shutdown should unblock promptly, took {:?}",
            start.elapsed(),
        );
    }

    #[test]
    fn on_start_failure_returns_error() {
        struct FailStart;
        impl BatchProcessor for FailStart {
            fn id(&self) -> StageId {
                StageId("fail_start")
            }
            fn on_start(&mut self) -> Result<(), StageError> {
                Err(StageError::ModelLoadFailed {
                    stage_id: StageId("fail_start"),
                    detail: "test".into(),
                })
            }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), StageError> {
                unreachable!()
            }
        }

        let (health_tx, _) = broadcast::channel(16);
        let result = BatchCoordinator::start(
            Box::new(FailStart),
            BatchConfig::default(),
            health_tx,
        );
        assert!(result.is_err(), "on_start failure should propagate");
    }

    #[test]
    fn non_detector_output_routed_correctly() {
        use nv_core::timestamp::MonotonicTs;
        use nv_perception::scene::{SceneFeature, SceneFeatureValue};

        /// Batch processor that produces scene features instead of
        /// detections — validates output model flexibility.
        struct SceneClassifier;
        impl BatchProcessor for SceneClassifier {
            fn id(&self) -> StageId {
                StageId("scene_clf")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::with_scene_features(vec![
                        SceneFeature {
                            name: "weather",
                            value: SceneFeatureValue::Scalar(0.9),
                            confidence: Some(0.95),
                            ts: MonotonicTs::from_nanos(0),
                        },
                    ]));
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(SceneClassifier),
            BatchConfig {
                max_batch_size: 2,
                max_latency: Duration::from_millis(20),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1));
        let output = result.expect("scene classifier should succeed");
        assert_eq!(output.scene_features.len(), 1);
        assert_eq!(output.scene_features[0].name, "weather");

        coord.shutdown();
    }

    #[test]
    fn slow_on_start_completes_successfully() {
        use std::sync::Barrier;

        /// Processor whose on_start blocks until signaled, simulating a
        /// slow-but-completing startup (e.g. loading a model from disk).
        struct SlowStart {
            barrier: Arc<Barrier>,
        }
        impl BatchProcessor for SlowStart {
            fn id(&self) -> StageId {
                StageId("slow_start")
            }
            fn on_start(&mut self) -> Result<(), StageError> {
                self.barrier.wait();
                Ok(())
            }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), StageError> {
                unreachable!()
            }
        }

        let barrier = Arc::new(Barrier::new(2));
        let (health_tx, _) = broadcast::channel(16);

        // Release the barrier from a helper thread after 100 ms so
        // on_start completes well within ON_START_TIMEOUT (30 s).
        // This validates that a blocking-but-eventually-completing
        // on_start succeeds and the coordinator cleans up normally.
        //
        // NOTE: the actual 30 s timeout path cannot be tested in a
        // fast unit test since ON_START_TIMEOUT is a module constant.
        let b = Arc::clone(&barrier);
        let _helper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            b.wait();
        });

        let result = BatchCoordinator::start(
            Box::new(SlowStart { barrier }),
            BatchConfig::default(),
            health_tx,
        );
        assert!(result.is_ok(), "slow-but-completing on_start should succeed");
        result.unwrap().shutdown();
    }

    #[test]
    fn on_start_failure_propagates_error_with_processor_id() {
        struct FailStart;
        impl BatchProcessor for FailStart {
            fn id(&self) -> StageId {
                StageId("gpu_init")
            }
            fn on_start(&mut self) -> Result<(), StageError> {
                Err(StageError::ModelLoadFailed {
                    stage_id: StageId("gpu_init"),
                    detail: "CUDA OOM".into(),
                })
            }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), StageError> {
                unreachable!()
            }
        }

        let (health_tx, _) = broadcast::channel(16);
        let result = BatchCoordinator::start(
            Box::new(FailStart),
            BatchConfig::default(),
            health_tx,
        );
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("gpu_init"),
            "error should surface the processor id, got: {msg}"
        );
    }

    #[test]
    fn response_timeout_config_defaults_to_5s() {
        let cfg = BatchConfig::default();
        assert!(cfg.response_timeout.is_none());
        // Effective timeout = max_latency + 5s (DEFAULT_RESPONSE_TIMEOUT)
    }

    #[test]
    fn response_timeout_config_custom_value() {
        let cfg = BatchConfig::new(4, Duration::from_millis(50))
            .unwrap()
            .with_response_timeout(Some(Duration::from_secs(2)));
        assert_eq!(cfg.response_timeout, Some(Duration::from_secs(2)));
    }

    #[test]
    fn response_timeout_zero_rejected() {
        let (proc, _) = CountingProcessor::new();
        let (health_tx, _) = broadcast::channel(16);
        let result = BatchCoordinator::start(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: Some(Duration::ZERO),
            },
            health_tx,
        );
        assert!(result.is_err(), "zero response_timeout should be rejected");
    }

    #[test]
    fn custom_response_timeout_applied() {
        // With a very short custom response timeout + a slow processor,
        // submissions should time out quickly.
        struct SlowProcessor;
        impl BatchProcessor for SlowProcessor {
            fn id(&self) -> StageId {
                StageId("slow")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                // Sleep longer than the configured response_timeout.
                std::thread::sleep(Duration::from_secs(3));
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (health_tx, _) = broadcast::channel(16);
        let coord = BatchCoordinator::start(
            Box::new(SlowProcessor),
            BatchConfig {
                max_batch_size: 1,
                max_latency: Duration::from_millis(10),
                queue_capacity: None,
                // Very short response timeout — should trigger before
                // the 3s processing completes.
                response_timeout: Some(Duration::from_millis(200)),
            },
            health_tx,
        ).unwrap();

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1));
        assert!(
            matches!(result, Err(BatchSubmitError::Timeout)),
            "expected Timeout with short response_timeout, got: {result:?}"
        );

        coord.shutdown();
    }

    #[test]
    fn batch_error_throttle_coalesces_events() {
        struct AlwaysFails;
        impl BatchProcessor for AlwaysFails {
            fn id(&self) -> StageId {
                StageId("always_fails")
            }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), StageError> {
                Err(StageError::ProcessingFailed {
                    stage_id: StageId("always_fails"),
                    detail: "persistent failure".into(),
                })
            }
        }

        let (coord, mut health_rx) = start_coordinator(
            Box::new(AlwaysFails),
            BatchConfig {
                max_batch_size: 1,
                max_latency: Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
            },
        );

        let handle = coord.handle();

        // Submit rapidly many times. Due to BATCH_ERROR_THROTTLE (1s),
        // only the first should produce a BatchError health event.
        for i in 0..5 {
            let _ = handle.submit_and_wait(make_entry(i));
        }

        // Count how many BatchError events were emitted.
        let mut event_count = 0;
        while let Ok(HealthEvent::BatchError { .. }) = health_rx.try_recv() {
            event_count += 1;
        }

        // Should be exactly 1 due to 1-second throttle (all submissions
        // happen in < 1s).
        assert_eq!(
            event_count, 1,
            "BatchError should be throttled to 1 per second, got {event_count}"
        );

        coord.shutdown();
    }
}
