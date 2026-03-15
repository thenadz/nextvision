//! Runtime, runtime builder, and runtime handle.
//!
//! The [`Runtime`] is the top-level owner constructed via [`RuntimeBuilder`].
//! After building, call [`Runtime::handle()`] to obtain a [`RuntimeHandle`] —
//! a cheaply cloneable control surface for adding/removing feeds, subscribing
//! to health and output events, and triggering shutdown.
//!
//! [`Runtime::shutdown()`] consumes the runtime, joins all worker threads,
//! and guarantees a clean stop.
//!
//! Each feed runs on a dedicated OS thread (see [`worker`](crate::worker)).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nv_core::error::{NvError, RuntimeError};
use nv_core::health::HealthEvent;
use nv_core::id::{FeedId, StageId};
use nv_media::DefaultMediaFactory;
use nv_media::ingress::MediaIngressFactory;
use nv_perception::BatchProcessor;
use tokio::sync::broadcast;

use crate::batch::{BatchConfig, BatchCoordinator, BatchHandle};
use crate::feed::{FeedConfig, FeedHandle};
use crate::output::{LagDetector, SharedOutput};
use crate::worker::{self, BroadcastHealthSink, FeedSharedState};

/// Default health broadcast channel capacity.
const DEFAULT_HEALTH_CAPACITY: usize = 256;

/// Default output broadcast channel capacity.
const DEFAULT_OUTPUT_CAPACITY: usize = 256;

/// Maximum time to wait for a feed worker thread to join during
/// remove/shutdown. If exceeded, the thread is detached.
const FEED_JOIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Builder for constructing a [`Runtime`].
///
/// The runtime uses a default media backend unless a custom
/// [`MediaIngressFactory`] is supplied via [`ingress_factory()`](Self::ingress_factory).
///
/// # Example
///
/// ```
/// use nv_runtime::Runtime;
///
/// let runtime = Runtime::builder()
///     .max_feeds(16)
///     .build()
///     .expect("failed to build runtime");
/// ```
pub struct RuntimeBuilder {
    max_feeds: usize,
    health_capacity: usize,
    output_capacity: usize,
    ingress_factory: Option<Box<dyn MediaIngressFactory>>,
}

impl RuntimeBuilder {
    /// Set the maximum number of concurrent feeds. Default: `64`.
    #[must_use]
    pub fn max_feeds(mut self, max: usize) -> Self {
        self.max_feeds = max;
        self
    }

    /// Set the health broadcast channel capacity. Default: `256`.
    #[must_use]
    pub fn health_capacity(mut self, cap: usize) -> Self {
        self.health_capacity = cap;
        self
    }

    /// Set the output broadcast channel capacity. Default: `256`.
    ///
    /// Controls how many [`OutputEnvelope`]s the aggregate output
    /// subscription channel can buffer before the ring buffer wraps.
    ///
    /// When the internal sentinel receiver detects ring-buffer wrap,
    /// the runtime emits a global [`HealthEvent::OutputLagged`] event
    /// carrying the sentinel-observed per-event delta (not cumulative).
    /// This indicates channel saturation / backpressure risk — it does
    /// **not** guarantee that any specific external subscriber lost
    /// messages.
    #[must_use]
    pub fn output_capacity(mut self, cap: usize) -> Self {
        self.output_capacity = cap;
        self
    }

    /// Set a custom `MediaIngressFactory`.
    ///
    /// By default the runtime uses the built-in media backend
    /// ([`DefaultMediaFactory`](nv_media::DefaultMediaFactory)).
    /// Replace this for testing or alternative backends.
    #[must_use]
    pub fn ingress_factory(mut self, factory: Box<dyn MediaIngressFactory>) -> Self {
        self.ingress_factory = Some(factory);
        self
    }

    /// Build the runtime.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::InvalidCapacity` if `health_capacity` or
    /// `output_capacity` is zero.
    pub fn build(self) -> Result<Runtime, NvError> {
        use nv_core::error::ConfigError;

        if self.health_capacity == 0 {
            return Err(ConfigError::InvalidCapacity {
                field: "health_capacity",
            }
            .into());
        }
        if self.output_capacity == 0 {
            return Err(ConfigError::InvalidCapacity {
                field: "output_capacity",
            }
            .into());
        }

        let (health_tx, _) = broadcast::channel(self.health_capacity);
        let (output_tx, sentinel_rx) = broadcast::channel(self.output_capacity);
        let lag_detector = Arc::new(LagDetector::new(sentinel_rx, self.output_capacity));

        let health_sink = Arc::new(BroadcastHealthSink::new(health_tx.clone()));
        let factory: Arc<dyn MediaIngressFactory> = match self.ingress_factory {
            Some(f) => Arc::from(f),
            None => Arc::new(DefaultMediaFactory::with_health_sink(health_sink as _)),
        };

        let inner = Arc::new(RuntimeInner {
            max_feeds: self.max_feeds,
            next_feed_id: AtomicU64::new(1),
            feeds: Mutex::new(HashMap::new()),
            coordinators: Mutex::new(Vec::new()),
            batch_ids: Mutex::new(HashSet::new()),
            health_tx,
            output_tx,
            lag_detector,
            shutdown: AtomicBool::new(false),
            factory,
            started_at: Instant::now(),
        });

        Ok(Runtime { inner })
    }
}

// ---------------------------------------------------------------------------
// Shared interior — Arc-wrapped, accessible from both Runtime and RuntimeHandle
// ---------------------------------------------------------------------------

struct RuntimeInner {
    max_feeds: usize,
    next_feed_id: AtomicU64,
    feeds: Mutex<HashMap<FeedId, RunningFeed>>,
    /// Batch coordinators owned by the runtime for lifecycle management.
    coordinators: Mutex<Vec<BatchCoordinator>>,
    /// Tracks all claimed batch-processor IDs (active + in-progress starts).
    /// Used as the serialization point for duplicate-ID rejection without
    /// holding the heavier `coordinators` lock during `BatchCoordinator::start`.
    batch_ids: Mutex<HashSet<StageId>>,
    health_tx: broadcast::Sender<HealthEvent>,
    output_tx: broadcast::Sender<SharedOutput>,
    lag_detector: Arc<LagDetector>,
    shutdown: AtomicBool,
    factory: Arc<dyn MediaIngressFactory>,
    /// Instant when the runtime was created. Used by `uptime()`.
    started_at: Instant,
}

/// Internal state tracked per running feed.
struct RunningFeed {
    shared: Arc<FeedSharedState>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl RuntimeInner {
    fn feed_count(&self) -> Result<usize, NvError> {
        let feeds = self
            .feeds
            .lock()
            .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?;
        Ok(feeds.len())
    }

    fn diagnostics(&self) -> Result<crate::diagnostics::RuntimeDiagnostics, NvError> {
        // Snapshot Arc refs under the feed lock, then release immediately.
        let shared_refs: Vec<Arc<FeedSharedState>> = {
            let feeds = self
                .feeds
                .lock()
                .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?;
            feeds.values().map(|entry| Arc::clone(&entry.shared)).collect()
        };

        // Build per-feed diagnostics outside the lock.
        let mut feed_diags: Vec<crate::diagnostics::FeedDiagnostics> = shared_refs
            .iter()
            .map(|shared| FeedHandle::new(Arc::clone(shared)).diagnostics())
            .collect();

        // Stable ordering by FeedId for deterministic dashboard diffing.
        feed_diags.sort_by_key(|d| d.feed_id.as_u64());

        // Batch coordinator diagnostics.
        let batches: Vec<crate::diagnostics::BatchDiagnostics> = self
            .coordinators
            .lock()
            .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?
            .iter()
            .map(|c| {
                let h = c.handle();
                crate::diagnostics::BatchDiagnostics {
                    processor_id: h.processor_id(),
                    metrics: h.metrics(),
                }
            })
            .collect();

        // Output channel lag status.
        let output_lag = self.lag_detector.status();

        Ok(crate::diagnostics::RuntimeDiagnostics {
            uptime: self.started_at.elapsed(),
            feed_count: feed_diags.len(),
            max_feeds: self.max_feeds,
            feeds: feed_diags,
            batches,
            output_lag,
        })
    }

    fn create_batch(
        &self,
        processor: Box<dyn BatchProcessor>,
        config: BatchConfig,
    ) -> Result<BatchHandle, NvError> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(NvError::Runtime(RuntimeError::ShutdownInProgress));
        }

        // Reserve the processor ID in the lightweight batch_ids set.  This
        // is the serialization point for duplicate rejection — it avoids
        // holding the heavier `coordinators` lock during the potentially
        // slow `BatchCoordinator::start` (up to ON_START_TIMEOUT = 30 s).
        let processor_id = processor.id();
        {
            let mut ids = self
                .batch_ids
                .lock()
                .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?;
            if !ids.insert(processor_id) {
                return Err(NvError::Config(
                    nv_core::error::ConfigError::DuplicateBatchProcessorId { id: processor_id },
                ));
            }
        }

        // Start coordinator without holding any lock (can block up to 30 s).
        let coordinator = match BatchCoordinator::start(processor, config, self.health_tx.clone()) {
            Ok(c) => c,
            Err(e) => {
                // Release the ID reservation on startup failure.
                if let Ok(mut ids) = self.batch_ids.lock() {
                    ids.remove(&processor_id);
                }
                return Err(e);
            }
        };
        let handle = coordinator.handle();

        {
            let mut coords = self
                .coordinators
                .lock()
                .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?;
            coords.push(coordinator);

            // Re-check shutdown *under the same lock* that shutdown_all
            // acquires to iterate coordinators.  If shutdown slipped in
            // during the (potentially 30 s) BatchCoordinator::start()
            // window, this coordinator was never signaled.  Remove it,
            // shut it down inline, and return an error so the caller
            // does not receive a handle to a dead coordinator.
            if self.shutdown.load(Ordering::Acquire) {
                if let Some(orphan) = coords.pop() {
                    drop(coords);
                    orphan.shutdown();
                }
                if let Ok(mut ids) = self.batch_ids.lock() {
                    ids.remove(&processor_id);
                }
                return Err(NvError::Runtime(RuntimeError::ShutdownInProgress));
            }
        }

        Ok(handle)
    }

    fn add_feed(&self, config: FeedConfig) -> Result<FeedHandle, NvError> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(NvError::Runtime(RuntimeError::ShutdownInProgress));
        }

        let mut feeds = self
            .feeds
            .lock()
            .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?;

        if feeds.len() >= self.max_feeds {
            return Err(NvError::Runtime(RuntimeError::FeedLimitExceeded {
                max: self.max_feeds,
            }));
        }

        let id = FeedId::new(self.next_feed_id.fetch_add(1, Ordering::Relaxed));

        let (shared, thread) = worker::spawn_feed_worker(
            id,
            config,
            Arc::clone(&self.factory),
            self.health_tx.clone(),
            self.output_tx.clone(),
            Arc::clone(&self.lag_detector),
        )?;

        let handle = FeedHandle::new(Arc::clone(&shared));

        feeds.insert(
            id,
            RunningFeed {
                shared,
                thread: Some(thread),
            },
        );

        Ok(handle)
    }

    fn remove_feed(&self, feed_id: FeedId) -> Result<(), NvError> {
        let mut feeds = self
            .feeds
            .lock()
            .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?;

        let entry = feeds
            .remove(&feed_id)
            .ok_or(NvError::Runtime(RuntimeError::FeedNotFound { feed_id }))?;

        entry.shared.request_shutdown();
        drop(feeds);

        if let Some(handle) = entry.thread {
            bounded_join(handle, feed_id, &self.health_tx);
        }
        Ok(())
    }

    fn shutdown_all(&self) -> Result<(), NvError> {
        // Release so that the Acquire re-check in create_batch() sees it.
        self.shutdown.store(true, Ordering::Release);

        // --- Phase 1: Signal everything to stop. ---
        //
        // Signal feeds first so they stop submitting new frames.
        let mut feeds = self
            .feeds
            .lock()
            .map_err(|_| NvError::Runtime(RuntimeError::RegistryPoisoned))?;

        for entry in feeds.values() {
            entry.shared.request_shutdown();
        }

        let entries: Vec<_> = feeds.drain().collect();
        drop(feeds);

        // Signal batch coordinators *before* joining feed threads.
        // Feed threads blocked in `submit_and_wait` unblock once the
        // coordinator processes remaining items and exits (disconnecting
        // response channels). Without this, feeds can hang for up to
        // `max_latency + RESPONSE_TIMEOUT_SAFETY` waiting for a batch
        // that the coordinator hasn't been told to stop.
        if let Ok(coordinators) = self.coordinators.lock() {
            for coordinator in coordinators.iter() {
                coordinator.signal_shutdown();
            }
        }

        // --- Phase 2: Join feed threads. ---
        //
        // Coordinators are shutting down concurrently, so feed threads
        // blocked on batch responses will unblock promptly.
        for (id, mut entry) in entries {
            if let Some(handle) = entry.thread.take() {
                bounded_join(handle, id, &self.health_tx);
            }
        }

        // --- Phase 3: Join batch coordinators. ---
        //
        // All feed threads are done (or detached). No new submissions.
        if let Ok(mut coordinators) = self.coordinators.lock() {
            for coordinator in coordinators.drain(..) {
                coordinator.shutdown();
            }
        }

        // Flush any pending sentinel-observed delta that was throttled
        // but never emitted.
        self.lag_detector.flush(&self.health_tx);

        Ok(())
    }
}

/// Join a feed worker thread with a bounded timeout.
///
/// If the thread does not finish within [`FEED_JOIN_TIMEOUT`], it is
/// detached (the helper thread will eventually join when the worker
/// finishes) and a `FeedStopped` health event with a timeout reason
/// is emitted.
fn bounded_join(
    handle: std::thread::JoinHandle<()>,
    feed_id: FeedId,
    health_tx: &broadcast::Sender<HealthEvent>,
) {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let _joiner = std::thread::Builder::new()
        .name(format!("nv-join-{feed_id}"))
        .spawn(move || {
            let result = handle.join();
            let _ = done_tx.send(result);
        });
    match done_rx.recv_timeout(FEED_JOIN_TIMEOUT) {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            tracing::error!(
                feed_id = %feed_id,
                "feed worker thread panicked during join",
            );
        }
        Err(_) => {
            tracing::warn!(
                feed_id = %feed_id,
                timeout_secs = FEED_JOIN_TIMEOUT.as_secs(),
                "feed worker thread did not finish within timeout — detaching",
            );
            let _ = health_tx.send(HealthEvent::FeedStopped {
                feed_id,
                reason: nv_core::health::StopReason::Fatal {
                    detail: format!(
                        "worker thread did not join within {}s — detached",
                        FEED_JOIN_TIMEOUT.as_secs()
                    ),
                },
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime — owning entry point
// ---------------------------------------------------------------------------

/// The top-level NextVision runtime.
///
/// Manages cross-feed concerns: feed registry, global limits, and shutdown.
/// Create via [`Runtime::builder()`].
///
/// Use [`handle()`](Runtime::handle) to obtain a cloneable [`RuntimeHandle`]
/// for concurrent control from multiple threads. The `Runtime` itself can
/// also be used directly for convenience.
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Runtime {
    /// Create a new [`RuntimeBuilder`].
    #[must_use]
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder {
            max_feeds: 64,
            health_capacity: DEFAULT_HEALTH_CAPACITY,
            output_capacity: DEFAULT_OUTPUT_CAPACITY,
            ingress_factory: None,
        }
    }

    /// Obtain a cloneable [`RuntimeHandle`].
    ///
    /// The handle provides the same control surface as `Runtime` but can
    /// be cloned and shared across threads.
    #[must_use]
    pub fn handle(&self) -> RuntimeHandle {
        RuntimeHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Number of currently active feeds.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::RegistryPoisoned` if the internal lock is poisoned.
    pub fn feed_count(&self) -> Result<usize, NvError> {
        self.inner.feed_count()
    }

    /// Maximum allowed concurrent feeds.
    #[must_use]
    pub fn max_feeds(&self) -> usize {
        self.inner.max_feeds
    }

    /// Elapsed time since the runtime was created.
    ///
    /// Monotonically increasing. Useful for uptime dashboards and
    /// health checks.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.inner.started_at.elapsed()
    }

    /// Get a consolidated diagnostics snapshot of the runtime and all feeds.
    ///
    /// Returns per-feed lifecycle state, metrics, queue depths, decode status,
    /// and view-system health, plus batch coordinator metrics and output
    /// channel lag status.
    ///
    /// Designed for periodic polling (1–5 s) by dashboards and health
    /// probes. Complement with [`health_subscribe()`](Self::health_subscribe)
    /// for event-driven state transitions.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::RegistryPoisoned` if an internal lock is poisoned.
    pub fn diagnostics(&self) -> Result<crate::diagnostics::RuntimeDiagnostics, NvError> {
        self.inner.diagnostics()
    }

    /// Subscribe to aggregate health events from all feeds.
    pub fn health_subscribe(&self) -> broadcast::Receiver<HealthEvent> {
        self.inner.health_tx.subscribe()
    }

    /// Subscribe to aggregate output from all feeds.
    ///
    /// Each subscriber receives an `Arc<OutputEnvelope>` for every output
    /// produced by any feed. The channel is bounded by the configured
    /// `output_capacity` (default 256). Slow subscribers will receive
    /// `RecvError::Lagged` when they fall behind.
    ///
    /// Channel saturation is monitored by an internal sentinel receiver.
    /// When the sentinel detects ring-buffer wrap, the runtime emits a
    /// global [`HealthEvent::OutputLagged`] event carrying the
    /// sentinel-observed per-event delta. This is a saturation signal,
    /// not a per-subscriber loss report. Live saturation state is also
    /// available via [`Runtime::diagnostics()`] in
    /// [`RuntimeDiagnostics::output_lag`](crate::diagnostics::RuntimeDiagnostics::output_lag).
    pub fn output_subscribe(&self) -> broadcast::Receiver<SharedOutput> {
        self.inner.output_tx.subscribe()
    }

    /// Add a new feed to the runtime.
    ///
    /// # Errors
    ///
    /// - `RuntimeError::FeedLimitExceeded` if the max feed count is reached.
    /// - `RuntimeError::ShutdownInProgress` if shutdown has been initiated.
    /// - `RuntimeError::ThreadSpawnFailed` if the OS thread cannot be created.
    pub fn add_feed(&self, config: FeedConfig) -> Result<FeedHandle, NvError> {
        self.inner.add_feed(config)
    }

    /// Create a shared batch coordinator for cross-feed inference.
    ///
    /// Returns a clonable [`BatchHandle`] that can be shared across
    /// multiple feeds via [`FeedPipeline::builder().batch(handle)`].
    ///
    /// The coordinator takes **ownership** of the processor (via `Box`).
    /// A single coordinator thread is the sole caller of all processor
    /// methods — no `Sync` bound is required. The coordinator is shut
    /// down automatically when the runtime shuts down.
    ///
    /// # Errors
    ///
    /// - `RuntimeError::ShutdownInProgress` if shutdown has been initiated.
    /// - `ConfigError::DuplicateBatchProcessorId` if a coordinator with the
    ///   same processor ID already exists.
    /// - `ConfigError::InvalidPolicy` if `config` has invalid values.
    /// - `RuntimeError::ThreadSpawnFailed` if the coordinator thread fails.
    pub fn create_batch(
        &self,
        processor: Box<dyn BatchProcessor>,
        config: BatchConfig,
    ) -> Result<BatchHandle, NvError> {
        self.inner.create_batch(processor, config)
    }

    /// Remove a feed by ID, stopping it gracefully.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::FeedNotFound` if the ID does not exist.
    pub fn remove_feed(&self, feed_id: FeedId) -> Result<(), NvError> {
        self.inner.remove_feed(feed_id)
    }

    /// Initiate graceful shutdown of all feeds.
    ///
    /// Signals all worker threads to stop, waits for them to terminate,
    /// and returns. After shutdown the runtime cannot accept new feeds.
    pub fn shutdown(self) -> Result<(), NvError> {
        self.inner.shutdown_all()
    }
}

// ---------------------------------------------------------------------------
// RuntimeHandle — cloneable control surface
// ---------------------------------------------------------------------------

/// Cloneable handle to the runtime.
///
/// Provides the same control surface as [`Runtime`] — add/remove feeds,
/// subscribe to health and output events, and trigger shutdown.
///
/// Obtain via [`Runtime::handle()`].
#[derive(Clone)]
pub struct RuntimeHandle {
    inner: Arc<RuntimeInner>,
}

impl RuntimeHandle {
    /// Number of currently active feeds.
    pub fn feed_count(&self) -> Result<usize, NvError> {
        self.inner.feed_count()
    }

    /// Maximum allowed concurrent feeds.
    #[must_use]
    pub fn max_feeds(&self) -> usize {
        self.inner.max_feeds
    }

    /// Elapsed time since the runtime was created.
    ///
    /// Monotonically increasing. Useful for uptime dashboards and
    /// health checks.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.inner.started_at.elapsed()
    }

    /// Get a consolidated diagnostics snapshot of the runtime and all feeds.
    ///
    /// See [`Runtime::diagnostics()`] for details.
    pub fn diagnostics(&self) -> Result<crate::diagnostics::RuntimeDiagnostics, NvError> {
        self.inner.diagnostics()
    }

    /// Subscribe to aggregate health events from all feeds.
    pub fn health_subscribe(&self) -> broadcast::Receiver<HealthEvent> {
        self.inner.health_tx.subscribe()
    }

    /// Subscribe to aggregate output from all feeds.
    ///
    /// Bounded by the configured `output_capacity`. Slow subscribers
    /// receive `RecvError::Lagged`. Channel saturation is reported via
    /// [`HealthEvent::OutputLagged`] (sentinel-observed, not
    /// per-subscriber loss).
    pub fn output_subscribe(&self) -> broadcast::Receiver<SharedOutput> {
        self.inner.output_tx.subscribe()
    }

    /// Add a new feed to the runtime.
    ///
    /// # Errors
    ///
    /// See [`Runtime::add_feed()`].
    pub fn add_feed(&self, config: FeedConfig) -> Result<FeedHandle, NvError> {
        self.inner.add_feed(config)
    }

    /// Create a shared batch coordinator.
    ///
    /// See [`Runtime::create_batch()`] for details.
    pub fn create_batch(
        &self,
        processor: Box<dyn BatchProcessor>,
        config: BatchConfig,
    ) -> Result<BatchHandle, NvError> {
        self.inner.create_batch(processor, config)
    }

    /// Remove a feed by ID, stopping it gracefully.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::FeedNotFound` if the ID does not exist.
    pub fn remove_feed(&self, feed_id: FeedId) -> Result<(), NvError> {
        self.inner.remove_feed(feed_id)
    }

    /// Trigger graceful shutdown of all feeds.
    ///
    /// Signals all worker threads to stop, waits for them to terminate,
    /// and returns. Unlike [`Runtime::shutdown()`], this does not consume
    /// the handle — it can be called from any clone.
    pub fn shutdown(&self) -> Result<(), NvError> {
        self.inner.shutdown_all()
    }
}

#[cfg(test)]
#[path = "runtime_tests/mod.rs"]
mod tests;
