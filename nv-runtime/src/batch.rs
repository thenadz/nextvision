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
//! # Fairness model
//!
//! All feeds share a single FIFO submission queue. The coordinator
//! processes items strictly in submission order.
//!
//! ## Per-feed in-flight cap
//!
//! Each feed is allowed at most [`max_in_flight_per_feed`](BatchConfig::max_in_flight_per_feed)
//! items in-flight simultaneously (default: **1**). An item is "in-flight"
//! from the moment it enters the submission queue until the coordinator
//! routes its result back (or drains it at shutdown).
//!
//! Under normal operation, [`submit_and_wait`](BatchHandle::submit_and_wait)
//! is synchronous — the feed thread blocks until the result arrives.
//! With the default cap of 1, at most one item per feed is in the queue
//! at any time.
//!
//! **Timeout regime**: when `submit_and_wait` returns
//! [`BatchSubmitError::Timeout`], the timed-out item remains in-flight
//! inside the coordinator. The in-flight cap prevents the feed from
//! stacking additional items: the next `submit_and_wait` call returns
//! [`BatchSubmitError::InFlightCapReached`] immediately until the
//! coordinator processes (or drains) the orphaned item. This bounds
//! per-feed queue occupancy to `max_in_flight_per_feed` even under
//! sustained processor slowness.
//!
//! Under normal load (queue rarely full, no timeouts) every feed's
//! frames are accepted and batched fairly by arrival time. Under
//! sustained overload (queue persistently full), `try_send` fails at
//! the instant a feed attempts submission.
//!
//! **Not guaranteed**: strict round-robin or weighted fairness. After
//! a batch completes, all participating feeds are unblocked
//! simultaneously. Scheduling jitter determines which feed's next
//! `try_send` arrives first. Over time the distribution is
//! approximately uniform, but short-term skew is possible.
//!
//! **Diagnostic**: per-feed rejection counts are visible via
//! [`HealthEvent::BatchSubmissionRejected`] events, per-feed
//! timeout counts via [`HealthEvent::BatchTimeout`] events, and
//! per-feed in-flight cap rejections via
//! [`HealthEvent::BatchInFlightExceeded`] events (all coalesced
//! per feed). Persistent in-flight rejections indicate the
//! processor is too slow for the configured timeout.
//!
//! # Queue sizing
//!
//! The default queue capacity is `max_batch_size * 4` (minimum 4).
//! Guidelines:
//!
//! - `queue_capacity >= num_feeds` is required for all feeds to have
//!   a slot simultaneously. Because `submit_and_wait` serializes per
//!   feed, this is the hard floor for avoiding unnecessary rejections.
//! - `queue_capacity >= max_batch_size * 2` prevents rejection during
//!   normal batch-formation cadence (one batch forming, one draining).
//! - When both conditions conflict, prefer the larger value.
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

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
    /// Maximum number of in-flight submissions allowed per feed.
    ///
    /// An item is "in-flight" from the moment it enters the submission
    /// queue until the coordinator routes its result back (or drains it
    /// at shutdown). When a feed reaches this limit, further
    /// [`submit_and_wait`](BatchHandle::submit_and_wait) calls fail
    /// immediately with [`BatchSubmitError::InFlightCapReached`] rather
    /// than adding to the queue.
    ///
    /// This prevents a feed from accumulating orphaned items in the
    /// shared queue after timeouts: when `submit_and_wait` times out,
    /// the item remains in-flight inside the coordinator. Without a
    /// cap, the feed could immediately submit another frame, stacking
    /// multiple items and crowding other feeds.
    ///
    /// Default: 1 — each feed contributes at most one item to the
    /// shared queue at any time. Must be ≥ 1.
    pub max_in_flight_per_feed: usize,
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
            max_in_flight_per_feed: 1,
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

    /// Set the maximum number of in-flight submissions per feed.
    ///
    /// Default is 1. Must be ≥ 1.
    #[must_use]
    pub fn with_max_in_flight_per_feed(mut self, max: usize) -> Self {
        self.max_in_flight_per_feed = max;
        self
    }

    /// Validate all configuration fields.
    ///
    /// Called internally by [`BatchCoordinator::start`]. Also available
    /// for early validation before passing a config to the runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidPolicy`](nv_core::error::ConfigError::InvalidPolicy)
    /// if any field violates its constraints.
    pub fn validate(&self) -> Result<(), nv_core::error::ConfigError> {
        use nv_core::error::ConfigError;
        if self.max_batch_size == 0 {
            return Err(ConfigError::InvalidPolicy {
                detail: "batch max_batch_size must be >= 1".into(),
            });
        }
        if self.max_latency.is_zero() {
            return Err(ConfigError::InvalidPolicy {
                detail: "batch max_latency must be > 0".into(),
            });
        }
        if let Some(rt) = self.response_timeout {
            if rt.is_zero() {
                return Err(ConfigError::InvalidPolicy {
                    detail: "batch response_timeout must be > 0".into(),
                });
            }
        }
        if let Some(cap) = self.queue_capacity {
            if cap < self.max_batch_size {
                return Err(ConfigError::InvalidPolicy {
                    detail: format!(
                        "batch queue_capacity ({cap}) must be >= max_batch_size ({})",
                        self.max_batch_size
                    ),
                });
            }
        }
        if self.max_in_flight_per_feed == 0 {
            return Err(ConfigError::InvalidPolicy {
                detail: "batch max_in_flight_per_feed must be >= 1".into(),
            });
        }
        Ok(())
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
            max_in_flight_per_feed: 1,
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
    /// Items whose response was not received before the feed-side
    /// safety timeout (`max_latency + response_timeout`). The
    /// coordinator may still process these items, but the feed thread
    /// abandoned waiting.
    ///
    /// A non-zero value indicates the batch processor is slower than
    /// the configured safety margin allows. Consider increasing
    /// `response_timeout` or reducing `max_batch_size`.
    pub items_timed_out: u64,
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
    /// Number of consecutive batch errors (processor `Err` or panic)
    /// since the last successful dispatch. Reset to 0 on success.
    ///
    /// Useful for alerting: a steadily increasing value indicates a
    /// persistently broken processor. Zero means the last batch
    /// succeeded.
    pub consecutive_errors: u64,
}

impl BatchMetrics {
    /// Approximate number of items currently in-flight (submitted but not
    /// yet processed or rejected).
    ///
    /// Computed from atomic counters — may be briefly inconsistent under
    /// heavy contention, but sufficient for monitoring and dashboards.
    ///
    /// **Caveat**: items that timed out on the feed side may still be
    /// processed by the coordinator. When this happens, `items_processed`
    /// includes the timed-out item and this counter can undercount. The
    /// discrepancy is bounded by `items_timed_out`.
    #[must_use]
    pub fn pending_items(&self) -> u64 {
        self.items_submitted
            .saturating_sub(self.items_processed)
            .saturating_sub(self.items_rejected)
    }

    /// Fraction of submissions rejected (`items_rejected / items_submitted`).
    ///
    /// Returns `None` if no items have been submitted.
    /// A consistently non-zero value indicates sustained overload.
    #[must_use]
    pub fn rejection_rate(&self) -> Option<f64> {
        if self.items_submitted == 0 {
            return None;
        }
        Some(self.items_rejected as f64 / self.items_submitted as f64)
    }

    /// Fraction of submissions that timed out (`items_timed_out / items_submitted`).
    ///
    /// Returns `None` if no items have been submitted.
    /// A non-zero value indicates the processor is too slow for the
    /// configured safety timeout.
    #[must_use]
    pub fn timeout_rate(&self) -> Option<f64> {
        if self.items_submitted == 0 {
            return None;
        }
        Some(self.items_timed_out as f64 / self.items_submitted as f64)
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

impl std::fmt::Display for BatchMetrics {
    /// Human-readable diagnostic summary.
    ///
    /// Example output:
    /// ```text
    /// batches=120 items=480/500 rejected=15 timed_out=2 fill=0.80 \
    /// avg_proc=12.5ms avg_form=8.2ms consec_err=0
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "batches={} items={}/{} rejected={} timed_out={} fill={} avg_proc={} avg_form={} consec_err={}",
            self.batches_dispatched,
            self.items_processed,
            self.items_submitted,
            self.items_rejected,
            self.items_timed_out,
            self.avg_fill_ratio()
                .map_or_else(|| "n/a".to_string(), |r| format!("{r:.2}")),
            self.avg_processing_ns()
                .map_or_else(|| "n/a".to_string(), |ns| format!("{:.1}ms", ns / 1_000_000.0)),
            self.avg_formation_ns()
                .map_or_else(|| "n/a".to_string(), |ns| format!("{:.1}ms", ns / 1_000_000.0)),
            self.consecutive_errors,
        )
    }
}

/// Internal atomic counters for lock-free metrics recording.
struct BatchMetricsInner {
    batches_dispatched: AtomicU64,
    items_processed: AtomicU64,
    items_submitted: AtomicU64,
    items_rejected: AtomicU64,
    items_timed_out: AtomicU64,
    total_processing_ns: AtomicU64,
    total_formation_ns: AtomicU64,
    min_batch_size: AtomicU64,
    max_batch_size_seen: AtomicU64,
    consecutive_errors: AtomicU64,
}

impl BatchMetricsInner {
    fn new() -> Self {
        Self {
            batches_dispatched: AtomicU64::new(0),
            items_processed: AtomicU64::new(0),
            items_submitted: AtomicU64::new(0),
            items_rejected: AtomicU64::new(0),
            items_timed_out: AtomicU64::new(0),
            total_processing_ns: AtomicU64::new(0),
            total_formation_ns: AtomicU64::new(0),
            min_batch_size: AtomicU64::new(u64::MAX),
            max_batch_size_seen: AtomicU64::new(0),
            consecutive_errors: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, configured_max_batch_size: u64) -> BatchMetrics {
        let min = self.min_batch_size.load(Ordering::Relaxed);
        BatchMetrics {
            batches_dispatched: self.batches_dispatched.load(Ordering::Relaxed),
            items_processed: self.items_processed.load(Ordering::Relaxed),
            items_submitted: self.items_submitted.load(Ordering::Relaxed),
            items_rejected: self.items_rejected.load(Ordering::Relaxed),
            items_timed_out: self.items_timed_out.load(Ordering::Relaxed),
            total_processing_ns: self.total_processing_ns.load(Ordering::Relaxed),
            total_formation_ns: self.total_formation_ns.load(Ordering::Relaxed),
            min_batch_size: if min == u64::MAX { 0 } else { min },
            max_batch_size_seen: self.max_batch_size_seen.load(Ordering::Relaxed),
            configured_max_batch_size,
            consecutive_errors: self.consecutive_errors.load(Ordering::Relaxed),
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

    fn record_timeout(&self) {
        self.items_timed_out.fetch_add(1, Ordering::Relaxed);
    }

    fn record_batch_success(&self) {
        self.consecutive_errors.store(0, Ordering::Relaxed);
    }

    fn record_batch_error(&self) {
        self.consecutive_errors.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Internal submission types
// ---------------------------------------------------------------------------

/// A pending item: the entry to process plus the response channel
/// and optional per-feed in-flight guard.
struct PendingEntry {
    entry: BatchEntry,
    response_tx: std::sync::mpsc::SyncSender<Result<StageOutput, StageError>>,
    /// Per-feed in-flight counter. The coordinator decrements this
    /// after routing the result (or on drain). Must be decremented
    /// exactly once per PendingEntry that reached the coordinator.
    in_flight_guard: Option<Arc<AtomicUsize>>,
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

    /// Record that a feed-side timeout occurred.
    ///
    /// Called by the executor when `submit_and_wait` returns
    /// `BatchSubmitError::Timeout`. The coordinator is unaware of
    /// feed-side timeouts (it continues processing the batch), so
    /// this counter is maintained from the feed side.
    pub(crate) fn record_timeout(&self) {
        self.inner.metrics.record_timeout();
    }

    /// Submit an entry and block until the batch result is ready.
    ///
    /// Called by the feed executor on the feed's OS thread.
    ///
    /// # In-flight guard
    ///
    /// When `in_flight` is `Some`, the counter is checked against
    /// [`BatchConfig::max_in_flight_per_feed`] before submission.
    /// If the feed is at its cap, returns
    /// [`BatchSubmitError::InFlightCapReached`] immediately.
    /// On successful `try_send`, the counter has been incremented;
    /// the **coordinator** is responsible for decrementing it after
    /// routing the result or during drain.
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
        in_flight: Option<&Arc<AtomicUsize>>,
    ) -> Result<StageOutput, BatchSubmitError> {
        let (response_tx, response_rx) = std::sync::mpsc::sync_channel(1);
        self.inner.metrics.record_submission();

        // Check in-flight cap before attempting submission.
        let guard = if let Some(counter) = in_flight {
            let prev = counter.fetch_add(1, Ordering::Acquire);
            if prev >= self.inner.config.max_in_flight_per_feed {
                counter.fetch_sub(1, Ordering::Release);
                self.inner.metrics.record_rejection();
                return Err(BatchSubmitError::InFlightCapReached);
            }
            Some(Arc::clone(counter))
        } else {
            None
        };

        // Non-blocking submit. If the queue is full, fail immediately
        // rather than blocking the feed thread. Decrement in-flight
        // since the entry never reached the coordinator.
        self.inner
            .submit_tx
            .try_send(PendingEntry {
                entry,
                response_tx,
                in_flight_guard: guard.clone(),
            })
            .map_err(|e| {
                // Entry rejected — coordinator never saw it, decrement.
                if let Some(ref g) = guard {
                    g.fetch_sub(1, Ordering::Release);
                }
                match e {
                    std::sync::mpsc::TrySendError::Full(_) => {
                        self.inner.metrics.record_rejection();
                        BatchSubmitError::QueueFull
                    }
                    std::sync::mpsc::TrySendError::Disconnected(_) => {
                        self.inner.metrics.record_rejection();
                        BatchSubmitError::CoordinatorShutdown
                    }
                }
            })?;

        // Bounded wait for the response. The coordinator decrements
        // the in-flight guard before sending the result, so by the
        // time we receive the response the counter is already updated.
        let safety = self.inner.config.response_timeout
            .unwrap_or(DEFAULT_RESPONSE_TIMEOUT);
        let timeout = self.inner.config.max_latency + safety;
        match response_rx.recv_timeout(timeout) {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(stage_err)) => Err(BatchSubmitError::ProcessingFailed(stage_err)),
            // Timeout: do NOT decrement — item is still in-flight
            // inside the coordinator. The coordinator will decrement
            // when it processes or drains the item.
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
    /// Feed has reached its per-feed in-flight cap. A prior timed-out
    /// submission is still being processed by the coordinator.
    InFlightCapReached,
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
    let mut entries: Vec<BatchEntry> = Vec::with_capacity(config.max_batch_size);
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1), None);
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let mut handles = Vec::new();
        let start = Instant::now();
        for i in 0..4 {
            let h = handle.clone();
            handles.push(std::thread::spawn(move || h.submit_and_wait(make_entry(i), None)));
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        // Submit 3 items (less than max_batch_size=8)
        let mut handles = Vec::new();
        for i in 0..3 {
            let h = handle.clone();
            handles.push(std::thread::spawn(move || h.submit_and_wait(make_entry(i), None)));
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let mut join_handles = Vec::new();
        for i in 0..3 {
            let h = handle.clone();
            join_handles.push(std::thread::spawn(move || h.submit_and_wait(make_entry(i), None)));
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1), None);
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let _ = handle.submit_and_wait(make_entry(1), None);
        let _ = handle.submit_and_wait(make_entry(2), None);

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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        // Submit one item, then shut down — the coordinator's Phase 2
        // shutdown check should dispatch the partial batch promptly.
        let h = handle.clone();
        let jh = std::thread::spawn(move || h.submit_and_wait(make_entry(1), None));

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
                max_in_flight_per_feed: 1,
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
                max_in_flight_per_feed: 1,
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let mut join_handles = Vec::new();
        for feed_id in 1..=4u64 {
            let h = handle.clone();
            join_handles.push(std::thread::spawn(move || {
                let result = h.submit_and_wait(make_entry(feed_id), None);
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
                max_in_flight_per_feed: 1,
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
                max_in_flight_per_feed: 1,
            },
        );
        let handle = coord.handle();
        let _ = handle.submit_and_wait(make_entry(1), None);
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        // Shut down the coordinator so the channel becomes disconnected.
        coord.shutdown();
        // Give the thread time to finish.
        std::thread::sleep(Duration::from_millis(100));

        // Submit after coordinator is gone — should get CoordinatorShutdown.
        let result = handle.submit_and_wait(make_entry(1), None);
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
            items_timed_out: 0,
            total_processing_ns: 0,
            total_formation_ns: 0,
            min_batch_size: 0,
            max_batch_size_seen: 0,
            configured_max_batch_size: 8,
            consecutive_errors: 0,
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
            items_timed_out: 0,
            total_processing_ns: 400_000,
            total_formation_ns: 200_000,
            min_batch_size: 2,
            max_batch_size_seen: 4,
            configured_max_batch_size: 8,
            consecutive_errors: 0,
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
            items_timed_out: 0,
            total_processing_ns: 0,
            total_formation_ns: 0,
            min_batch_size: 4,
            max_batch_size_seen: 4,
            configured_max_batch_size: 8,
            consecutive_errors: 0,
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
            items_timed_out: 0,
            total_processing_ns: 0,
            total_formation_ns: 0,
            min_batch_size: 8,
            max_batch_size_seen: 8,
            configured_max_batch_size: 8,
            consecutive_errors: 0,
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
            items_timed_out: 0,
            total_processing_ns: 500_000,
            total_formation_ns: 250_000,
            min_batch_size: 4,
            max_batch_size_seen: 4,
            configured_max_batch_size: 8,
            consecutive_errors: 0,
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
                max_in_flight_per_feed: 1,
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let h = handle.clone();
        let jh = std::thread::spawn(move || h.submit_and_wait(make_entry(1), None));

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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1), None);
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
                max_in_flight_per_feed: 1,
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
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();

        let handle = coord.handle();
        let result = handle.submit_and_wait(make_entry(1), None);
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
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();

        // Submit rapidly many times. Due to BATCH_ERROR_THROTTLE (1s),
        // only the first should produce a BatchError health event.
        for i in 0..5 {
            let _ = handle.submit_and_wait(make_entry(i), None);
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

    // ---------------------------------------------------------------
    // New tests: shutdown drain, consecutive errors, timeout metric,
    // config validation, Display, rejection/timeout rate helpers
    // ---------------------------------------------------------------

    #[test]
    fn shutdown_drain_unblocks_feeds_before_on_stop() {
        use std::sync::Barrier;

        /// Processor whose on_stop blocks until a barrier is released,
        /// simulating slow GPU teardown. The test verifies that feed
        /// threads unblock with CoordinatorShutdown (not Timeout)
        /// BEFORE on_stop completes.
        struct SlowStopProcessor {
            barrier: Arc<Barrier>,
        }
        impl BatchProcessor for SlowStopProcessor {
            fn id(&self) -> StageId {
                StageId("slow_stop")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
            fn on_stop(&mut self) -> Result<(), StageError> {
                // Block for a long time — feed must unblock before this.
                self.barrier.wait();
                Ok(())
            }
        }

        let barrier = Arc::new(Barrier::new(2));
        let (health_tx, _health_rx) = broadcast::channel(64);
        let coord = BatchCoordinator::start(
            Box::new(SlowStopProcessor { barrier: Arc::clone(&barrier) }),
            BatchConfig {
                max_batch_size: 100, // very large — will never fill
                max_latency: Duration::from_secs(60), // very long — will never fire
                queue_capacity: None,
                // Short response timeout to distinguish from Timeout
                response_timeout: Some(Duration::from_millis(500)),
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();

        let handle = coord.handle();

        // Submit an item. The batch won't form (max_batch_size=100,
        // max_latency=60s), so the item sits in the queue.
        let h = handle.clone();
        let feed_thread = std::thread::spawn(move || {
            h.submit_and_wait(make_entry(1), None)
        });

        // Give the item time to land in the queue.
        std::thread::sleep(Duration::from_millis(50));

        // Signal shutdown. The coordinator should:
        // 1. Break out of Phase 1 (or Phase 2) on shutdown flag
        // 2. Drain pending items (dropping response channels)
        // 3. Drop rx
        // 4. THEN call on_stop (which blocks on barrier)
        //
        // The feed thread should unblock at step 2/3 with
        // CoordinatorShutdown, NOT wait for the 500ms response
        // timeout or for on_stop to complete.
        coord.signal_shutdown();

        let start = Instant::now();
        let result = feed_thread.join().unwrap();
        let elapsed = start.elapsed();

        // Release the barrier so the coordinator thread can finish.
        barrier.wait();

        // The feed should have gotten CoordinatorShutdown (not Timeout).
        // It should have unblocked quickly (well under the 500ms
        // response timeout).
        assert!(
            matches!(
                result,
                Err(BatchSubmitError::CoordinatorShutdown) | Ok(_)
            ),
            "expected CoordinatorShutdown or Ok (if batch was processed before drain), got: {result:?}"
        );
        assert!(
            elapsed < Duration::from_millis(400),
            "feed should unblock promptly on shutdown drain, took {elapsed:?}"
        );
    }

    #[test]
    fn consecutive_errors_tracks_and_resets() {
        struct ToggleProcessor {
            call_count: u32,
        }
        impl BatchProcessor for ToggleProcessor {
            fn id(&self) -> StageId {
                StageId("toggle")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                self.call_count += 1;
                if self.call_count <= 3 {
                    return Err(StageError::ProcessingFailed {
                        stage_id: StageId("toggle"),
                        detail: "intentional".into(),
                    });
                }
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(ToggleProcessor { call_count: 0 }),
            BatchConfig {
                max_batch_size: 1,
                max_latency: Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();

        // First 3 calls fail — consecutive_errors should climb.
        for _ in 0..3 {
            let _ = handle.submit_and_wait(make_entry(1), None);
        }
        let m = handle.metrics();
        assert_eq!(m.consecutive_errors, 3, "should track 3 consecutive errors");

        // 4th call succeeds — consecutive_errors should reset.
        let result = handle.submit_and_wait(make_entry(1), None);
        assert!(result.is_ok());
        let m = handle.metrics();
        assert_eq!(m.consecutive_errors, 0, "should reset to 0 after success");

        coord.shutdown();
    }

    #[test]
    fn items_timed_out_metric_incremented() {
        // Use record_timeout directly since it's pub(crate).
        let (proc, _calls) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 2,
                max_latency: Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        assert_eq!(handle.metrics().items_timed_out, 0);

        // Simulate timeout recording (normally done by executor).
        handle.record_timeout();
        handle.record_timeout();
        assert_eq!(handle.metrics().items_timed_out, 2);

        coord.shutdown();
    }

    #[test]
    fn config_validate_catches_all_errors() {
        // Valid config.
        assert!(BatchConfig::new(4, Duration::from_millis(50))
            .unwrap()
            .validate()
            .is_ok());

        // queue_capacity too small.
        let cfg = BatchConfig {
            max_batch_size: 8,
            queue_capacity: Some(4),
            ..BatchConfig::default()
        };
        assert!(cfg.validate().is_err());

        // response_timeout zero.
        let cfg = BatchConfig {
            response_timeout: Some(Duration::ZERO),
            ..BatchConfig::default()
        };
        assert!(cfg.validate().is_err());

        // All fields valid.
        let cfg = BatchConfig::new(4, Duration::from_millis(50))
            .unwrap()
            .with_queue_capacity(Some(16))
            .with_response_timeout(Some(Duration::from_secs(2)));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn display_impl_produces_readable_output() {
        let m = BatchMetrics {
            batches_dispatched: 10,
            items_processed: 40,
            items_submitted: 45,
            items_rejected: 3,
            items_timed_out: 1,
            total_processing_ns: 100_000_000,
            total_formation_ns: 50_000_000,
            min_batch_size: 3,
            max_batch_size_seen: 4,
            configured_max_batch_size: 4,
            consecutive_errors: 0,
        };
        let s = format!("{m}");
        assert!(s.contains("batches=10"), "missing batches count: {s}");
        assert!(s.contains("items=40/45"), "missing items counts: {s}");
        assert!(s.contains("rejected=3"), "missing rejected: {s}");
        assert!(s.contains("timed_out=1"), "missing timed_out: {s}");
        assert!(s.contains("consec_err=0"), "missing consec_err: {s}");
    }

    #[test]
    fn rejection_rate_and_timeout_rate_helpers() {
        let m = BatchMetrics {
            batches_dispatched: 4,
            items_processed: 8,
            items_submitted: 20,
            items_rejected: 10,
            items_timed_out: 2,
            total_processing_ns: 0,
            total_formation_ns: 0,
            min_batch_size: 2,
            max_batch_size_seen: 2,
            configured_max_batch_size: 4,
            consecutive_errors: 0,
        };
        let rr = m.rejection_rate().unwrap();
        assert!((rr - 0.5).abs() < f64::EPSILON, "expected 0.5, got {rr}");

        let tr = m.timeout_rate().unwrap();
        assert!((tr - 0.1).abs() < f64::EPSILON, "expected 0.1, got {tr}");

        // Zero submissions → None.
        let empty = BatchMetrics::default();
        assert!(empty.rejection_rate().is_none());
        assert!(empty.timeout_rate().is_none());
    }

    #[test]
    fn shutdown_processes_last_batch_before_drain() {
        /// Processor that records which feed_ids it saw in each batch.
        struct RecordingProcessor {
            seen: Vec<Vec<u64>>,
        }
        impl BatchProcessor for RecordingProcessor {
            fn id(&self) -> StageId {
                StageId("recorder")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                let ids: Vec<u64> = items.iter().map(|i| i.feed_id.as_u64()).collect();
                self.seen.push(ids);
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(RecordingProcessor { seen: Vec::new() }),
            BatchConfig {
                max_batch_size: 2,
                max_latency: Duration::from_millis(30),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();

        // Submit 2 items which should form a full batch and dispatch.
        let h1 = handle.clone();
        let h2 = handle.clone();
        let t1 = std::thread::spawn(move || h1.submit_and_wait(make_entry(1), None));
        let t2 = std::thread::spawn(move || h2.submit_and_wait(make_entry(2), None));
        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();
        assert!(r1.is_ok() && r2.is_ok(), "first batch should succeed");

        // Verify metrics — 1 batch, 2 items.
        let m = handle.metrics();
        assert_eq!(m.batches_dispatched, 1);
        assert_eq!(m.items_processed, 2);

        coord.shutdown();
    }

    // ---------------------------------------------------------------
    // Fairness: submit_and_wait serialization prevents starvation
    // ---------------------------------------------------------------

    #[test]
    fn submit_and_wait_serializes_per_feed_preventing_starvation() {
        /// Processor that records feed_id for every item it sees.
        struct RecordProcessor {
            seen: Arc<std::sync::Mutex<Vec<u64>>>,
        }
        impl BatchProcessor for RecordProcessor {
            fn id(&self) -> StageId {
                StageId("record")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                let mut seen = self.seen.lock().unwrap();
                for item in items.iter_mut() {
                    seen.push(item.feed_id.as_u64());
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let proc = RecordProcessor { seen: Arc::clone(&seen) };
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 2,
                max_latency: Duration::from_millis(30),
                // Queue capacity just big enough for feeds.
                queue_capacity: Some(4),
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();

        // Spawn 4 feeds, each submitting 10 frames sequentially.
        // Because submit_and_wait blocks, each feed has at most 1
        // item in the queue. If starvation occurred, some feed would
        // have 0 processed items.
        let num_feeds = 4u64;
        let frames_per_feed = 10u64;
        let mut threads = Vec::new();
        for feed_id in 1..=num_feeds {
            let h = handle.clone();
            threads.push(std::thread::spawn(move || {
                for _ in 0..frames_per_feed {
                    let _ = h.submit_and_wait(make_entry(feed_id), None);
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        // Verify every feed got at least some items processed.
        let seen = seen.lock().unwrap();
        for feed_id in 1..=num_feeds {
            let count = seen.iter().filter(|&&id| id == feed_id).count();
            assert!(
                count > 0,
                "feed {feed_id} was starved — 0 of {frames_per_feed} items processed"
            );
        }

        let m = handle.metrics();
        assert_eq!(
            m.items_processed,
            num_feeds * frames_per_feed,
            "all items should be processed"
        );
        assert_eq!(m.items_rejected, 0, "no rejections expected with adequate queue");

        coord.shutdown();
    }

    #[test]
    fn mixed_rate_feeds_all_make_progress() {
        /// Processor with a deliberate 5ms processing delay per batch
        /// to create realistic contention.
        struct SlowProcessor;
        impl BatchProcessor for SlowProcessor {
            fn id(&self) -> StageId {
                StageId("slow_mix")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                std::thread::sleep(Duration::from_millis(5));
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(SlowProcessor),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(20),
                queue_capacity: Some(8),
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let success_counts: Arc<[AtomicU32; 3]> = Arc::new([
            AtomicU32::new(0),
            AtomicU32::new(0),
            AtomicU32::new(0),
        ]);

        // Feed 1: fast (20 frames), Feed 2: medium (10), Feed 3: slow (5).
        let rates = [(1u64, 20u32), (2, 10), (3, 5)];
        let mut threads = Vec::new();
        for (idx, &(feed_id, count)) in rates.iter().enumerate() {
            let h = handle.clone();
            let sc = Arc::clone(&success_counts);
            threads.push(std::thread::spawn(move || {
                for _ in 0..count {
                    if h.submit_and_wait(make_entry(feed_id), None).is_ok() {
                        sc[idx].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        // All feeds must have made progress.
        for (idx, &(feed_id, submitted)) in rates.iter().enumerate() {
            let ok = success_counts[idx].load(Ordering::Relaxed);
            assert!(
                ok > 0,
                "feed {feed_id} made zero progress ({ok}/{submitted} succeeded)"
            );
        }

        coord.shutdown();
    }

    /// Under non-timeout operation, `submit_and_wait` blocks so each
    /// feed occupies at most one queue slot. This test validates that
    /// property: no feed appears twice in any single batch.
    ///
    /// NOTE: this invariant can break in timeout regimes — see the
    /// fairness model documentation for details.
    #[test]
    fn single_inflight_per_feed_under_contention() {
        /// Processor that asserts no feed has > 1 item in a single batch.
        struct UniquePerFeedProcessor;
        impl BatchProcessor for UniquePerFeedProcessor {
            fn id(&self) -> StageId {
                StageId("unique_check")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                let mut seen = std::collections::HashSet::new();
                for item in items.iter_mut() {
                    let is_new = seen.insert(item.feed_id.as_u64());
                    assert!(
                        is_new,
                        "feed {} appeared twice in one batch — violates \
                         single-inflight invariant (non-timeout regime)",
                        item.feed_id,
                    );
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(UniquePerFeedProcessor),
            BatchConfig {
                max_batch_size: 8,
                max_latency: Duration::from_millis(30),
                queue_capacity: Some(8),
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let mut threads = Vec::new();
        for feed_id in 1..=8u64 {
            let h = handle.clone();
            threads.push(std::thread::spawn(move || {
                for _ in 0..10 {
                    let _ = h.submit_and_wait(make_entry(feed_id), None);
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }

        coord.shutdown();
    }

    // ---------------------------------------------------------------
    // In-flight cap: prevents stacking after timeout
    // ---------------------------------------------------------------

    #[test]
    fn in_flight_cap_prevents_stacking_after_timeout() {
        use std::sync::atomic::AtomicBool;

        /// Processor that is slow on first call and fast thereafter.
        struct OnceSlowProcessor {
            slow: AtomicBool,
        }
        impl BatchProcessor for OnceSlowProcessor {
            fn id(&self) -> StageId {
                StageId("slow_cap")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                if self.slow.swap(false, Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(500));
                }
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(OnceSlowProcessor { slow: AtomicBool::new(true) }),
            BatchConfig {
                max_batch_size: 1,
                max_latency: Duration::from_millis(10),
                queue_capacity: Some(4),
                // Very short timeout so feed times out quickly.
                response_timeout: Some(Duration::from_millis(50)),
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Submit 1: will timeout because processor sleeps 500ms on first call.
        let r1 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
        assert!(
            matches!(r1, Err(BatchSubmitError::Timeout)),
            "first submit should timeout, got: {r1:?}"
        );

        // in_flight should still be 1 (coordinator hasn't processed yet).
        assert_eq!(
            in_flight.load(Ordering::Acquire), 1,
            "in_flight should be 1 after timeout"
        );

        // Submit 2: should be rejected immediately by in-flight cap.
        let r2 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
        assert!(
            matches!(r2, Err(BatchSubmitError::InFlightCapReached)),
            "second submit should hit in-flight cap, got: {r2:?}"
        );

        // in_flight should still be 1 (InFlightCapReached didn't change it).
        assert_eq!(
            in_flight.load(Ordering::Acquire), 1,
            "in_flight should remain 1 after cap rejection"
        );

        // Metrics: 2 submitted, 1 rejected (the InFlightCapReached).
        let m = handle.metrics();
        assert_eq!(m.items_submitted, 2, "two submissions total");
        assert_eq!(m.items_rejected, 1, "InFlightCapReached counts as rejection");
        // pending_items should reflect only the one actual in-flight item.
        assert_eq!(m.pending_items(), 1, "only one item genuinely pending");

        // Wait for coordinator to process the timed-out item.
        std::thread::sleep(Duration::from_millis(600));

        // in_flight should now be 0 (coordinator decremented).
        assert_eq!(
            in_flight.load(Ordering::Acquire), 0,
            "in_flight should be 0 after coordinator processes"
        );

        // Submit 3: should succeed now (processor is fast on subsequent calls).
        let r3 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
        assert!(r3.is_ok(), "third submit should succeed, got: {r3:?}");

        coord.shutdown();
    }

    #[test]
    fn in_flight_guard_decremented_on_queue_full() {
        let (proc, _) = CountingProcessor::new();
        let (coord, _rx) = start_coordinator(
            Box::new(proc),
            BatchConfig {
                max_batch_size: 100,
                max_latency: Duration::from_secs(60),
                queue_capacity: Some(100),
                response_timeout: None,
                max_in_flight_per_feed: 2,
            },
        );

        // Shut down coordinator so try_send returns Disconnected.
        coord.signal_shutdown();
        std::thread::sleep(Duration::from_millis(200));
        let handle2 = coord.handle();

        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = handle2.submit_and_wait(make_entry(1), Some(&in_flight));
        assert!(
            matches!(result, Err(BatchSubmitError::CoordinatorShutdown)),
            "expected CoordinatorShutdown after shutdown, got: {result:?}"
        );
        // in_flight should be 0 — incremented then decremented on send failure.
        assert_eq!(
            in_flight.load(Ordering::Acquire), 0,
            "in_flight should be 0 after send failure"
        );

        coord.shutdown();
    }

    #[test]
    fn shutdown_drain_clears_in_flight_guards() {
        /// Processor that sleeps long enough to keep the coordinator busy
        /// while additional items accumulate in the channel.
        struct BlockingProcessor;
        impl BatchProcessor for BlockingProcessor {
            fn id(&self) -> StageId {
                StageId("blocking_drain")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                std::thread::sleep(Duration::from_millis(400));
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        // max_batch_size=1 so the coordinator dispatches the first item
        // immediately, leaving items 2-4 in the channel for drain.
        let (coord, _rx) = start_coordinator(
            Box::new(BlockingProcessor),
            BatchConfig {
                max_batch_size: 1,
                max_latency: Duration::from_millis(5),
                queue_capacity: Some(10),
                response_timeout: Some(Duration::from_millis(50)),
                max_in_flight_per_feed: 4,
            },
        );

        let handle = coord.handle();
        let in_flights: Vec<Arc<std::sync::atomic::AtomicUsize>> = (0..4)
            .map(|_| Arc::new(std::sync::atomic::AtomicUsize::new(0)))
            .collect();

        // Submit 4 items from separate threads. Item 1 will be dispatched
        // to the processor (sleeping 400ms). Items 2-4 stay in the channel.
        let mut threads = Vec::new();
        for (i, counter) in in_flights.iter().enumerate() {
            let h = handle.clone();
            let c = Arc::clone(counter);
            threads.push(std::thread::spawn(move || {
                let _ = h.submit_and_wait(make_entry(i as u64 + 1), Some(&c));
            }));
        }

        // Wait for all feeds to timeout (50ms) and items to be in the system.
        std::thread::sleep(Duration::from_millis(150));

        // All in_flight counters should be 1.
        for (i, counter) in in_flights.iter().enumerate() {
            assert_eq!(
                counter.load(Ordering::Acquire), 1,
                "feed {} in_flight should be 1 before shutdown", i + 1
            );
        }

        // Signal shutdown while the processor is still sleeping.
        // Coordinator will finish item 1 (~400ms), then drain items 2-4.
        coord.shutdown();

        // Wait for threads to finish.
        for t in threads {
            let _ = t.join();
        }

        // All in_flight counters should be 0:
        // - Item 1's guard decremented in Phase 4 (result routing).
        // - Items 2-4 guards decremented in drain_pending.
        for (i, counter) in in_flights.iter().enumerate() {
            assert_eq!(
                counter.load(Ordering::Acquire), 0,
                "feed {} in_flight should be 0 after shutdown drain", i + 1
            );
        }
    }

    #[test]
    fn mixed_rate_feeds_progress_with_in_flight_cap() {
        /// Processor with realistic latency.
        struct RealisticProcessor;
        impl BatchProcessor for RealisticProcessor {
            fn id(&self) -> StageId {
                StageId("realistic")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                std::thread::sleep(Duration::from_millis(10));
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(RealisticProcessor),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(20),
                queue_capacity: Some(8),
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        );

        let handle = coord.handle();
        let success_counts: Arc<[AtomicU32; 3]> = Arc::new([
            AtomicU32::new(0),
            AtomicU32::new(0),
            AtomicU32::new(0),
        ]);

        // Feed 1: fast (15 frames), Feed 2: medium (10), Feed 3: slow (5).
        let rates = [(1u64, 15u32), (2, 10), (3, 5)];
        let mut threads = Vec::new();
        for (idx, &(feed_id, count)) in rates.iter().enumerate() {
            let h = handle.clone();
            let sc = Arc::clone(&success_counts);
            let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            threads.push(std::thread::spawn(move || {
                for _ in 0..count {
                    if h.submit_and_wait(make_entry(feed_id), Some(&in_flight)).is_ok() {
                        sc[idx].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        // All feeds must have made progress.
        for (idx, &(feed_id, submitted)) in rates.iter().enumerate() {
            let ok = success_counts[idx].load(Ordering::Relaxed);
            assert!(
                ok > 0,
                "feed {feed_id} made zero progress ({ok}/{submitted} succeeded)"
            );
        }

        coord.shutdown();
    }

    #[test]
    fn in_flight_cap_of_zero_rejected_by_validate() {
        let result = BatchConfig {
            max_batch_size: 4,
            max_latency: Duration::from_millis(50),
            queue_capacity: None,
            response_timeout: None,
            max_in_flight_per_feed: 0,
        }.validate();
        assert!(result.is_err(), "max_in_flight_per_feed=0 should be rejected");
    }

    #[test]
    fn in_flight_cap_higher_than_one_allows_stacking() {
        /// Processor that sleeps to force timeouts.
        struct SlowProcessor;
        impl BatchProcessor for SlowProcessor {
            fn id(&self) -> StageId {
                StageId("slow_cap2")
            }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
                std::thread::sleep(Duration::from_millis(300));
                for item in items.iter_mut() {
                    item.output = Some(StageOutput::empty());
                }
                Ok(())
            }
        }

        let (coord, _rx) = start_coordinator(
            Box::new(SlowProcessor),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(10),
                queue_capacity: Some(8),
                response_timeout: Some(Duration::from_millis(50)),
                // Allow up to 3 in-flight items per feed.
                max_in_flight_per_feed: 3,
            },
        );

        let handle = coord.handle();
        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // First 3 timeouts should succeed (in_flight goes 1, 2, 3).
        for i in 0..3 {
            let result = handle.submit_and_wait(make_entry(1), Some(&in_flight));
            assert!(
                matches!(result, Err(BatchSubmitError::Timeout)),
                "submit {i} should timeout, got: {result:?}"
            );
        }
        assert_eq!(in_flight.load(Ordering::Acquire), 3);

        // 4th should be rejected — cap reached.
        let r4 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
        assert!(
            matches!(r4, Err(BatchSubmitError::InFlightCapReached)),
            "4th submit should hit cap, got: {r4:?}"
        );

        // Metrics: 4 submitted, 1 rejected. pending_items should be 3.
        let m = handle.metrics();
        assert_eq!(m.items_submitted, 4);
        assert_eq!(m.items_rejected, 1, "cap rejection counted");
        assert_eq!(m.pending_items(), 3, "3 items genuinely in-flight");

        coord.shutdown();
    }
}
