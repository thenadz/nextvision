use std::sync::atomic::{AtomicU64, Ordering};

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

// ---------------------------------------------------------------------------
// BatchMetricsInner
// ---------------------------------------------------------------------------

/// Internal atomic counters for lock-free metrics recording.
pub(super) struct BatchMetricsInner {
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
    pub(super) fn new() -> Self {
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

    pub(super) fn snapshot(&self, configured_max_batch_size: u64) -> BatchMetrics {
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

    pub(super) fn record_dispatch(&self, batch_size: usize, formation_ns: u64, processing_ns: u64) {
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

    pub(super) fn record_submission(&self) {
        self.items_submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record_rejection(&self) {
        self.items_rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record_timeout(&self) {
        self.items_timed_out.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record_batch_success(&self) {
        self.consecutive_errors.store(0, Ordering::Relaxed);
    }

    pub(super) fn record_batch_error(&self) {
        self.consecutive_errors.fetch_add(1, Ordering::Relaxed);
    }
}
