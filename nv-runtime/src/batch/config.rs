use std::time::Duration;

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
    /// [`submit_and_wait`](super::BatchHandle::submit_and_wait) returns
    /// [`BatchSubmitError::QueueFull`](super::BatchSubmitError::QueueFull).
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
    /// [`submit_and_wait`](super::BatchHandle::submit_and_wait) calls fail
    /// immediately with [`BatchSubmitError::InFlightCapReached`](super::BatchSubmitError::InFlightCapReached)
    /// rather than adding to the queue.
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
    /// Maximum time to wait for [`BatchProcessor::on_start()`] to
    /// complete before returning an error.
    ///
    /// GPU-backed processors (e.g. TensorRT engine compilation) may
    /// need significantly longer than CPU-only models. Set this to
    /// accommodate worst-case first-run warm-up on the target hardware.
    ///
    /// Defaults to 30 seconds when `None`. Must be > 0 when specified.
    pub startup_timeout: Option<Duration>,
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
            startup_timeout: None,
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

    /// Set the maximum time to wait for `on_start()` to complete.
    ///
    /// Pass `None` for the default (30 seconds). GPU-backed processors
    /// (e.g. TensorRT engine build on first run) may need 2–5 minutes.
    /// Must be > 0 when specified.
    #[must_use]
    pub fn with_startup_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.startup_timeout = timeout;
        self
    }

    /// Validate all configuration fields.
    ///
    /// Called internally by [`BatchCoordinator::start`](super::BatchCoordinator::start).
    /// Also available for early validation before passing a config to the runtime.
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
        if let Some(st) = self.startup_timeout {
            if st.is_zero() {
                return Err(ConfigError::InvalidPolicy {
                    detail: "batch startup_timeout must be > 0".into(),
                });
            }
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
            startup_timeout: None,
        }
    }
}
