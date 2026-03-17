use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_perception::StageOutput;
use nv_perception::batch::BatchEntry;

use super::config::BatchConfig;
use super::metrics::{BatchMetrics, BatchMetricsInner};

/// Default response safety timeout when [`BatchConfig::response_timeout`]
/// is `None`. See that field's documentation for semantics.
const DEFAULT_RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Internal submission types
// ---------------------------------------------------------------------------

/// A pending item: the entry to process plus the response channel
/// and optional per-feed in-flight guard.
pub(super) struct PendingEntry {
    pub(super) entry: BatchEntry,
    pub(super) response_tx: std::sync::mpsc::SyncSender<Result<StageOutput, StageError>>,
    /// Per-feed in-flight counter. The coordinator decrements this
    /// after routing the result (or on drain). Must be decremented
    /// exactly once per PendingEntry that reached the coordinator.
    pub(super) in_flight_guard: Option<Arc<AtomicUsize>>,
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
    pub(super) submit_tx: std::sync::mpsc::SyncSender<PendingEntry>,
    pub(super) metrics: Arc<BatchMetricsInner>,
    pub(super) processor_id: StageId,
    pub(super) config: BatchConfig,
    pub(super) capabilities: Option<nv_perception::stage::StageCapabilities>,
}

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
        self.inner
            .metrics
            .snapshot(self.inner.config.max_batch_size as u64)
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
        let safety = self
            .inner
            .config
            .response_timeout
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
