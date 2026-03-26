//! Shared batch coordination infrastructure.
//!
//! This module provides the runtime-side machinery for cross-feed batch
//! inference: a `BatchCoordinator` that collects frames from feed
//! threads, forms bounded batches, dispatches to a user-supplied
//! `BatchProcessor`, and routes results
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
//! Each feed thread submits a `BatchEntry` (frame + view snapshot) via
//! a bounded channel and blocks on a per-item response channel. The
//! coordinator thread collects items until `max_batch_size` or
//! `max_latency`, calls the processor, and routes results back.
//!
//! # Backpressure
//!
//! Submission uses `try_send` (non-blocking). If the coordinator queue is
//! full, the feed thread receives `BatchSubmitError::QueueFull` and the
//! frame is dropped with a `HealthEvent::BatchSubmissionRejected` event.
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
//! Under normal operation, `submit_and_wait`
//! is synchronous — the feed thread blocks until the result arrives.
//! With the default cap of 1, at most one item per feed is in the queue
//! at any time.
//!
//! **Timeout regime**: when `submit_and_wait` returns
//! `BatchSubmitError::Timeout`, the timed-out item remains in-flight
//! inside the coordinator. The in-flight cap prevents the feed from
//! stacking additional items: the next `submit_and_wait` call returns
//! `BatchSubmitError::InFlightCapReached` immediately until the
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
//! `HealthEvent::BatchSubmissionRejected` events, per-feed
//! timeout counts via `HealthEvent::BatchTimeout` events, and
//! per-feed in-flight cap rejections via
//! `HealthEvent::BatchInFlightExceeded` events (all coalesced
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
//! `BatchProcessor::on_start()` is given [`BatchConfig::startup_timeout`]
//! (default 30 s, configurable). GPU-backed processors such as TensorRT
//! may need several minutes for first-run engine compilation; set a
//! longer timeout via [`BatchConfig::with_startup_timeout`].
//! If it exceeds the timeout, the coordinator attempts a 2-second bounded
//! join before returning an error. If the thread is still alive after the
//! grace period it is detached (inherent safe-Rust limitation).
//!
//! ## Response timeout
//!
//! `submit_and_wait` blocks for at most `max_latency + response_timeout`
//! before returning `BatchSubmitError::Timeout`. The response timeout
//! defaults to 5 s (`DEFAULT_RESPONSE_TIMEOUT`) and can be configured
//! via [`BatchConfig::response_timeout`].
//!
//! ## Expected vs unexpected coordinator loss
//!
//! `PipelineExecutor` distinguishes
//! expected shutdown (feed/runtime is shutting down) from unexpected
//! coordinator death by checking the feed's shutdown flag. Unexpected
//! loss emits exactly one `HealthEvent::StageError` per feed.
//!
//! # Observability
//!
//! [`BatchHandle::metrics()`] returns a [`BatchMetrics`] snapshot with:
//! batches dispatched, items processed, items rejected, processing
//! latency, formation latency, and batch-size distribution.

mod config;
mod coordinator;
mod handle;
mod metrics;

// Public re-exports (visible to downstream crates via `nv_runtime::batch::*`
// and re-exported from `nv_runtime` in lib.rs).
pub use config::BatchConfig;
pub use handle::BatchHandle;
pub use metrics::BatchMetrics;

// Crate-internal re-exports (used by executor, runtime, worker modules).
pub(crate) use coordinator::BatchCoordinator;
pub(crate) use handle::BatchSubmitError;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
