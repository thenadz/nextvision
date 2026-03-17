//! Adapters bridging the media ingress into the feed worker's frame queue.
//!
//! Contains [`FeedFrameSink`] (the [`FrameSink`] implementation that pushes
//! into a [`FrameQueue`]) and [`BackpressureThrottle`] (coalesces per-frame
//! drop events to avoid health-event storms).

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nv_core::error::MediaError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_frame::FrameEnvelope;
use nv_media::ingress::FrameSink;
use tokio::sync::broadcast;

use crate::queue::{FrameQueue, PushOutcome};

use super::shared_state::FeedSharedState;

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
pub(super) struct BackpressureThrottle {
    inner: Mutex<BpThrottleInner>,
}

struct BpThrottleInner {
    in_backpressure: bool,
    accumulated: u64,
    last_event: Instant,
}

impl BackpressureThrottle {
    pub(super) fn new() -> Self {
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
// FrameSink adapter — bridges media ingress → FrameQueue
// ---------------------------------------------------------------------------

/// Adapter that implements [`FrameSink`] by pushing into a [`FrameQueue`].
///
/// Created per feed session. The `Arc<FrameQueue>` is shared with the
/// feed worker thread's pop loop.
pub(super) struct FeedFrameSink {
    pub queue: Arc<FrameQueue>,
    pub shared: Arc<FeedSharedState>,
    pub health_tx: broadcast::Sender<HealthEvent>,
    pub feed_id: FeedId,
    pub bp_throttle: BackpressureThrottle,
}

impl FrameSink for FeedFrameSink {
    fn on_frame(&self, frame: FrameEnvelope) {
        self.shared.frames_received.fetch_add(1, Ordering::Relaxed);
        let outcome = self.queue.push(frame);
        match outcome {
            PushOutcome::Accepted => {}
            PushOutcome::DroppedOldest | PushOutcome::Rejected => {
                self.shared.frames_dropped.fetch_add(1, Ordering::Relaxed);
                self.bp_throttle.record_drop(&self.health_tx, self.feed_id);
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
