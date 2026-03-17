//! Bounded frame queue with configurable backpressure.
//!
//! The [`FrameQueue`] sits between the media ingress (producer — GStreamer
//! streaming thread) and the pipeline executor (consumer — feed worker
//! thread). It enforces the configured [`BackpressurePolicy`]:
//!
//! - **`DropOldest`** — evicts the oldest frame to make room (default, real-time friendly).
//! - **`DropNewest`** — rejects the incoming frame when full.
//! - **`Block`** — blocks the producer until space is available (use with caution).
//!
//! The queue is closed by calling [`close()`](FrameQueue::close), which wakes
//! all waiters and causes subsequent [`pop()`](FrameQueue::pop) calls to
//! return `None`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use nv_frame::FrameEnvelope;

use crate::backpressure::BackpressurePolicy;

/// Outcome of pushing a frame into the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PushOutcome {
    /// Frame was accepted into the queue.
    Accepted,
    /// Queue was full: the oldest frame was evicted to make room.
    DroppedOldest,
    /// Queue was full or closed: the incoming frame was discarded.
    Rejected,
}

/// Result from a [`FrameQueue::pop`] call with deadline support.
#[derive(Debug)]
pub(crate) enum PopResult {
    /// A frame was available and returned.
    Frame(FrameEnvelope),
    /// The queue was closed or shutdown was requested.
    Closed,
    /// The deadline elapsed with no frame available.
    Timeout,
    /// The consumer was explicitly woken for control-plane processing
    /// (e.g., a bus error or EOS that requires the worker to tick the
    /// source). No frame is available, but the caller should take action.
    Wake,
}

/// Internal mutable state behind the lock.
struct QueueInner {
    buf: VecDeque<FrameEnvelope>,
    closed: bool,
    /// Set by [`wake_consumer()`](FrameQueue::wake_consumer) to signal a
    /// control-plane wake. Cleared by [`pop()`](FrameQueue::pop) when it
    /// returns [`PopResult::Wake`].
    woken: bool,
    total_received: u64,
    total_dropped: u64,
}

/// Bounded frame queue with configurable backpressure.
///
/// Thread-safe: the producer (GStreamer streaming thread) and consumer
/// (feed worker thread) operate concurrently. Uses `Mutex` + `Condvar`
/// for simplicity and correctness — contention is minimal because the
/// producer pushes at frame rate (~30 Hz) and the consumer pops at
/// processing rate.
///
/// Shutdown and close are event-driven: [`wake_consumer()`](Self::wake_consumer)
/// notifies the consumer condvar so there is no fixed polling delay.
pub(crate) struct FrameQueue {
    inner: Mutex<QueueInner>,
    /// Signaled when a frame is pushed (wakes the consumer).
    not_empty: Condvar,
    /// Signaled when a frame is popped (wakes a blocked producer).
    not_full: Condvar,
    depth: usize,
    policy: BackpressurePolicy,
}

impl FrameQueue {
    /// Create a new bounded frame queue.
    pub fn new(policy: BackpressurePolicy) -> Self {
        let depth = policy.queue_depth();
        Self {
            inner: Mutex::new(QueueInner {
                buf: VecDeque::with_capacity(depth),
                closed: false,
                woken: false,
                total_received: 0,
                total_dropped: 0,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            depth,
            policy,
        }
    }

    /// Push a frame, applying the configured backpressure policy.
    ///
    /// Called from the media ingress (GStreamer streaming thread).
    pub fn push(&self, frame: FrameEnvelope) -> PushOutcome {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.closed {
            tracing::debug!("FrameQueue::push — rejected: queue is closed");
            return PushOutcome::Rejected;
        }
        inner.total_received += 1;

        // Fast path: space available.
        if inner.buf.len() < self.depth {
            inner.buf.push_back(frame);
            self.not_empty.notify_one();
            return PushOutcome::Accepted;
        }

        // Queue is full — apply policy.
        match self.policy {
            BackpressurePolicy::DropOldest { .. } => {
                inner.buf.pop_front();
                inner.total_dropped += 1;
                inner.buf.push_back(frame);
                self.not_empty.notify_one();
                PushOutcome::DroppedOldest
            }
            BackpressurePolicy::DropNewest { .. } => {
                inner.total_dropped += 1;
                PushOutcome::Rejected
            }
            BackpressurePolicy::Block { .. } => {
                // Block until space is available or the queue is closed.
                while inner.buf.len() >= self.depth && !inner.closed {
                    inner = self.not_full.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
                if inner.closed {
                    return PushOutcome::Rejected;
                }
                inner.buf.push_back(frame);
                self.not_empty.notify_one();
                PushOutcome::Accepted
            }
        }
    }

    /// Pop the next frame, blocking until one is available or the deadline
    /// elapses.
    ///
    /// - `shutdown` — checked on every wake; returns `Closed` when set.
    /// - `deadline` — if `Some`, returns `Timeout` when the deadline passes
    ///   with no frame available. If `None`, waits indefinitely (pure
    ///   event-driven — woken by push, close, or
    ///   [`wake_consumer()`](Self::wake_consumer)).
    ///
    /// Called from the feed worker thread.
    pub fn pop(&self, shutdown: &AtomicBool, deadline: Option<Instant>) -> PopResult {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            // Highest priority: deliver any buffered frame.
            if let Some(frame) = inner.buf.pop_front() {
                self.not_full.notify_one();
                return PopResult::Frame(frame);
            }
            // Terminal: closed or shutdown.
            if inner.closed || shutdown.load(Ordering::Relaxed) {
                return PopResult::Closed;
            }
            // Control-plane wake: the backend signaled a lifecycle event.
            if inner.woken {
                inner.woken = false;
                return PopResult::Wake;
            }
            // Nothing ready — wait for a frame, wake, close, or deadline.
            match deadline {
                Some(dl) => {
                    let now = Instant::now();
                    if now >= dl {
                        return PopResult::Timeout;
                    }
                    let (guard, result) = self
                        .not_empty
                        .wait_timeout(inner, dl - now)
                        .unwrap_or_else(|e| e.into_inner());
                    inner = guard;
                    if result.timed_out() && inner.buf.is_empty() {
                        if inner.closed || shutdown.load(Ordering::Relaxed) {
                            return PopResult::Closed;
                        }
                        if inner.woken {
                            inner.woken = false;
                            return PopResult::Wake;
                        }
                        return PopResult::Timeout;
                    }
                }
                None => {
                    inner = self
                        .not_empty
                        .wait(inner)
                        .unwrap_or_else(|e| e.into_inner());
                }
            }
        }
    }

    /// Wake the consumer without pushing a frame.
    ///
    /// Sets the `woken` flag so that [`pop()`](Self::pop) returns
    /// [`PopResult::Wake`] instead of re-entering the wait loop.
    /// Used by the media backend (via [`FrameSink::on_eos()`] /
    /// [`FrameSink::wake()`]) to signal lifecycle events that require
    /// the worker to tick the source.
    pub fn wake_consumer(&self) {
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.woken = true;
        }
        self.not_empty.notify_all();
    }

    /// Close the queue: reject future pushes and wake all waiters.
    pub fn close(&self) {
        tracing::debug!("FrameQueue::close — closing queue");
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.closed = true;
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Current number of frames in the queue.
    ///
    /// Acquires the internal lock briefly. The result is immediately
    /// stale under concurrent push/pop, but is suitable for monitoring
    /// and dashboards.
    pub(crate) fn depth(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .buf
            .len()
    }

    /// Maximum capacity of the queue.
    #[must_use]
    pub(crate) fn capacity(&self) -> usize {
        self.depth
    }
}

#[cfg(test)]
impl FrameQueue {
    pub fn stats(&self) -> (u64, u64) {
        let inner = self.inner.lock().unwrap();
        (inner.total_received, inner.total_dropped)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::{FeedId, MonotonicTs, TypedMetadata, WallTs};
    use nv_frame::PixelFormat;
    use std::time::Duration;

    fn test_frame(seq: u64) -> FrameEnvelope {
        FrameEnvelope::new_owned(
            FeedId::new(1),
            seq,
            MonotonicTs::from_nanos(seq * 33_333_333),
            WallTs::from_micros(0),
            2,
            2,
            PixelFormat::Rgb8,
            6,
            vec![0u8; 12],
            TypedMetadata::new(),
        )
    }

    #[test]
    fn push_pop_basic() {
        let q = FrameQueue::new(BackpressurePolicy::DropOldest { queue_depth: 4 });
        assert_eq!(q.push(test_frame(0)), PushOutcome::Accepted);
        assert_eq!(q.push(test_frame(1)), PushOutcome::Accepted);
        assert_eq!(q.len(), 2);

        let shutdown = AtomicBool::new(false);
        let PopResult::Frame(f) = q.pop(&shutdown, None) else {
            panic!("expected frame");
        };
        assert_eq!(f.seq(), 0);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn drop_oldest_evicts_when_full() {
        let q = FrameQueue::new(BackpressurePolicy::DropOldest { queue_depth: 2 });
        assert_eq!(q.push(test_frame(0)), PushOutcome::Accepted);
        assert_eq!(q.push(test_frame(1)), PushOutcome::Accepted);
        assert_eq!(q.push(test_frame(2)), PushOutcome::DroppedOldest);
        assert_eq!(q.len(), 2);

        let shutdown = AtomicBool::new(false);
        // Frame 0 was evicted; first available is frame 1.
        let PopResult::Frame(f) = q.pop(&shutdown, None) else {
            panic!("expected frame");
        };
        assert_eq!(f.seq(), 1);

        let (received, dropped) = q.stats();
        assert_eq!(received, 3);
        assert_eq!(dropped, 1);
    }

    #[test]
    fn drop_newest_rejects_when_full() {
        let q = FrameQueue::new(BackpressurePolicy::DropNewest { queue_depth: 2 });
        assert_eq!(q.push(test_frame(0)), PushOutcome::Accepted);
        assert_eq!(q.push(test_frame(1)), PushOutcome::Accepted);
        assert_eq!(q.push(test_frame(2)), PushOutcome::Rejected);
        assert_eq!(q.len(), 2);

        let shutdown = AtomicBool::new(false);
        // Frame 2 was rejected; queue still has 0 and 1.
        let PopResult::Frame(f) = q.pop(&shutdown, None) else {
            panic!("expected frame");
        };
        assert_eq!(f.seq(), 0);
    }

    #[test]
    fn close_wakes_consumer() {
        let q = std::sync::Arc::new(FrameQueue::new(BackpressurePolicy::default()));
        let q2 = q.clone();
        let shutdown = std::sync::Arc::new(AtomicBool::new(false));
        let sd = shutdown.clone();

        let handle = std::thread::spawn(move || q2.pop(&sd, None));

        // Give the consumer time to block.
        std::thread::sleep(Duration::from_millis(50));
        q.close();

        let result = handle.join().unwrap();
        assert!(
            matches!(result, PopResult::Closed),
            "pop should return Closed after close"
        );
    }

    #[test]
    fn shutdown_wakes_consumer() {
        let q = std::sync::Arc::new(FrameQueue::new(BackpressurePolicy::default()));
        let q2 = q.clone();
        let shutdown = std::sync::Arc::new(AtomicBool::new(false));
        let sd = shutdown.clone();

        let handle = std::thread::spawn(move || q2.pop(&sd, None));

        std::thread::sleep(Duration::from_millis(50));
        shutdown.store(true, Ordering::Relaxed);
        q.wake_consumer();

        let result = handle.join().unwrap();
        assert!(
            matches!(result, PopResult::Closed),
            "pop should return Closed after shutdown"
        );
    }

    #[test]
    fn block_policy_waits_for_space() {
        let q = std::sync::Arc::new(FrameQueue::new(BackpressurePolicy::Block {
            queue_depth: 1,
        }));
        let q2 = q.clone();

        // Fill the queue.
        assert_eq!(q.push(test_frame(0)), PushOutcome::Accepted);

        // Producer blocks because queue is full.
        let handle = std::thread::spawn(move || q2.push(test_frame(1)));

        // Give producer time to block.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(q.len(), 1);

        // Consumer pops, freeing space.
        let shutdown = AtomicBool::new(false);
        let PopResult::Frame(_) = q.pop(&shutdown, None) else {
            panic!("expected frame");
        };

        let outcome = handle.join().unwrap();
        assert_eq!(outcome, PushOutcome::Accepted);
    }

    #[test]
    fn pop_with_deadline_returns_timeout() {
        let q = FrameQueue::new(BackpressurePolicy::DropOldest { queue_depth: 4 });
        let shutdown = AtomicBool::new(false);
        let deadline = Instant::now() + Duration::from_millis(10);
        let result = q.pop(&shutdown, Some(deadline));
        assert!(
            matches!(result, PopResult::Timeout),
            "expected Timeout on empty queue with deadline"
        );
    }

    #[test]
    fn push_after_close_is_rejected() {
        let q = FrameQueue::new(BackpressurePolicy::default());
        q.close();
        assert_eq!(q.push(test_frame(0)), PushOutcome::Rejected);
    }

    #[test]
    fn depth_and_capacity() {
        let q = FrameQueue::new(BackpressurePolicy::DropOldest { queue_depth: 4 });
        assert_eq!(q.capacity(), 4);
        assert_eq!(q.depth(), 0);

        q.push(test_frame(0));
        q.push(test_frame(1));
        assert_eq!(q.depth(), 2);

        let shutdown = AtomicBool::new(false);
        let _ = q.pop(&shutdown, None);
        assert_eq!(q.depth(), 1);

        let _ = q.pop(&shutdown, None);
        assert_eq!(q.depth(), 0);
    }

    #[test]
    fn depth_under_backpressure() {
        let q = FrameQueue::new(BackpressurePolicy::DropOldest { queue_depth: 2 });
        q.push(test_frame(0));
        q.push(test_frame(1));
        assert_eq!(q.depth(), 2);

        // Push a third frame — oldest evicted, depth stays at capacity.
        q.push(test_frame(2));
        assert_eq!(q.depth(), 2);
        assert_eq!(q.capacity(), 2);
    }

    #[test]
    fn depth_returns_zero_after_close() {
        let q = FrameQueue::new(BackpressurePolicy::DropOldest { queue_depth: 4 });
        q.push(test_frame(0));
        q.push(test_frame(1));
        q.close();
        // Items are still in the buffer after close (close just prevents
        // new pushes and wakes waiters).
        assert_eq!(q.depth(), 2);

        // Drain the queue.
        let shutdown = AtomicBool::new(false);
        let _ = q.pop(&shutdown, None);
        let _ = q.pop(&shutdown, None);
        assert_eq!(q.depth(), 0);
    }
}
