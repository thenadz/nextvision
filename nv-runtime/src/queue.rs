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
use std::time::Duration;

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

/// Internal mutable state behind the lock.
struct QueueInner {
    buf: VecDeque<FrameEnvelope>,
    closed: bool,
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
pub(crate) struct FrameQueue {
    inner: Mutex<QueueInner>,
    /// Signaled when a frame is pushed (wakes the consumer).
    not_empty: Condvar,
    /// Signaled when a frame is popped (wakes a blocked producer).
    not_full: Condvar,
    depth: usize,
    policy: BackpressurePolicy,
}

/// Poll timeout for checking shutdown between waits.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

impl FrameQueue {
    /// Create a new bounded frame queue.
    pub fn new(policy: BackpressurePolicy) -> Self {
        let depth = policy.queue_depth();
        Self {
            inner: Mutex::new(QueueInner {
                buf: VecDeque::with_capacity(depth),
                closed: false,
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
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
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
                    let (guard, _) = self
                        .not_full
                        .wait_timeout(inner, POLL_INTERVAL)
                        .unwrap();
                    inner = guard;
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

    /// Pop the next frame, blocking until one is available.
    ///
    /// Returns `None` if the queue has been closed or `shutdown` is set.
    /// Called from the feed worker thread.
    pub fn pop(&self, shutdown: &AtomicBool) -> Option<FrameEnvelope> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(frame) = inner.buf.pop_front() {
                self.not_full.notify_one();
                return Some(frame);
            }
            if inner.closed || shutdown.load(Ordering::Relaxed) {
                return None;
            }
            let (guard, _) = self
                .not_empty
                .wait_timeout(inner, POLL_INTERVAL)
                .unwrap();
            inner = guard;
        }
    }

    /// Close the queue: reject future pushes and wake all waiters.
    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Returns `(total_received, total_dropped)`.
    pub fn stats(&self) -> (u64, u64) {
        let inner = self.inner.lock().unwrap();
        (inner.total_received, inner.total_dropped)
    }

    /// Current number of buffered frames.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().buf.len()
    }

    /// Whether the queue has been closed.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::{FeedId, MonotonicTs, TypedMetadata, WallTs};
    use nv_frame::PixelFormat;

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
        let f = q.pop(&shutdown).unwrap();
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
        let f = q.pop(&shutdown).unwrap();
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
        let f = q.pop(&shutdown).unwrap();
        assert_eq!(f.seq(), 0);
    }

    #[test]
    fn close_wakes_consumer() {
        let q = std::sync::Arc::new(FrameQueue::new(BackpressurePolicy::default()));
        let q2 = q.clone();
        let shutdown = std::sync::Arc::new(AtomicBool::new(false));
        let sd = shutdown.clone();

        let handle = std::thread::spawn(move || q2.pop(&sd));

        // Give the consumer time to block.
        std::thread::sleep(Duration::from_millis(50));
        q.close();

        let result = handle.join().unwrap();
        assert!(result.is_none(), "pop should return None after close");
    }

    #[test]
    fn shutdown_wakes_consumer() {
        let q = std::sync::Arc::new(FrameQueue::new(BackpressurePolicy::default()));
        let q2 = q.clone();
        let shutdown = std::sync::Arc::new(AtomicBool::new(false));
        let sd = shutdown.clone();

        let handle = std::thread::spawn(move || q2.pop(&sd));

        std::thread::sleep(Duration::from_millis(50));
        shutdown.store(true, Ordering::Relaxed);

        let result = handle.join().unwrap();
        assert!(result.is_none(), "pop should return None after shutdown");
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
        let _ = q.pop(&shutdown).unwrap();

        let outcome = handle.join().unwrap();
        assert_eq!(outcome, PushOutcome::Accepted);
    }

    #[test]
    fn push_after_close_is_rejected() {
        let q = FrameQueue::new(BackpressurePolicy::default());
        q.close();
        assert_eq!(q.push(test_frame(0)), PushOutcome::Rejected);
    }
}
