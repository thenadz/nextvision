//! Sink-related tests: slow sink, blocking sink shutdown, and
//! sink backpressure throttling.

use super::super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nv_test_util::mock_stage::NoOpStage;
use tokio::sync::broadcast;

use crate::output::{OutputSink, SharedOutput};
use crate::shutdown::{RestartPolicy, RestartTrigger};

use super::harness::*;

// ---------------------------------------------------------------------------
// Slow sink does not block frame processing
// ---------------------------------------------------------------------------

/// Output sink that deliberately sleeps to simulate slow I/O.
struct SlowSink {
    count: Arc<AtomicU64>,
    delay: std::time::Duration,
}

impl OutputSink for SlowSink {
    fn emit(&self, _output: SharedOutput) {
        self.count.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(self.delay);
    }
}

#[test]
fn slow_sink_does_not_block_processing() {
    let count = Arc::new(AtomicU64::new(0));
    let sink = SlowSink {
        count: Arc::clone(&count),
        delay: std::time::Duration::from_millis(100),
    };

    let runtime = build_runtime(50);

    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(10));

    let m = handle.metrics();
    assert!(
        m.frames_processed >= 50,
        "processing should complete all frames regardless of sink speed (got {})",
        m.frames_processed
    );

    let emitted = count.load(Ordering::Relaxed);
    assert!(
        emitted > 0 && emitted <= m.frames_processed,
        "sink should have received some (not all) outputs: got {emitted}"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Bounded sink shutdown
// ---------------------------------------------------------------------------

/// Sink that blocks forever in emit() to simulate a stuck downstream.
struct BlockingSink {
    count: Arc<AtomicU64>,
}

impl OutputSink for BlockingSink {
    fn emit(&self, _output: SharedOutput) {
        self.count.fetch_add(1, Ordering::Relaxed);
        // Block forever — simulates I/O hung downstream.
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }
}

/// Verify feed shutdown completes within bounded time even when
/// OutputSink::emit() is blocked indefinitely.
#[test]
fn bounded_shutdown_when_sink_blocks() {
    let count = Arc::new(AtomicU64::new(0));
    let sink = BlockingSink {
        count: Arc::clone(&count),
    };

    let runtime = build_runtime(20);

    let _handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    // Let some frames flow so the sink thread picks up at least one.
    std::thread::sleep(std::time::Duration::from_millis(200));

    let start = std::time::Instant::now();
    runtime.shutdown().unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "shutdown should be bounded even with blocking sink (took {:?})",
        elapsed,
    );
}

// ---------------------------------------------------------------------------
// SinkBackpressure throttling
// ---------------------------------------------------------------------------

/// Sink that never keeps up — every emit takes 500ms.
struct VerySlowSink {
    count: Arc<AtomicU64>,
}

impl OutputSink for VerySlowSink {
    fn emit(&self, _output: SharedOutput) {
        self.count.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

/// Under sustained sink backpressure, SinkBackpressure events should
/// be coalesced rather than emitted per-drop.
#[test]
fn sink_backpressure_throttling() {
    let frame_count = 100u64;
    let sink_count = Arc::new(AtomicU64::new(0));
    let sink = VerySlowSink {
        count: Arc::clone(&sink_count),
    };

    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count,
            fail_on_start: false,
            frame_delay: std::time::Duration::from_millis(1),
        }))
        .health_capacity(4096)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();

    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 0,
                restart_on: RestartTrigger::Never,
                ..Default::default()
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(10));

    // Count SinkBackpressure events.
    let mut bp_events = 0u64;
    let mut total_dropped_reported = 0u64;
    loop {
        match health_rx.try_recv() {
            Ok(HealthEvent::SinkBackpressure { outputs_dropped, .. }) => {
                bp_events += 1;
                total_dropped_reported += outputs_dropped;
            }
            Ok(_) => continue,
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    assert!(bp_events > 0, "should see at least one SinkBackpressure event");
    assert!(
        bp_events < frame_count / 2,
        "throttling should coalesce SinkBackpressure events: got {bp_events} for {frame_count} frames"
    );
    assert!(
        total_dropped_reported > 0,
        "coalesced events should carry accumulated drop counts"
    );

    runtime.shutdown().unwrap();
}
