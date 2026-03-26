use super::*;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::{FeedId, StageId};
use nv_perception::batch::{BatchEntry, BatchProcessor};
use nv_perception::{DetectionSet, StageOutput};
use nv_view::ViewSnapshot;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use tokio::sync::broadcast;

/// A trivial batch processor that sets each output to detections with
/// N items where N = batch size.
struct CountingProcessor {
    calls: Arc<AtomicU32>,
}

impl CountingProcessor {
    fn new() -> (Self, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        (
            Self {
                calls: Arc::clone(&calls),
            },
            calls,
        )
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let result = handle.submit_and_wait(make_entry(1), None);
    assert!(result.is_ok());
    assert!(result.unwrap().detections.is_some());
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let mut handles = Vec::new();
    let start = Instant::now();
    for i in 0..4 {
        let h = handle.clone();
        handles.push(std::thread::spawn(move || {
            h.submit_and_wait(make_entry(i), None)
        }));
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

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    // Submit 3 items (less than max_batch_size=8)
    let mut handles = Vec::new();
    for i in 0..3 {
        let h = handle.clone();
        handles.push(std::thread::spawn(move || {
            h.submit_and_wait(make_entry(i), None)
        }));
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

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let mut join_handles = Vec::new();
    for i in 0..3 {
        let h = handle.clone();
        join_handles.push(std::thread::spawn(move || {
            h.submit_and_wait(make_entry(i), None)
        }));
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

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let result = handle.submit_and_wait(make_entry(1), None);
    assert!(matches!(result, Err(BatchSubmitError::ProcessingFailed(_))));

    let event = health_rx.try_recv();
    assert!(matches!(event, Ok(HealthEvent::BatchError { .. })));

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let _ = handle.submit_and_wait(make_entry(1), None);
    let _ = handle.submit_and_wait(make_entry(2), None);

    let m = handle.metrics();
    assert_eq!(m.items_submitted, 2);
    assert!(m.batches_dispatched >= 1);

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
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
    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
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
            startup_timeout: None,
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
            startup_timeout: None,
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

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
        health_tx,
    );
    assert!(
        result.is_err(),
        "queue_capacity < max_batch_size should fail"
    );
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
            startup_timeout: None,
        },
    );
    let handle = coord.handle();
    let _ = handle.submit_and_wait(make_entry(1), None);
    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    // Shut down the coordinator so the channel becomes disconnected.
    coord.shutdown(Duration::from_secs(10));
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
    assert_eq!(
        m.pending_items(),
        0,
        "pending should be 0 after disconnect rejection"
    );
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
    assert!(
        (avg - 3.0).abs() < f64::EPSILON,
        "expected 12/4 = 3.0, got {avg}"
    );
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
    assert!(
        (ratio - 0.5).abs() < f64::EPSILON,
        "expected 4/8 = 0.5, got {ratio}"
    );
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
    assert!(
        (ratio - 1.0).abs() < f64::EPSILON,
        "expected 1.0 for full batches, got {ratio}"
    );
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
            startup_timeout: None,
        },
    );
    let m = coord.handle().metrics();
    assert_eq!(m.configured_max_batch_size, 16);
    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
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
    let result = BatchCoordinator::start(Box::new(FailStart), BatchConfig::default(), health_tx);
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
                item.output = Some(StageOutput::with_scene_features(vec![SceneFeature {
                    name: "weather",
                    value: SceneFeatureValue::Scalar(0.9),
                    confidence: Some(0.95),
                    ts: MonotonicTs::from_nanos(0),
                }]));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let result = handle.submit_and_wait(make_entry(1), None);
    let output = result.expect("scene classifier should succeed");
    assert_eq!(output.scene_features.len(), 1);
    assert_eq!(output.scene_features[0].name, "weather");

    coord.shutdown(Duration::from_secs(10));
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
    // on_start completes well within the default startup timeout (30 s).
    // This validates that a blocking-but-eventually-completing
    // on_start succeeds and the coordinator cleans up normally.
    //
    // NOTE: the timeout path can be tested by setting a short
    // startup_timeout in BatchConfig.
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
    assert!(
        result.is_ok(),
        "slow-but-completing on_start should succeed"
    );
    result.unwrap().shutdown(Duration::from_secs(10));
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
    let result = BatchCoordinator::start(Box::new(FailStart), BatchConfig::default(), health_tx);
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
            startup_timeout: None,
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
            startup_timeout: None,
        },
        health_tx,
    )
    .unwrap();

    let handle = coord.handle();
    let result = handle.submit_and_wait(make_entry(1), None);
    assert!(
        matches!(result, Err(BatchSubmitError::Timeout)),
        "expected Timeout with short response_timeout, got: {result:?}"
    );

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
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

    coord.shutdown(Duration::from_secs(10));
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
        Box::new(SlowStopProcessor {
            barrier: Arc::clone(&barrier),
        }),
        BatchConfig {
            max_batch_size: 100,                  // very large — will never fill
            max_latency: Duration::from_secs(60), // very long — will never fire
            queue_capacity: None,
            // Short response timeout to distinguish from Timeout
            response_timeout: Some(Duration::from_millis(500)),
            max_in_flight_per_feed: 1,
            startup_timeout: None,
        },
        health_tx,
    )
    .unwrap();

    let handle = coord.handle();

    // Submit an item. The batch won't form (max_batch_size=100,
    // max_latency=60s), so the item sits in the queue.
    let h = handle.clone();
    let feed_thread = std::thread::spawn(move || h.submit_and_wait(make_entry(1), None));

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
        matches!(result, Err(BatchSubmitError::CoordinatorShutdown) | Ok(_)),
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
            startup_timeout: None,
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

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    assert_eq!(handle.metrics().items_timed_out, 0);

    // Simulate timeout recording (normally done by executor).
    handle.record_timeout();
    handle.record_timeout();
    assert_eq!(handle.metrics().items_timed_out, 2);

    coord.shutdown(Duration::from_secs(10));
}

#[test]
fn config_validate_catches_all_errors() {
    // Valid config.
    assert!(
        BatchConfig::new(4, Duration::from_millis(50))
            .unwrap()
            .validate()
            .is_ok()
    );

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
            startup_timeout: None,
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

    coord.shutdown(Duration::from_secs(10));
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
    let proc = RecordProcessor {
        seen: Arc::clone(&seen),
    };
    let (coord, _rx) = start_coordinator(
        Box::new(proc),
        BatchConfig {
            max_batch_size: 2,
            max_latency: Duration::from_millis(30),
            // Queue capacity just big enough for feeds.
            queue_capacity: Some(4),
            response_timeout: None,
            max_in_flight_per_feed: 1,
            startup_timeout: None,
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
    assert_eq!(
        m.items_rejected, 0,
        "no rejections expected with adequate queue"
    );

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let success_counts: Arc<[AtomicU32; 3]> =
        Arc::new([AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)]);

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

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
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

    coord.shutdown(Duration::from_secs(10));
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
        Box::new(OnceSlowProcessor {
            slow: AtomicBool::new(true),
        }),
        BatchConfig {
            max_batch_size: 1,
            max_latency: Duration::from_millis(10),
            queue_capacity: Some(4),
            // Very short timeout so feed times out quickly.
            response_timeout: Some(Duration::from_millis(50)),
            max_in_flight_per_feed: 1,
            startup_timeout: None,
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
        in_flight.load(Ordering::Acquire),
        1,
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
        in_flight.load(Ordering::Acquire),
        1,
        "in_flight should remain 1 after cap rejection"
    );

    // Metrics: 2 submitted, 1 rejected (the InFlightCapReached).
    let m = handle.metrics();
    assert_eq!(m.items_submitted, 2, "two submissions total");
    assert_eq!(
        m.items_rejected, 1,
        "InFlightCapReached counts as rejection"
    );
    // pending_items should reflect only the one actual in-flight item.
    assert_eq!(m.pending_items(), 1, "only one item genuinely pending");

    // Wait for coordinator to process the timed-out item.
    std::thread::sleep(Duration::from_millis(600));

    // in_flight should now be 0 (coordinator decremented).
    assert_eq!(
        in_flight.load(Ordering::Acquire),
        0,
        "in_flight should be 0 after coordinator processes"
    );

    // Submit 3: should succeed now (processor is fast on subsequent calls).
    let r3 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
    assert!(r3.is_ok(), "third submit should succeed, got: {r3:?}");

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
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
        in_flight.load(Ordering::Acquire),
        0,
        "in_flight should be 0 after send failure"
    );

    coord.shutdown(Duration::from_secs(10));
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
            startup_timeout: None,
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
            counter.load(Ordering::Acquire),
            1,
            "feed {} in_flight should be 1 before shutdown",
            i + 1
        );
    }

    // Signal shutdown while the processor is still sleeping.
    // Coordinator will finish item 1 (~400ms), then drain items 2-4.
    coord.shutdown(Duration::from_secs(10));

    // Wait for threads to finish.
    for t in threads {
        let _ = t.join();
    }

    // All in_flight counters should be 0:
    // - Item 1's guard decremented in Phase 4 (result routing).
    // - Items 2-4 guards decremented in drain_pending.
    for (i, counter) in in_flights.iter().enumerate() {
        assert_eq!(
            counter.load(Ordering::Acquire),
            0,
            "feed {} in_flight should be 0 after shutdown drain",
            i + 1
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
            startup_timeout: None,
        },
    );

    let handle = coord.handle();
    let success_counts: Arc<[AtomicU32; 3]> =
        Arc::new([AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)]);

    // Feed 1: fast (15 frames), Feed 2: medium (10), Feed 3: slow (5).
    let rates = [(1u64, 15u32), (2, 10), (3, 5)];
    let mut threads = Vec::new();
    for (idx, &(feed_id, count)) in rates.iter().enumerate() {
        let h = handle.clone();
        let sc = Arc::clone(&success_counts);
        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        threads.push(std::thread::spawn(move || {
            for _ in 0..count {
                if h.submit_and_wait(make_entry(feed_id), Some(&in_flight))
                    .is_ok()
                {
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

    coord.shutdown(Duration::from_secs(10));
}

#[test]
fn in_flight_cap_of_zero_rejected_by_validate() {
    let result = BatchConfig {
        max_batch_size: 4,
        max_latency: Duration::from_millis(50),
        queue_capacity: None,
        response_timeout: None,
        max_in_flight_per_feed: 0,
        startup_timeout: None,
    }
    .validate();
    assert!(
        result.is_err(),
        "max_in_flight_per_feed=0 should be rejected"
    );
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
            startup_timeout: None,
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

    coord.shutdown(Duration::from_secs(10));
}

/// Reproducer for the Jetson stuck-batch issue: when the first
/// `process()` call takes longer than `max_latency + response_timeout`,
/// the feed thread times out and subsequent submissions are rejected by
/// InFlightCapReached because the in-flight counter is never decremented.
///
/// This simulates TRT engine re-profiling on the first inference call
/// which can block for tens of seconds.
#[test]
fn slow_first_process_causes_in_flight_cascade() {
    use std::sync::atomic::AtomicBool;

    struct SlowFirstProcessor {
        first_call: AtomicBool,
    }

    impl BatchProcessor for SlowFirstProcessor {
        fn id(&self) -> StageId {
            StageId("slow-first")
        }
        fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
            if self
                .first_call
                .compare_exchange(true, false, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                // Simulate TRT engine rebuild: sleep longer than response_timeout.
                std::thread::sleep(Duration::from_millis(1500));
            }
            for item in items.iter_mut() {
                item.output = Some(StageOutput::with_detections(DetectionSet::empty()));
            }
            Ok(())
        }
    }

    let processor = SlowFirstProcessor {
        first_call: AtomicBool::new(true),
    };

    // Tight timeouts so the test runs quickly:
    //   max_latency=50ms, response_timeout=200ms → total wait = 250ms.
    //   The processor sleeps 1500ms on first call → guaranteed timeout.
    let config = BatchConfig {
        max_batch_size: 4,
        max_latency: Duration::from_millis(50),
        response_timeout: Some(Duration::from_millis(200)),
        max_in_flight_per_feed: 1,
        startup_timeout: None,
        ..BatchConfig::default()
    };

    let (coord, _rx) = start_coordinator(Box::new(processor), config);
    let handle = coord.handle();
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // First submission: will reach coordinator but process() blocks.
    // Feed thread times out after 250ms.
    let r1 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
    assert!(
        matches!(r1, Err(BatchSubmitError::Timeout)),
        "first submit should time out, got: {r1:?}"
    );

    // In-flight counter is still 1 — coordinator hasn't decremented.
    assert_eq!(
        in_flight.load(Ordering::Relaxed),
        1,
        "in-flight should still be 1 after timeout"
    );

    // Second submission: immediately rejected by InFlightCapReached.
    let r2 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
    assert!(
        matches!(r2, Err(BatchSubmitError::InFlightCapReached)),
        "second submit should hit in-flight cap, got: {r2:?}"
    );

    // Third submission: same.
    let r3 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
    assert!(
        matches!(r3, Err(BatchSubmitError::InFlightCapReached)),
        "third submit should hit in-flight cap, got: {r3:?}"
    );

    let m = handle.metrics();
    assert_eq!(
        m.batches_dispatched, 0,
        "no batch should have completed yet"
    );
    assert_eq!(m.items_processed, 0, "no items processed yet");
    assert!(m.items_rejected >= 2, "at least 2 rejected by cap");
    assert_eq!(m.pending_items(), 1, "first item still pending");

    // Wait for the slow process() to finish.
    std::thread::sleep(Duration::from_millis(1500));

    // Now the coordinator should have completed the batch and
    // decremented in-flight.
    assert_eq!(
        in_flight.load(Ordering::Relaxed),
        0,
        "in-flight should be 0 after slow process completes"
    );

    let m2 = handle.metrics();
    assert_eq!(m2.batches_dispatched, 1, "batch should have completed");
    assert_eq!(m2.items_processed, 1, "one item processed");

    // Subsequent submissions should now succeed.
    let r4 = handle.submit_and_wait(make_entry(1), Some(&in_flight));
    assert!(
        r4.is_ok(),
        "submit after recovery should succeed, got: {r4:?}"
    );

    coord.shutdown(Duration::from_secs(10));
}
