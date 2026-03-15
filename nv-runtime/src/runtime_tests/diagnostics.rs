//! Diagnostics and telemetry tests: terminal FeedStopped coherence,
//! runtime/feed uptime, queue telemetry, startup liveness, and
//! composite diagnostics snapshots.

use super::super::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use nv_core::config::{CameraMode, SourceSpec};
use nv_core::id::StageId;
use nv_media::ingress::{FrameSink, IngressOptions, MediaIngress, MediaIngressFactory};
use nv_perception::{Stage, StageOutput};
use nv_test_util::mock_stage::NoOpStage;
use tokio::sync::broadcast;

use crate::shutdown::{RestartPolicy, RestartTrigger};

use super::harness::*;

// ---------------------------------------------------------------------------
// Terminal FeedStopped coherence
// ---------------------------------------------------------------------------

#[test]
fn terminal_stop_emits_exactly_one_feed_stopped() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::failing()))
        .health_capacity(256)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let (sink, _count) = CountingSink::new();
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

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive(), "feed should have stopped");

    // Small settling time for health events to flush.
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Count FeedStopped events for this feed.
    let feed_id = handle.id();
    let mut feed_stopped_count = 0u32;
    let mut stop_reasons = Vec::new();
    loop {
        match health_rx.try_recv() {
            Ok(HealthEvent::FeedStopped { feed_id: fid, ref reason }) if fid == feed_id => {
                feed_stopped_count += 1;
                stop_reasons.push(format!("{reason:?}"));
            }
            Ok(_) => continue,
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                panic!("health channel lagged by {n} — increase health_capacity");
            }
            Err(_) => break,
        }
    }

    assert_eq!(
        feed_stopped_count, 1,
        "expected exactly one FeedStopped event, got {feed_stopped_count}: {stop_reasons:?}",
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Runtime uptime
// ---------------------------------------------------------------------------

#[test]
fn runtime_uptime_monotonic() {
    let runtime = build_runtime(0);

    let t1 = runtime.uptime();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let t2 = runtime.uptime();
    assert!(t2 > t1, "uptime should be monotonically increasing");

    let handle = runtime.handle();
    let t3 = handle.uptime();
    assert!(t3 >= t2, "handle uptime should be >= previous runtime uptime");

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Feed uptime (session-scoped)
// ---------------------------------------------------------------------------

#[test]
fn feed_uptime_advances_while_running() {
    let runtime = build_runtime(100);

    let (sink, _count) = CountingSink::new();
    let feed = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    // Let the feed process a few frames.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let u1 = feed.uptime();
    std::thread::sleep(std::time::Duration::from_millis(30));
    let u2 = feed.uptime();
    assert!(u2 > u1, "feed uptime should advance: u1={u1:?} u2={u2:?}");

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Queue telemetry
// ---------------------------------------------------------------------------

#[test]
fn queue_telemetry_reports_capacity() {
    let runtime = build_runtime(50);

    let (sink, _count) = CountingSink::new();
    let feed = runtime
        .add_feed(
            FeedConfig::builder()
                .source(SourceSpec::rtsp("rtsp://mock/stream"))
                .camera_mode(CameraMode::Fixed)
                .stages(vec![Box::new(NoOpStage::new("noop")) as Box<dyn Stage>])
                .output_sink(Box::new(sink))
                .backpressure(crate::BackpressurePolicy::DropOldest { queue_depth: 8 })
                .sink_queue_capacity(12)
                .build()
                .unwrap(),
        )
        .unwrap();

    // Give the feed time to start.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let qt = feed.queue_telemetry();
    assert_eq!(qt.source_capacity, 8, "source capacity should match configured backpressure");
    assert_eq!(qt.sink_capacity, 12, "sink capacity should match configured value");
    assert!(qt.source_depth <= qt.source_capacity);
    assert!(qt.sink_depth <= qt.sink_capacity);

    runtime.shutdown().unwrap();
}

#[test]
fn queue_telemetry_after_shutdown_is_zero() {
    let runtime = build_runtime(3);

    let (sink, _count) = CountingSink::new();
    let feed = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&feed, std::time::Duration::from_secs(5));

    let qt = feed.queue_telemetry();
    assert_eq!(qt.source_depth, 0, "source depth should be 0 after feed stopped");

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Startup liveness: source with tick hint but no frames must not hang
// ---------------------------------------------------------------------------

/// Mock source that emits zero frames but provides a tick hint.
struct TickHintNoFrameIngress {
    feed_id: FeedId,
    spec: SourceSpec,
    ticks: std::sync::atomic::AtomicU32,
}

impl MediaIngress for TickHintNoFrameIngress {
    fn start(&mut self, _sink: Box<dyn FrameSink>) -> Result<(), nv_core::error::MediaError> {
        Ok(())
    }
    fn stop(&mut self) -> Result<(), nv_core::error::MediaError> { Ok(()) }
    fn pause(&mut self) -> Result<(), nv_core::error::MediaError> { Ok(()) }
    fn resume(&mut self) -> Result<(), nv_core::error::MediaError> { Ok(()) }

    fn tick(&mut self) -> nv_media::ingress::TickOutcome {
        let n = self.ticks.fetch_add(1, Ordering::Relaxed);
        if n >= 3 {
            nv_media::ingress::TickOutcome::stopped()
        } else {
            nv_media::ingress::TickOutcome::reconnecting(
                std::time::Duration::from_millis(10),
            )
        }
    }

    fn source_spec(&self) -> &SourceSpec { &self.spec }
    fn feed_id(&self) -> FeedId { self.feed_id }
}

struct TickHintNoFrameFactory;
impl MediaIngressFactory for TickHintNoFrameFactory {
    fn create(
        &self,
        options: IngressOptions,
    ) -> Result<Box<dyn MediaIngress>, nv_core::error::MediaError> {
        Ok(Box::new(TickHintNoFrameIngress {
            feed_id: options.feed_id,
            spec: options.spec,
            ticks: std::sync::atomic::AtomicU32::new(0),
        }))
    }
}

#[test]
fn startup_liveness_tick_hint_no_frames() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(TickHintNoFrameFactory))
        .build()
        .unwrap();

    let (sink, _count) = CountingSink::new();
    let feed = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&feed, std::time::Duration::from_secs(5));
    assert!(!feed.is_alive(), "feed should have stopped");

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Diagnostics snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn feed_diagnostics_returns_composite_snapshot() {
    let runtime = build_runtime(5);
    let (sink, _count) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("diag"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 0,
                restart_on: RestartTrigger::Never,
                ..Default::default()
            },
        ))
        .unwrap();

    // Let the feed process some frames while alive.
    std::thread::sleep(std::time::Duration::from_millis(100));

    let diag = handle.diagnostics();
    assert_eq!(diag.feed_id, handle.id());
    assert_eq!(diag.view.status, crate::diagnostics::ViewStatus::Stable);
    assert!((diag.view.stability_score - 1.0).abs() < f32::EPSILON);

    wait_for_stop(&handle, std::time::Duration::from_secs(10));

    let diag2 = handle.diagnostics();
    assert!(!diag2.alive, "feed should be dead after EOS");
    assert!(diag2.metrics.frames_processed > 0);

    runtime.shutdown().unwrap();
}

#[test]
fn runtime_diagnostics_includes_all_feeds() {
    let runtime = Runtime::builder()
        .max_feeds(4)
        .ingress_factory(Box::new(MockFactory::new(100)))
        .build()
        .unwrap();

    let (s1, _) = CountingSink::new();
    let (s2, _) = CountingSink::new();
    let h1 = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("d1"))],
            Box::new(s1),
        ))
        .unwrap();
    let h2 = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("d2"))],
            Box::new(s2),
        ))
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(100));

    let diag = runtime.diagnostics().unwrap();
    assert_eq!(diag.feed_count, 2);
    assert_eq!(diag.feeds.len(), 2);
    assert!(diag.max_feeds == 4);
    assert!(diag.uptime >= std::time::Duration::from_millis(50));

    let ids: std::collections::HashSet<_> = diag.feeds.iter().map(|f| f.feed_id).collect();
    assert!(ids.contains(&h1.id()));
    assert!(ids.contains(&h2.id()));

    for f in &diag.feeds {
        assert!(f.alive, "feed {} should be alive", f.feed_id);
    }

    runtime.shutdown().unwrap();
}

#[test]
fn runtime_diagnostics_reflects_feed_removal() {
    let runtime = build_runtime(1000);

    let (s1, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("rem"))],
            Box::new(s1),
        ))
        .unwrap();
    let feed_id = handle.id();

    let diag1 = runtime.diagnostics().unwrap();
    assert_eq!(diag1.feed_count, 1);

    runtime.remove_feed(feed_id).unwrap();

    let diag2 = runtime.diagnostics().unwrap();
    assert_eq!(diag2.feed_count, 0);
    assert!(diag2.feeds.is_empty());

    runtime.shutdown().unwrap();
}

#[test]
fn feed_diagnostics_shows_paused_state() {
    let runtime = build_runtime(1000);
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("pause"))],
            Box::new(sink),
        ))
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    assert!(!handle.diagnostics().paused);
    handle.pause().unwrap();
    assert!(handle.diagnostics().paused);
    handle.resume().unwrap();
    assert!(!handle.diagnostics().paused);

    runtime.shutdown().unwrap();
}

#[test]
fn diagnostics_available_via_runtime_handle() {
    let runtime = build_runtime(100);
    let rh = runtime.handle();

    let (sink, _) = CountingSink::new();
    rh.add_feed(build_config(
        vec![Box::new(NoOpStage::new("rh"))],
        Box::new(sink),
    ))
    .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    let diag = rh.diagnostics().unwrap();
    assert_eq!(diag.feed_count, 1);
    assert!(diag.feeds[0].alive);

    runtime.shutdown().unwrap();
}

#[test]
fn runtime_diagnostics_includes_batch_coordinators() {
    use nv_core::error::StageError;
    use nv_perception::batch::{BatchEntry, BatchProcessor};

    struct TestProcessor;
    impl BatchProcessor for TestProcessor {
        fn id(&self) -> StageId {
            StageId("test_batch")
        }
        fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
            for item in items.iter_mut() {
                item.output = Some(StageOutput::empty());
            }
            Ok(())
        }
    }

    let runtime = build_runtime(100);
    let batch_handle = runtime
        .create_batch(
            Box::new(TestProcessor),
            BatchConfig {
                max_batch_size: 4,
                max_latency: std::time::Duration::from_millis(50),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        )
        .unwrap();

    let (sink, _count) = CountingSink::new();
    let pipeline = crate::FeedPipeline::builder()
        .batch(batch_handle.clone())
        .unwrap()
        .build();
    let feed_handle = runtime
        .add_feed(
            FeedConfig::builder()
                .source(SourceSpec::rtsp("rtsp://mock/stream"))
                .camera_mode(CameraMode::Fixed)
                .feed_pipeline(pipeline)
                .output_sink(Box::new(sink))
                .build()
                .expect("valid config"),
        )
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(200));

    let diag = runtime.diagnostics().unwrap();

    assert_eq!(diag.batches.len(), 1);
    assert_eq!(diag.batches[0].processor_id, StageId("test_batch"));
    assert!(
        diag.batches[0].metrics.items_submitted > 0,
        "batch should have received submissions"
    );

    let feed_diag = diag
        .feeds
        .iter()
        .find(|f| f.feed_id == feed_handle.id())
        .expect("feed should be in diagnostics");
    assert_eq!(
        feed_diag.batch_processor_id,
        Some(StageId("test_batch"))
    );

    runtime.shutdown().unwrap();
}

#[test]
fn create_batch_rejects_duplicate_processor_id() {
    use nv_core::error::StageError;
    use nv_perception::batch::{BatchEntry, BatchProcessor};

    struct DupProc;
    impl BatchProcessor for DupProc {
        fn id(&self) -> StageId {
            StageId("dup_id")
        }
        fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
            for item in items.iter_mut() {
                item.output = Some(StageOutput::empty());
            }
            Ok(())
        }
    }

    let runtime = build_runtime(100);
    let cfg = BatchConfig {
        max_batch_size: 4,
        max_latency: std::time::Duration::from_millis(50),
        queue_capacity: None,
        response_timeout: None,
        max_in_flight_per_feed: 1,
    };

    let _h1 = runtime.create_batch(Box::new(DupProc), cfg.clone()).unwrap();
    let result = runtime.create_batch(Box::new(DupProc), cfg);
    let err = result.err().expect("second create_batch should fail");
    match &err {
        NvError::Config(nv_core::error::ConfigError::DuplicateBatchProcessorId { id }) => {
            assert_eq!(id.0, "dup_id", "error should carry the duplicate id");
        }
        other => panic!("expected DuplicateBatchProcessorId, got: {other:?}"),
    }

    runtime.shutdown().unwrap();
}

#[test]
fn create_batch_concurrent_duplicate_id_exactly_one_wins() {
    use nv_core::error::StageError;
    use nv_perception::batch::{BatchEntry, BatchProcessor};
    use std::sync::Barrier;

    struct RaceProc;
    impl BatchProcessor for RaceProc {
        fn id(&self) -> StageId {
            StageId("race_id")
        }
        fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
            for item in items.iter_mut() {
                item.output = Some(StageOutput::empty());
            }
            Ok(())
        }
    }

    let runtime = Arc::new(build_runtime(100));
    let cfg = BatchConfig {
        max_batch_size: 4,
        max_latency: std::time::Duration::from_millis(50),
        queue_capacity: None,
        response_timeout: None,
        max_in_flight_per_feed: 1,
    };

    const THREADS: usize = 8;
    let barrier = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let rt = Arc::clone(&runtime);
            let c = cfg.clone();
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || {
                b.wait();
                rt.create_batch(Box::new(RaceProc), c)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let ok_count = results.iter().filter(|r| r.is_ok()).count();
    let dup_count = results.iter().filter(|r| {
        matches!(
            r,
            Err(NvError::Config(
                nv_core::error::ConfigError::DuplicateBatchProcessorId { .. }
            ))
        )
    }).count();

    assert_eq!(ok_count, 1, "exactly one caller should succeed");
    assert_eq!(dup_count, THREADS - 1, "all others should get DuplicateBatchProcessorId");

    Arc::try_unwrap(runtime)
        .ok()
        .expect("all threads finished — Arc should be unique")
        .shutdown()
        .unwrap();
}

#[test]
fn runtime_diagnostics_output_lag_default_is_clean() {
    let runtime = build_runtime(5);
    let (sink, _) = CountingSink::new();
    runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("lag"))],
            Box::new(sink),
        ))
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    let diag = runtime.diagnostics().unwrap();
    assert!(!diag.output_lag.in_lag);
    assert_eq!(diag.output_lag.pending_lost, 0);

    runtime.shutdown().unwrap();
}

#[test]
fn runtime_diagnostics_output_lag_reflects_saturation() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 60,
            fail_on_start: false,
            frame_delay: std::time::Duration::ZERO,
        }))
        .output_capacity(2)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();

    let _output_rx = runtime.output_subscribe();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("sat"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let _diag = runtime.diagnostics().unwrap();

    let mut total_lost: u64 = 0;
    loop {
        match health_rx.try_recv() {
            Ok(HealthEvent::OutputLagged { messages_lost }) => {
                total_lost += messages_lost;
            }
            Ok(_) => {}
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }
    assert!(
        total_lost > 0,
        "health events should report lag during saturation"
    );

    runtime.shutdown().unwrap();
}

#[test]
fn feed_diagnostics_batch_processor_id_none_without_batch() {
    let runtime = build_runtime(100);
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("solo"))],
            Box::new(sink),
        ))
        .unwrap();

    let diag = handle.diagnostics();
    assert!(
        diag.batch_processor_id.is_none(),
        "non-batched feed should have no batch_processor_id"
    );

    runtime.shutdown().unwrap();
}

#[test]
fn runtime_diagnostics_feeds_sorted_by_id() {
    let runtime = Runtime::builder()
        .max_feeds(10)
        .ingress_factory(Box::new(MockFactory::new(1000)))
        .build()
        .unwrap();

    let mut handles = Vec::new();
    let stage_names: [&'static str; 4] = ["s0", "s1", "s2", "s3"];
    for name in &stage_names {
        let (s, _) = CountingSink::new();
        let h = runtime
            .add_feed(build_config(
                vec![Box::new(NoOpStage::new(name))],
                Box::new(s),
            ))
            .unwrap();
        handles.push(h);
    }

    let diag = runtime.diagnostics().unwrap();
    assert_eq!(diag.feeds.len(), 4);

    for w in diag.feeds.windows(2) {
        assert!(
            w[0].feed_id.as_u64() < w[1].feed_id.as_u64(),
            "feeds should be sorted by FeedId: {} >= {}",
            w[0].feed_id,
            w[1].feed_id
        );
    }

    runtime.shutdown().unwrap();
}

/// Regression test: `create_batch` inside `on_start` when `shutdown_all`
/// runs concurrently must return `ShutdownInProgress`.
#[test]
fn create_batch_during_shutdown_returns_shutdown_error() {
    use nv_core::error::StageError;
    use nv_perception::batch::{BatchEntry, BatchProcessor};
    use std::sync::Condvar;

    struct Gate {
        entered: Mutex<bool>,
        release: Condvar,
    }

    struct SlowStartProc {
        gate: Arc<Gate>,
    }

    impl BatchProcessor for SlowStartProc {
        fn id(&self) -> StageId {
            StageId("slow_start")
        }
        fn on_start(&mut self) -> Result<(), StageError> {
            let mut entered = self.gate.entered.lock().unwrap();
            *entered = true;
            self.gate.release.notify_all();
            let _guard = self
                .gate
                .release
                .wait_while(entered, |e| *e)
                .unwrap();
            Ok(())
        }
        fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
            for item in items.iter_mut() {
                item.output = Some(StageOutput::empty());
            }
            Ok(())
        }
    }

    let gate = Arc::new(Gate {
        entered: Mutex::new(false),
        release: Condvar::new(),
    });

    let runtime = build_runtime(100);
    let handle = runtime.handle();
    let cfg = BatchConfig {
        max_batch_size: 4,
        max_latency: std::time::Duration::from_millis(50),
        queue_capacity: None,
        response_timeout: None,
        max_in_flight_per_feed: 1,
    };

    let gate_clone = Arc::clone(&gate);
    let handle_clone = handle.clone();
    let batch_thread = std::thread::spawn(move || {
        handle_clone.create_batch(
            Box::new(SlowStartProc { gate: gate_clone }),
            cfg,
        )
    });

    // Wait for on_start to be entered on the coordinator thread.
    {
        let entered = gate.entered.lock().unwrap();
        let _guard = gate
            .release
            .wait_while(entered, |e| !*e)
            .unwrap();
    }

    handle.shutdown().ok();

    // Release on_start so create_batch can finish its post-insert check.
    {
        let mut entered = gate.entered.lock().unwrap();
        *entered = false;
        gate.release.notify_all();
        drop(entered);
    }

    let result = batch_thread.join().expect("batch thread should not panic");
    match &result {
        Err(NvError::Runtime(RuntimeError::ShutdownInProgress)) => {
            // Expected.
        }
        Err(e) => panic!(
            "expected ShutdownInProgress after concurrent shutdown, got error: {e}"
        ),
        Ok(_) => panic!(
            "expected ShutdownInProgress after concurrent shutdown, but create_batch succeeded"
        ),
    }
}
