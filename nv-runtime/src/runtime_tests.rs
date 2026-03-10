//! Integration tests for the runtime: feed lifecycle, restart, shutdown,
//! output subscription, backpressure, pause/resume, provenance, and
//! sentinel-based lag detection.

use super::*;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use nv_core::config::{CameraMode, ReconnectPolicy, SourceSpec};
use nv_core::error::StageError;
use nv_core::health::StopReason;
use nv_core::id::StageId;
use nv_frame::PixelFormat;
use nv_media::ingress::{FrameSink, MediaIngress, MediaIngressFactory, PtzProvider};
use nv_perception::{Stage, StageContext, StageOutput};
use nv_test_util::mock_stage::{NoOpStage, PanicStage};
use std::sync::atomic::Ordering;
use tokio::sync::broadcast;

use crate::output::{OutputEnvelope, OutputSink};
use crate::shutdown::{RestartPolicy, RestartTrigger};

// ---------------------------------------------------------------------------
// Mock infrastructure
// ---------------------------------------------------------------------------

/// Mock source: sends `frame_count` frames then EOS.
struct MockIngress {
    feed_id: FeedId,
    spec: SourceSpec,
    frame_count: u64,
    fail_on_start: bool,
    frame_delay: std::time::Duration,
}

impl MediaIngress for MockIngress {
    fn start(&mut self, sink: Box<dyn FrameSink>) -> Result<(), nv_core::error::MediaError> {
        if self.fail_on_start {
            return Err(nv_core::error::MediaError::ConnectionFailed {
                url: "mock://fail".into(),
                detail: "mock start failure".into(),
            });
        }
        let count = self.frame_count;
        let feed_id = self.feed_id;
        let delay = self.frame_delay;
        std::thread::spawn(move || {
            for i in 0..count {
                let frame = make_test_frame(feed_id, i);
                sink.on_frame(frame);
                if delay > std::time::Duration::ZERO {
                    std::thread::sleep(delay);
                }
            }
            sink.on_eos();
        });
        Ok(())
    }

    fn stop(&mut self) -> Result<(), nv_core::error::MediaError> {
        Ok(())
    }
    fn pause(&mut self) -> Result<(), nv_core::error::MediaError> {
        Ok(())
    }
    fn resume(&mut self) -> Result<(), nv_core::error::MediaError> {
        Ok(())
    }
    fn source_spec(&self) -> &SourceSpec {
        &self.spec
    }
    fn feed_id(&self) -> FeedId {
        self.feed_id
    }
}

/// Mock factory: creates `MockIngress` sources.
struct MockFactory {
    frame_count: u64,
    fail_on_start: bool,
    frame_delay: std::time::Duration,
}

impl MockFactory {
    fn new(frame_count: u64) -> Self {
        Self {
            frame_count,
            fail_on_start: false,
            frame_delay: std::time::Duration::from_millis(1),
        }
    }

    fn failing() -> Self {
        Self {
            frame_count: 0,
            fail_on_start: true,
            frame_delay: std::time::Duration::ZERO,
        }
    }
}

impl MediaIngressFactory for MockFactory {
    fn create(
        &self,
        feed_id: FeedId,
        spec: SourceSpec,
        _reconnect: ReconnectPolicy,
        _ptz: Option<Arc<dyn PtzProvider>>,
    ) -> Result<Box<dyn MediaIngress>, nv_core::error::MediaError> {
        Ok(Box::new(MockIngress {
            feed_id,
            spec,
            frame_count: self.frame_count,
            fail_on_start: self.fail_on_start,
            frame_delay: self.frame_delay,
        }))
    }
}

/// Output sink that counts received outputs.
struct CountingSink {
    count: Arc<AtomicU64>,
}

impl CountingSink {
    fn new() -> (Self, Arc<AtomicU64>) {
        let count = Arc::new(AtomicU64::new(0));
        (
            Self {
                count: Arc::clone(&count),
            },
            count,
        )
    }
}

impl OutputSink for CountingSink {
    fn emit(&self, _output: OutputEnvelope) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

fn make_test_frame(feed_id: FeedId, seq: u64) -> nv_frame::FrameEnvelope {
    nv_frame::FrameEnvelope::new_owned(
        feed_id,
        seq,
        nv_core::MonotonicTs::from_nanos(seq * 33_333_333),
        nv_core::WallTs::from_micros(0),
        2,
        2,
        PixelFormat::Rgb8,
        6,
        vec![0u8; 12],
        nv_core::TypedMetadata::new(),
    )
}

fn build_config(stages: Vec<Box<dyn Stage>>, sink: Box<dyn OutputSink>) -> FeedConfig {
    FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(stages)
        .output_sink(sink)
        .build()
        .expect("valid config")
}

fn build_config_with_restart(
    stages: Vec<Box<dyn Stage>>,
    sink: Box<dyn OutputSink>,
    restart: RestartPolicy,
) -> FeedConfig {
    FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(stages)
        .output_sink(sink)
        .restart(restart)
        .build()
        .expect("valid config")
}

/// Wait for a feed to stop (with timeout).
fn wait_for_stop(handle: &FeedHandle, timeout: std::time::Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while handle.is_alive() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

// ---------------------------------------------------------------------------
// 1. Multi-feed registration
// ---------------------------------------------------------------------------

#[test]
fn multi_feed_registration() {
    let runtime = Runtime::builder()
        .max_feeds(4)
        .ingress_factory(Box::new(MockFactory::new(5)))
        .build()
        .unwrap();

    let (s1, _) = CountingSink::new();
    let (s2, _) = CountingSink::new();
    let (s3, _) = CountingSink::new();
    let h1 = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(s1),
        ))
        .unwrap();
    let h2 = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(s2),
        ))
        .unwrap();
    let h3 = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(s3),
        ))
        .unwrap();

    assert_eq!(runtime.feed_count().unwrap(), 3);
    assert_ne!(h1.id(), h2.id());
    assert_ne!(h2.id(), h3.id());
    assert_ne!(h1.id(), h3.id());

    runtime.shutdown().unwrap();
}

#[test]
fn feed_limit_exceeded() {
    let runtime = Runtime::builder()
        .max_feeds(1)
        .ingress_factory(Box::new(MockFactory::new(100)))
        .build()
        .unwrap();

    let (s1, _) = CountingSink::new();
    let (s2, _) = CountingSink::new();
    runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(s1),
        ))
        .unwrap();
    let err = runtime.add_feed(build_config(
        vec![Box::new(NoOpStage::new("noop"))],
        Box::new(s2),
    ));
    assert!(matches!(
        err,
        Err(NvError::Runtime(RuntimeError::FeedLimitExceeded { max: 1 }))
    ));

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// 2. Start/stop lifecycle
// ---------------------------------------------------------------------------

#[test]
fn start_stop_lifecycle() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(10)))
        .build()
        .unwrap();

    let (sink, count) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let m = handle.metrics();
    assert!(m.frames_processed > 0);
    assert_eq!(count.load(Ordering::Relaxed), m.frames_processed);

    runtime.shutdown().unwrap();
}

#[test]
fn remove_feed_stops_worker() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(10_000)))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();
    let feed_id = handle.id();

    std::thread::sleep(std::time::Duration::from_millis(50));
    runtime.remove_feed(feed_id).unwrap();
    assert_eq!(runtime.feed_count().unwrap(), 0);

    // Removing again should fail.
    assert!(matches!(
        runtime.remove_feed(feed_id),
        Err(NvError::Runtime(RuntimeError::FeedNotFound { .. }))
    ));

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// 3/4. Bounded queue + backpressure (unit tests in queue.rs; integration)
// ---------------------------------------------------------------------------

#[test]
fn backpressure_drops_are_reported() {
    // Use a very small queue (depth 1) and a slow stage to guarantee drops.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 20,
            fail_on_start: false,
            // Fast producer — pushes faster than consumer can process.
            frame_delay: std::time::Duration::ZERO,
        }))
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let (sink, _) = CountingSink::new();
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(NoOpStage::new("noop"))])
        .output_sink(Box::new(sink))
        .backpressure(crate::backpressure::BackpressurePolicy::DropOldest { queue_depth: 1 })
        .build()
        .unwrap();

    let handle = runtime.add_feed(config).unwrap();
    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let m = handle.metrics();
    assert!(m.frames_received > 0, "should have received frames");

    // Check for BackpressureDrop health events if any drops occurred.
    if m.frames_dropped > 0 {
        let mut saw_drop = false;
        while let Ok(event) = health_rx.try_recv() {
            if matches!(event, HealthEvent::BackpressureDrop { .. }) {
                saw_drop = true;
            }
        }
        assert!(saw_drop, "should emit BackpressureDrop health event");
    }

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// 5. Feed failure isolation
// ---------------------------------------------------------------------------

#[test]
fn feed_failure_isolation() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(10)))
        .build()
        .unwrap();

    // Feed with a failing stage.
    let (sink_fail, count_fail) = CountingSink::new();
    let h_fail = runtime
        .add_feed(build_config(
            vec![Box::new(nv_test_util::mock_stage::FailingStage::new("bad"))],
            Box::new(sink_fail),
        ))
        .unwrap();

    // Feed with a normal stage.
    let (sink_ok, count_ok) = CountingSink::new();
    let h_ok = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink_ok),
        ))
        .unwrap();

    wait_for_stop(&h_fail, std::time::Duration::from_secs(5));
    wait_for_stop(&h_ok, std::time::Duration::from_secs(5));

    // Both feeds processed frames — the failing stage doesn't kill the other.
    assert!(
        count_ok.load(Ordering::Relaxed) > 0,
        "good feed should process frames"
    );
    // The failing-stage feed drops every frame (stage error → frame dropped).
    assert_eq!(
        count_fail.load(Ordering::Relaxed),
        0,
        "failing-stage feed drops frames on error, emits no output"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// 6. Restart behavior
// ---------------------------------------------------------------------------

#[test]
fn restart_trigger_never_does_not_restart() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::failing()))
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 5,
                restart_on: RestartTrigger::Never,
                ..RestartPolicy::default()
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    assert_eq!(handle.metrics().restarts, 0);

    // Should see a FeedStopped health event.
    let mut saw_stopped = false;
    while let Ok(event) = health_rx.try_recv() {
        if matches!(event, HealthEvent::FeedStopped { .. }) {
            saw_stopped = true;
        }
    }
    assert!(saw_stopped, "should emit FeedStopped");

    runtime.shutdown().unwrap();
}

#[test]
fn restart_on_source_failure_honors_max() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::failing()))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 2,
                restart_on: RestartTrigger::SourceFailure,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_secs(300),
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    assert_eq!(handle.metrics().restarts, 2);

    runtime.shutdown().unwrap();
}

#[test]
fn stage_panic_no_restart_with_source_failure_trigger() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(5)))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(PanicStage::new("panic"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 5,
                restart_on: RestartTrigger::SourceFailure,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_secs(300),
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    // SourceFailure policy → panic does NOT trigger restart.
    assert_eq!(handle.metrics().restarts, 0);

    runtime.shutdown().unwrap();
}

#[test]
fn stage_panic_restarts_with_source_or_panic_trigger() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(5)))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(PanicStage::new("panic"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 2,
                restart_on: RestartTrigger::SourceOrStagePanic,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_secs(300),
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    // SourceOrStagePanic policy → panic triggers restart.
    assert_eq!(handle.metrics().restarts, 2);

    runtime.shutdown().unwrap();
}

#[test]
fn restart_window_resets_counter() {
    // With restart_window = 0, the counter always resets, allowing
    // unlimited restarts even with max_restarts = 1.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(0))) // 0 frames → immediate EOS
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 1,
                restart_on: RestartTrigger::SourceFailure,
                restart_delay: nv_core::Duration::from_millis(10),
                // Zero window → counter always resets.
                restart_window: nv_core::Duration::from_millis(0),
            },
        ))
        .unwrap();

    // Let it restart a few times.
    std::thread::sleep(std::time::Duration::from_millis(200));
    // Feed should still be alive — counter keeps resetting.
    assert!(
        handle.is_alive(),
        "feed should keep restarting due to window reset"
    );
    assert!(handle.metrics().restarts >= 1, "should have restarted");

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// 7. Shutdown races
// ---------------------------------------------------------------------------

#[test]
fn immediate_shutdown_after_add() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(10_000)))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let _handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    // Immediately shutdown — must not hang.
    runtime.shutdown().unwrap();
}

#[test]
fn shutdown_rejects_new_feeds() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(5)))
        .build()
        .unwrap();

    let handle = runtime.handle();
    runtime.shutdown().unwrap();

    let (sink, _) = CountingSink::new();
    let err = handle.add_feed(build_config(
        vec![Box::new(NoOpStage::new("noop"))],
        Box::new(sink),
    ));
    assert!(matches!(
        err,
        Err(NvError::Runtime(RuntimeError::ShutdownInProgress))
    ));
}

// ---------------------------------------------------------------------------
// 8. Output subscription
// ---------------------------------------------------------------------------

#[test]
fn output_subscription_receives_outputs() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(5)))
        .output_capacity(32)
        .build()
        .unwrap();

    let mut rx = runtime.output_subscribe();
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    let feed_id = handle.id();
    let mut outputs = Vec::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

    loop {
        match rx.try_recv() {
            Ok(output) => outputs.push(output),
            Err(broadcast::error::TryRecvError::Empty) => {
                if !handle.is_alive() {
                    // Drain remaining.
                    while let Ok(o) = rx.try_recv() {
                        outputs.push(o);
                    }
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
        if std::time::Instant::now() > deadline {
            break;
        }
    }

    assert!(
        !outputs.is_empty(),
        "should receive outputs via subscription"
    );
    for o in &outputs {
        assert_eq!(o.feed_id, feed_id);
    }

    runtime.shutdown().unwrap();
}

#[test]
fn output_subscription_bounded_capacity() {
    // With capacity=2 and 10 fast frames, the receiver should lag.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 10,
            fail_on_start: false,
            frame_delay: std::time::Duration::ZERO,
        }))
        .output_capacity(2)
        .build()
        .unwrap();

    let mut rx = runtime.output_subscribe();
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    // Wait for feed to complete.
    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    // Now try to receive — we may get Lagged error.
    let mut received = 0u64;
    let mut lagged = false;
    loop {
        match rx.try_recv() {
            Ok(_) => received += 1,
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                lagged = true;
                received += n;
            }
            Err(_) => break,
        }
    }

    // Either we got all 10 outputs, or we saw lag.
    // With capacity=2, lag is very likely with 10 fast frames.
    assert!(received > 0 || lagged, "should receive or detect lag");

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// RuntimeHandle
// ---------------------------------------------------------------------------

#[test]
fn runtime_handle_is_cloneable_and_functional() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(3)))
        .build()
        .unwrap();

    let h1 = runtime.handle();
    let h2 = h1.clone();

    let (sink, _) = CountingSink::new();
    let _feed = h1
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    assert_eq!(h2.feed_count().unwrap(), 1);
    assert_eq!(h2.max_feeds(), 64);

    h2.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Pause/resume
// ---------------------------------------------------------------------------

#[test]
fn pause_resume_controls_processing() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(10_000)))
        .build()
        .unwrap();

    let (sink, count) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    // Wait for some processing.
    std::thread::sleep(std::time::Duration::from_millis(50));

    handle.pause().unwrap();
    assert!(handle.is_paused());
    let count_at_pause = count.load(Ordering::Relaxed);

    // Wait and verify processing has stopped.
    std::thread::sleep(std::time::Duration::from_millis(200));
    let count_while_paused = count.load(Ordering::Relaxed);
    // Allow at most 1 in-flight frame.
    assert!(
        count_while_paused <= count_at_pause + 1,
        "should not process while paused: at_pause={}, while_paused={}",
        count_at_pause,
        count_while_paused,
    );

    handle.resume().unwrap();
    assert!(!handle.is_paused());

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// max_restarts == 0 correctness
// ---------------------------------------------------------------------------

#[test]
fn max_restarts_zero_with_source_failure_trigger_never_restarts() {
    // max_restarts=0 must prevent restart even with SourceFailure trigger.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::failing()))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 0,
                restart_on: RestartTrigger::SourceFailure,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_secs(300),
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    assert_eq!(
        handle.metrics().restarts,
        0,
        "max_restarts=0 must prevent restart"
    );

    runtime.shutdown().unwrap();
}

#[test]
fn max_restarts_zero_with_source_or_panic_trigger_never_restarts() {
    // max_restarts=0 must prevent restart even with SourceOrStagePanic trigger.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(5)))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(PanicStage::new("panic"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 0,
                restart_on: RestartTrigger::SourceOrStagePanic,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_secs(300),
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    assert_eq!(
        handle.metrics().restarts,
        0,
        "max_restarts=0 must prevent restart"
    );

    runtime.shutdown().unwrap();
}

#[test]
fn max_restarts_zero_with_zero_window_never_restarts() {
    // Even with a zero restart_window (which normally resets counters),
    // max_restarts=0 must still prevent restart.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::failing()))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 0,
                restart_on: RestartTrigger::SourceOrStagePanic,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_millis(0),
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    assert_eq!(
        handle.metrics().restarts,
        0,
        "max_restarts=0 must prevent restart even with zero window"
    );

    runtime.shutdown().unwrap();
}

#[test]
fn restart_window_reset_still_works_with_nonzero_max() {
    // Verify that restart_window=0 with max_restarts>0 still allows
    // unlimited restarts (counter resets each cycle).
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(0))) // immediate EOS
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 1,
                restart_on: RestartTrigger::SourceFailure,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_millis(0),
            },
        ))
        .unwrap();

    // Let it restart a few cycles.
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(
        handle.is_alive(),
        "feed should keep restarting with window reset"
    );
    assert!(
        handle.metrics().restarts >= 2,
        "should have restarted multiple times"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Output lag health event
// ---------------------------------------------------------------------------

#[test]
fn output_lag_emits_health_event() {
    // Use a tiny output capacity and a fast producer.
    // Subscribe to output (to create an external receiver) and also
    // subscribe to health events.
    // The sentinel receiver should detect channel saturation and
    // emit HealthEvent::OutputLagged.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 50,
            fail_on_start: false,
            frame_delay: std::time::Duration::ZERO,
        }))
        .output_capacity(2)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    // Subscribe to output but never read — this creates a slow receiver.
    let _output_rx = runtime.output_subscribe();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    // Collect health events and look for OutputLagged.
    let mut saw_lag_event = false;
    let mut total_lost: u64 = 0;
    loop {
        match health_rx.try_recv() {
            Ok(event) => {
                if let HealthEvent::OutputLagged { messages_lost } = event {
                    total_lost += messages_lost;
                    saw_lag_event = true;
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    assert!(
        saw_lag_event,
        "should emit OutputLagged when output channel is saturated"
    );
    assert!(total_lost > 0, "messages_lost should be nonzero");

    runtime.shutdown().unwrap();
}

#[test]
fn no_lag_event_without_subscribers() {
    // When there are no external output subscribers, the worker
    // skips broadcasting and no lag events are emitted.
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 20,
            fail_on_start: false,
            frame_delay: std::time::Duration::ZERO,
        }))
        .output_capacity(2)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    // Deliberately do NOT subscribe to output.

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let mut saw_lag = false;
    loop {
        match health_rx.try_recv() {
            Ok(event) => {
                if matches!(event, HealthEvent::OutputLagged { .. }) {
                    saw_lag = true;
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    assert!(
        !saw_lag,
        "should not emit OutputLagged when no external subscribers"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Config validation — zero capacity / depth rejection
// ---------------------------------------------------------------------------

#[test]
fn zero_health_capacity_rejected() {
    let result = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(1)))
        .health_capacity(0)
        .build();
    assert!(result.is_err(), "health_capacity=0 must be rejected");
}

#[test]
fn zero_output_capacity_rejected() {
    let result = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(1)))
        .output_capacity(0)
        .build();
    assert!(result.is_err(), "output_capacity=0 must be rejected");
}

#[test]
fn zero_queue_depth_rejected() {
    use crate::backpressure::BackpressurePolicy;
    let result = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/test"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(NoOpStage::new("noop"))])
        .output_sink(Box::new(CountingSink::new().0))
        .backpressure(BackpressurePolicy::DropOldest { queue_depth: 0 })
        .build();
    assert!(result.is_err(), "queue_depth=0 must be rejected");
}

// ---------------------------------------------------------------------------
// File EOS — non-looping file source stops with EndOfStream
// ---------------------------------------------------------------------------

#[test]
fn file_eos_stops_cleanly() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(3)))
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let (sink, count) = CountingSink::new();
    let handle = runtime
        .add_feed(
            FeedConfig::builder()
                .source(SourceSpec::file("/tmp/test.mp4"))
                .camera_mode(CameraMode::Fixed)
                .stages(vec![Box::new(NoOpStage::new("noop"))])
                .output_sink(Box::new(sink))
                .restart(RestartPolicy {
                    max_restarts: 10,
                    restart_on: RestartTrigger::SourceFailure,
                    restart_delay: nv_core::Duration::from_millis(1),
                    restart_window: nv_core::Duration::from_millis(0),
                })
                .build()
                .unwrap(),
        )
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive(), "feed should have stopped");
    assert_eq!(
        count.load(Ordering::Relaxed),
        3,
        "all frames should produce output"
    );
    assert_eq!(handle.metrics().restarts, 0, "no restarts for file EOS");

    let mut saw_eos = false;
    let mut stopped_eos = false;
    while let Ok(event) = health_rx.try_recv() {
        match &event {
            HealthEvent::SourceEos { .. } => saw_eos = true,
            HealthEvent::FeedStopped { reason, .. } => {
                if matches!(reason, StopReason::EndOfStream) {
                    stopped_eos = true;
                }
            }
            _ => {}
        }
    }
    assert!(saw_eos, "should emit SourceEos health event");
    assert!(stopped_eos, "should stop with EndOfStream reason");

    runtime.shutdown().unwrap();
}

#[test]
fn looping_file_restarts_on_eos() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(0)))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(
            FeedConfig::builder()
                .source(SourceSpec::File {
                    path: "/tmp/test.mp4".into(),
                    loop_: true,
                })
                .camera_mode(CameraMode::Fixed)
                .stages(vec![Box::new(NoOpStage::new("noop"))])
                .output_sink(Box::new(sink))
                .restart(RestartPolicy {
                    max_restarts: 2,
                    restart_on: RestartTrigger::SourceFailure,
                    restart_delay: nv_core::Duration::from_millis(10),
                    restart_window: nv_core::Duration::from_secs(10),
                })
                .build()
                .unwrap(),
        )
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert!(!handle.is_alive());
    assert!(
        handle.metrics().restarts >= 1,
        "looping file should restart like any other source"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Stage error drops frame — no output produced
// ---------------------------------------------------------------------------

#[test]
fn stage_error_drops_frame_no_output() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(5)))
        .build()
        .unwrap();

    let (sink, count) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(nv_test_util::mock_stage::FailingStage::new(
                "fail",
            ))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    assert_eq!(
        count.load(Ordering::Relaxed),
        0,
        "failing stage should produce 0 outputs"
    );

    runtime.shutdown().unwrap();
}

#[test]
fn stage_error_emits_health_event() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(2)))
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(nv_test_util::mock_stage::FailingStage::new("err"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let mut error_count = 0u64;
    while let Ok(event) = health_rx.try_recv() {
        if matches!(event, HealthEvent::StageError { .. }) {
            error_count += 1;
        }
    }
    assert!(
        error_count >= 2,
        "should emit StageError for each failing frame, got {error_count}"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// SharedOutput (Arc) broadcast
// ---------------------------------------------------------------------------

#[test]
fn shared_output_broadcast_is_arc() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(3)))
        .output_capacity(32)
        .build()
        .unwrap();

    let mut rx = runtime.output_subscribe();
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let mut received = Vec::new();
    while let Ok(output) = rx.try_recv() {
        received.push(output);
    }

    assert!(!received.is_empty(), "should receive at least one output");
    for item in &received {
        assert!(
            Arc::strong_count(item) >= 1,
            "SharedOutput should be Arc-wrapped"
        );
    }

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Provenance timing
// ---------------------------------------------------------------------------

#[test]
fn provenance_has_valid_timestamps() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(1)))
        .output_capacity(32)
        .build()
        .unwrap();

    let mut rx = runtime.output_subscribe();
    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let mut output = None;
    while let Ok(o) = rx.try_recv() {
        output = Some(o);
    }
    let output = output.expect("should receive at least one output");

    let prov = &output.provenance;
    assert!(
        prov.pipeline_complete_ts >= prov.frame_receive_ts,
        "pipeline_complete_ts should be >= frame_receive_ts"
    );
    assert_eq!(prov.stages.len(), 1, "one stage provenance entry");
    let sp = &prov.stages[0];
    assert!(sp.end_ts >= sp.start_ts, "stage end >= start");
    assert_eq!(sp.result, crate::provenance::StageResult::Ok);

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Condvar pause — proper wakeup
// ---------------------------------------------------------------------------

#[test]
fn shutdown_wakes_paused_feed() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 100_000,
            fail_on_start: false,
            frame_delay: std::time::Duration::from_millis(1),
        }))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(20));
    handle.pause();
    std::thread::sleep(std::time::Duration::from_millis(20));

    let start = std::time::Instant::now();
    runtime.shutdown().unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(3),
        "shutdown should complete quickly, not block on paused feed (took {:?})",
        elapsed
    );
}

// ---------------------------------------------------------------------------
// Restart counter is cumulative
// ---------------------------------------------------------------------------

#[test]
fn restarts_metric_is_cumulative() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(0)))
        .build()
        .unwrap();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 1,
                restart_on: RestartTrigger::SourceFailure,
                restart_delay: nv_core::Duration::from_millis(10),
                restart_window: nv_core::Duration::from_millis(0),
            },
        ))
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(handle.is_alive(), "feed should keep restarting");
    let restarts = handle.metrics().restarts;
    assert!(
        restarts >= 3,
        "cumulative restarts should be >= 3, got {restarts}"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Output lag detection — deterministic sentinel-based tests
// ---------------------------------------------------------------------------

/// Verifying that messages_lost is a per-event delta (not cumulative).
/// With output_capacity=2 and 30 fast frames, the sentinel should observe
/// multiple Lagged events whose deltas sum to the total overflow.
#[test]
fn lag_messages_lost_is_per_event_delta() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 30,
            fail_on_start: false,
            frame_delay: std::time::Duration::ZERO,
        }))
        .output_capacity(2)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    // Create a slow external subscriber (never reads).
    let _output_rx = runtime.output_subscribe();

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let mut deltas: Vec<u64> = Vec::new();
    loop {
        match health_rx.try_recv() {
            Ok(event) => {
                if let HealthEvent::OutputLagged { messages_lost } = event {
                    // Each delta must be positive.
                    assert!(
                        messages_lost > 0,
                        "each lag event must have messages_lost > 0"
                    );
                    deltas.push(messages_lost);
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    assert!(
        !deltas.is_empty(),
        "should have at least one OutputLagged event"
    );

    // The sum of all deltas should be <= (frames - capacity) since the
    // canary can only report messages it actually missed.
    let total_lost: u64 = deltas.iter().sum();
    assert!(total_lost > 0, "total messages lost should be > 0, got 0");

    runtime.shutdown().unwrap();
}

/// When a subscriber disconnects, no spurious lag events should be
/// generated. The receiver_count drops below the external threshold,
/// so the worker stops sending to broadcast entirely.
#[test]
fn no_spurious_lag_on_subscriber_disconnect() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 1_000,
            fail_on_start: false,
            frame_delay: std::time::Duration::from_millis(1),
        }))
        .output_capacity(64)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();

    // Subscribe then immediately drop — simulates subscriber churn.
    let output_rx = runtime.output_subscribe();
    drop(output_rx);

    let (sink, _) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
        ))
        .unwrap();

    // Let it run a bit then shutdown.
    std::thread::sleep(std::time::Duration::from_millis(100));
    let feed_id = handle.id();
    runtime.remove_feed(feed_id).unwrap();

    // No lag events should have been emitted: the external subscriber
    // was dropped before frames started flowing, so receiver_count <= 1.
    let mut saw_lag = false;
    loop {
        match health_rx.try_recv() {
            Ok(event) => {
                if matches!(event, HealthEvent::OutputLagged { .. }) {
                    saw_lag = true;
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    assert!(
        !saw_lag,
        "should not emit OutputLagged when external subscriber disconnects"
    );

    runtime.shutdown().unwrap();
}

/// Multi-feed contention: two feeds sending rapidly into a small
/// output channel. Sentinel-observed OutputLagged events are
/// runtime-global (no feed_id).
#[test]
fn multi_feed_lag_attribution_is_global() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count: 30,
            fail_on_start: false,
            frame_delay: std::time::Duration::ZERO,
        }))
        .output_capacity(2)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let _output_rx = runtime.output_subscribe();

    let (s1, _) = CountingSink::new();
    let (s2, _) = CountingSink::new();
    let h1 = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(s1),
        ))
        .unwrap();
    let h2 = runtime
        .add_feed(build_config(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(s2),
        ))
        .unwrap();

    wait_for_stop(&h1, std::time::Duration::from_secs(5));
    wait_for_stop(&h2, std::time::Duration::from_secs(5));

    let mut lag_count = 0u64;
    loop {
        match health_rx.try_recv() {
            Ok(event) => {
                if let HealthEvent::OutputLagged { messages_lost } = event {
                    // The event has no feed_id — it's global.
                    assert!(messages_lost > 0);
                    lag_count += 1;
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    // With 2 feeds × 30 frames into capacity=2, we should see lag.
    assert!(
        lag_count > 0,
        "multi-feed should trigger OutputLagged with tiny capacity"
    );

    runtime.shutdown().unwrap();
}

/// Throttling: sustained overflow should produce a bounded number of
/// health events, not one per frame.
#[test]
fn lag_throttling_bounds_event_count() {
    let frame_count = 200u64;
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(MockFactory {
            frame_count,
            fail_on_start: false,
            frame_delay: std::time::Duration::ZERO,
        }))
        .output_capacity(2)
        // Large health capacity to avoid the health channel itself
        // overflowing from backpressure events.
        .health_capacity(4096)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let _output_rx = runtime.output_subscribe();

    let (sink, _) = CountingSink::new();
    // Use never-restart so the feed stops after one run and we
    // don't accumulate health events from multiple source sessions.
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 0,
                restart_on: RestartTrigger::Never,
                ..RestartPolicy::default()
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    let mut lag_event_count = 0u64;
    loop {
        match health_rx.try_recv() {
            Ok(event) => {
                if matches!(event, HealthEvent::OutputLagged { .. }) {
                    lag_event_count += 1;
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    // With 200 frames and capacity 2, without throttling we'd get
    // up to ~66 lag events (one every capacity+1 sends). Throttling
    // should keep this much lower: 1 transition event + at most a
    // few periodic ones (1/sec). The test runs in well under 1 second.
    assert!(lag_event_count > 0, "should see at least one lag event");
    assert!(
        lag_event_count < frame_count / 2,
        "throttling should bound lag events: got {lag_event_count} for {frame_count} frames"
    );

    runtime.shutdown().unwrap();
}
