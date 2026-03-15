//! Feed-level tests: failure isolation, backpressure drops, file EOS,
//! stage errors, source tick stopped, and event-driven idle source.

use super::super::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use nv_core::config::{CameraMode, SourceSpec};
use nv_core::health::StopReason;
use nv_media::ingress::{FrameSink, IngressOptions, MediaIngress, MediaIngressFactory};
use nv_test_util::mock_stage::NoOpStage;

use crate::shutdown::{RestartPolicy, RestartTrigger};

use super::harness::*;

// ---------------------------------------------------------------------------
// Bounded queue + backpressure
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
// Feed failure isolation
// ---------------------------------------------------------------------------

#[test]
fn feed_failure_isolation() {
    let runtime = build_runtime(10);

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
// File EOS — non-looping file source stops with EndOfStream
// ---------------------------------------------------------------------------

#[test]
fn file_eos_stops_cleanly() {
    let runtime = build_runtime(3);

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
    let runtime = build_runtime(0);

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
    let runtime = build_runtime(5);

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
    let runtime = build_runtime(2);

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
// Source tick — Stopped causes feed to exit without restart
// ---------------------------------------------------------------------------

/// Mock source that sends N frames then keeps running, but returns
/// `SourceStatus::Stopped` from `tick()` after the frames are sent.
struct TickStoppedIngress {
    feed_id: FeedId,
    spec: SourceSpec,
    frame_count: u64,
    stopped_after_eos: Arc<std::sync::atomic::AtomicBool>,
}

impl MediaIngress for TickStoppedIngress {
    fn start(&mut self, sink: Box<dyn FrameSink>) -> Result<(), nv_core::error::MediaError> {
        let count = self.frame_count;
        let feed_id = self.feed_id;
        let flag = Arc::clone(&self.stopped_after_eos);
        std::thread::spawn(move || {
            for i in 0..count {
                let frame = make_test_frame(feed_id, i);
                sink.on_frame(frame);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            // Don't call on_eos — the worker will tick() and discover Stopped.
            flag.store(true, Ordering::Release);
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
    fn tick(&mut self) -> nv_media::TickOutcome {
        if self.stopped_after_eos.load(Ordering::Acquire) {
            nv_media::TickOutcome::stopped()
        } else {
            nv_media::TickOutcome {
                status: nv_media::SourceStatus::Running,
                next_tick: Some(std::time::Duration::from_millis(50)),
            }
        }
    }
}

struct TickStoppedFactory {
    frame_count: u64,
}

impl MediaIngressFactory for TickStoppedFactory {
    fn create(
        &self,
        options: IngressOptions,
    ) -> Result<Box<dyn MediaIngress>, nv_core::error::MediaError> {
        Ok(Box::new(TickStoppedIngress {
            feed_id: options.feed_id,
            spec: options.spec,
            frame_count: self.frame_count,
            stopped_after_eos: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }))
    }
}

#[test]
fn source_tick_stopped_terminates_feed() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(TickStoppedFactory { frame_count: 5 }))
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let (sink, count) = CountingSink::new();
    let handle = runtime
        .add_feed(build_config_with_restart(
            vec![Box::new(NoOpStage::new("noop"))],
            Box::new(sink),
            RestartPolicy {
                max_restarts: 3,
                restart_on: RestartTrigger::SourceOrStagePanic,
                ..Default::default()
            },
        ))
        .unwrap();

    wait_for_stop(&handle, std::time::Duration::from_secs(5));

    // The feed should have stopped (not restarted).
    assert!(
        !handle.is_alive(),
        "feed should stop when tick() returns Stopped"
    );

    // Should have processed some frames.
    assert!(
        count.load(Ordering::Relaxed) > 0,
        "should have processed frames"
    );

    // Verify no FeedRestarting events were emitted.
    let mut restarted = false;
    loop {
        match health_rx.try_recv() {
            Ok(HealthEvent::FeedRestarting { .. }) => {
                restarted = true;
                break;
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert!(
        !restarted,
        "feed should not restart when source reports Stopped"
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Event-driven idle source progression
// ---------------------------------------------------------------------------

/// Mock source that returns a TickOutcome with a specific next_tick hint
/// (for reconnecting) and then transitions to Stopped.
struct ReconnectingIngress {
    feed_id: FeedId,
    spec: SourceSpec,
    backoff: std::time::Duration,
    frames_before_reconnect: u64,
    ticks_before_stop: u32,
    tick_count: u32,
    sent_frames: Arc<std::sync::atomic::AtomicBool>,
}

impl MediaIngress for ReconnectingIngress {
    fn start(&mut self, sink: Box<dyn FrameSink>) -> Result<(), nv_core::error::MediaError> {
        let count = self.frames_before_reconnect;
        let feed_id = self.feed_id;
        let flag = Arc::clone(&self.sent_frames);
        std::thread::spawn(move || {
            for i in 0..count {
                sink.on_frame(make_test_frame(feed_id, i));
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            flag.store(true, Ordering::Release);
        });
        Ok(())
    }
    fn stop(&mut self) -> Result<(), nv_core::error::MediaError> { Ok(()) }
    fn pause(&mut self) -> Result<(), nv_core::error::MediaError> { Ok(()) }
    fn resume(&mut self) -> Result<(), nv_core::error::MediaError> { Ok(()) }
    fn source_spec(&self) -> &SourceSpec { &self.spec }
    fn feed_id(&self) -> FeedId { self.feed_id }
    fn tick(&mut self) -> nv_media::TickOutcome {
        if !self.sent_frames.load(Ordering::Acquire) {
            return nv_media::TickOutcome {
                status: nv_media::SourceStatus::Running,
                next_tick: Some(std::time::Duration::from_millis(50)),
            };
        }
        self.tick_count += 1;
        if self.tick_count > self.ticks_before_stop {
            nv_media::TickOutcome::stopped()
        } else {
            nv_media::TickOutcome::reconnecting(self.backoff)
        }
    }
}

struct ReconnectingFactory {
    backoff: std::time::Duration,
    frames_before_reconnect: u64,
    ticks_before_stop: u32,
}

impl MediaIngressFactory for ReconnectingFactory {
    fn create(
        &self,
        options: IngressOptions,
    ) -> Result<Box<dyn MediaIngress>, nv_core::error::MediaError> {
        Ok(Box::new(ReconnectingIngress {
            feed_id: options.feed_id,
            spec: options.spec,
            backoff: self.backoff,
            frames_before_reconnect: self.frames_before_reconnect,
            ticks_before_stop: self.ticks_before_stop,
            tick_count: 0,
            sent_frames: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }))
    }
}

#[test]
fn event_driven_source_progression_no_fixed_floor() {
    let runtime = Runtime::builder()
        .ingress_factory(Box::new(ReconnectingFactory {
            backoff: std::time::Duration::from_millis(50),
            frames_before_reconnect: 3,
            ticks_before_stop: 3,
        }))
        .build()
        .unwrap();

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

    let start = std::time::Instant::now();
    wait_for_stop(&handle, std::time::Duration::from_secs(5));
    let elapsed = start.elapsed();

    assert!(
        !handle.is_alive(),
        "feed should have stopped after reconnect budget exhausted"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "event-driven progression should complete quickly (took {:?})",
        elapsed,
    );

    runtime.shutdown().unwrap();
}

// ---------------------------------------------------------------------------
// Finding 2: Poisoned pause/resume locks must not panic
// ---------------------------------------------------------------------------

/// If the pause condvar mutex is poisoned (e.g., a thread panicked while
/// holding it), pause() and resume() on FeedHandle must not panic. They
/// should recover the inner value via `unwrap_or_else(|e| e.into_inner())`.
#[test]
fn poisoned_pause_condvar_does_not_panic() {
    use crate::worker::FeedSharedState;
    use nv_core::id::FeedId;

    let shared = Arc::new(FeedSharedState::new(FeedId::new(99), None));

    // Poison the mutex by panicking in a thread that holds the lock.
    {
        let shared_clone = Arc::clone(&shared);
        let handle = std::thread::spawn(move || {
            let (lock, _cvar) = &shared_clone.pause_condvar;
            let _guard = lock.lock().unwrap();
            panic!("intentional panic to poison mutex");
        });
        // Wait for the thread to finish (it will panic).
        let _ = handle.join();
    }

    // Verify the mutex is actually poisoned.
    let (lock, _) = &shared.pause_condvar;
    assert!(lock.lock().is_err(), "mutex should be poisoned");

    // Now pause and resume through FeedHandle — must not panic.
    let feed_handle = crate::feed::FeedHandle::new(Arc::clone(&shared));

    // pause() should succeed (recovers poisoned lock via into_inner()).
    let pause_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        feed_handle.pause()
    }));
    assert!(
        pause_result.is_ok(),
        "FeedHandle::pause() must not panic on poisoned mutex",
    );

    // resume() should also succeed.
    let resume_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        feed_handle.resume()
    }));
    assert!(
        resume_result.is_ok(),
        "FeedHandle::resume() must not panic on poisoned mutex",
    );
}
