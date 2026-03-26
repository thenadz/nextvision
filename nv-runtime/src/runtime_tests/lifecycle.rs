//! Lifecycle tests: multi-feed registration, start/stop, shutdown races,
//! restart behavior, pause/resume, runtime handle, condvar pause.

use super::super::*;
use std::sync::atomic::Ordering;

use nv_test_util::mock_stage::{NoOpStage, PanicStage};

use crate::shutdown::{RestartPolicy, RestartTrigger};

use super::harness::*;

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
    let runtime = build_runtime(10);

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
    let runtime = build_runtime(10_000);

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
// 7. Shutdown races
// ---------------------------------------------------------------------------

#[test]
fn immediate_shutdown_after_add() {
    let runtime = build_runtime(10_000);

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
    let runtime = build_runtime(5);

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
// RuntimeHandle
// ---------------------------------------------------------------------------

#[test]
fn runtime_handle_is_cloneable_and_functional() {
    let runtime = build_runtime(3);

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
    let runtime = build_runtime(10_000);

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
    // Allow a small number of in-flight frames: the worker may be
    // mid-processing when pause is observed, and the async sink thread
    // may still be draining its bounded channel.  Under heavy CPU
    // contention (full test suite) this can be 2-3 frames.
    assert!(
        count_while_paused <= count_at_pause + 3,
        "should not process while paused: at_pause={}, while_paused={}",
        count_at_pause,
        count_while_paused,
    );

    handle.resume().unwrap();
    assert!(!handle.is_paused());

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
    let runtime = build_runtime(5);

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
    let runtime = build_runtime(5);

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
// max_restarts == 0 correctness
// ---------------------------------------------------------------------------

#[test]
fn max_restarts_zero_with_source_failure_trigger_never_restarts() {
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
    let runtime = build_runtime(5);

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
    let _ = handle.pause();
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
    let runtime = build_runtime(0);

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
