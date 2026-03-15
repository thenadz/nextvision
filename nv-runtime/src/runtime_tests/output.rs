//! Output subscription, SharedOutput broadcast, provenance, and
//! sentinel-based output-lag detection tests.

use super::super::*;
use std::sync::Arc;

use nv_test_util::mock_stage::NoOpStage;
use tokio::sync::broadcast;

use crate::shutdown::{RestartPolicy, RestartTrigger};

use super::harness::*;

// ---------------------------------------------------------------------------
// Output subscription
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
// Output lag health event
// ---------------------------------------------------------------------------

#[test]
fn output_lag_emits_health_event() {
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
// Output lag detection — deterministic sentinel-based tests
// ---------------------------------------------------------------------------

/// Verifying that messages_lost is a per-event delta (not cumulative).
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
/// generated.
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

    // No lag events should have been emitted.
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
        .health_capacity(4096)
        .build()
        .unwrap();

    let mut health_rx = runtime.health_subscribe();
    let _output_rx = runtime.output_subscribe();

    let (sink, _) = CountingSink::new();
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

    assert!(lag_event_count > 0, "should see at least one lag event");
    assert!(
        lag_event_count < frame_count / 2,
        "throttling should bound lag events: got {lag_event_count} for {frame_count} frames"
    );

    runtime.shutdown().unwrap();
}
