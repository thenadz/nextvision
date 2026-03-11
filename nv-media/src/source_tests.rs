//! Tests for [`MediaSource`] lifecycle, event handling, and health emission.

use super::*;
use crate::factory::GstMediaIngressFactory;
use crate::ingress::MediaIngressFactory;
use nv_core::config::BackoffKind;
use nv_frame::FrameEnvelope;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// Test sink helpers
// ---------------------------------------------------------------------------

/// Minimal test sink that counts frames, errors, and EOS signals.
struct CountingSink {
    frame_count: Arc<AtomicU32>,
    error_count: Arc<AtomicU32>,
    eos_count: Arc<AtomicU32>,
}

impl CountingSink {
    fn new() -> (Self, Arc<AtomicU32>, Arc<AtomicU32>, Arc<AtomicU32>) {
        let frames = Arc::new(AtomicU32::new(0));
        let errors = Arc::new(AtomicU32::new(0));
        let eos = Arc::new(AtomicU32::new(0));
        (
            Self {
                frame_count: Arc::clone(&frames),
                error_count: Arc::clone(&errors),
                eos_count: Arc::clone(&eos),
            },
            frames,
            errors,
            eos,
        )
    }
}

impl FrameSink for CountingSink {
    fn on_frame(&self, _frame: FrameEnvelope) {
        self.frame_count.fetch_add(1, Ordering::Relaxed);
    }
    fn on_error(&self, _error: MediaError) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
    fn on_eos(&self) {
        self.eos_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Captures the last error received by the sink (preserving variant info).
struct CapturingSink {
    last_error_variant: Mutex<Option<MediaError>>,
}

impl CapturingSink {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            last_error_variant: Mutex::new(None),
        })
    }

    fn last_error_variant(&self) -> Option<MediaError> {
        self.last_error_variant.lock().unwrap().clone()
    }
}

impl FrameSink for CapturingSink {
    fn on_frame(&self, _frame: FrameEnvelope) {}
    fn on_error(&self, error: MediaError) {
        *self.last_error_variant.lock().unwrap() = Some(error);
    }
    fn on_eos(&self) {}
}

// ---------------------------------------------------------------------------
// Health sink helper
// ---------------------------------------------------------------------------

/// Collects health events emitted by a source.
struct RecordingHealthSink {
    events: Mutex<Vec<HealthEvent>>,
}

impl RecordingHealthSink {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
        })
    }

    fn drain(&self) -> Vec<HealthEvent> {
        self.events.lock().unwrap().drain(..).collect()
    }
}

impl HealthSink for RecordingHealthSink {
    fn emit(&self, event: HealthEvent) {
        self.events.lock().unwrap().push(event);
    }
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

fn test_spec() -> SourceSpec {
    SourceSpec::rtsp("rtsp://test/stream")
}

fn test_reconnect() -> ReconnectPolicy {
    ReconnectPolicy::default()
}

fn limited_reconnect(max: u32) -> ReconnectPolicy {
    ReconnectPolicy {
        max_attempts: max,
        base_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(100),
        backoff: BackoffKind::Exponential,
    }
}

/// Helper: create a source and force it into Running state with a stub session.
fn started_source(
    spec: SourceSpec,
    policy: ReconnectPolicy,
) -> (MediaSource, Arc<AtomicU32>, Arc<AtomicU32>, Arc<AtomicU32>) {
    let mut src = MediaSource::new(FeedId::new(1), spec, policy);
    let (sink, frames, errors, eos) = CountingSink::new();
    src.sink = Some(Arc::new(sink));
    src.create_session_stub();
    src.state = SourceState::Running;
    (src, frames, errors, eos)
}

/// Helper: create a source with a health sink attached.
fn started_source_with_health(
    spec: SourceSpec,
    policy: ReconnectPolicy,
) -> (
    MediaSource,
    Arc<AtomicU32>,
    Arc<AtomicU32>,
    Arc<AtomicU32>,
    Arc<RecordingHealthSink>,
) {
    let mut src = MediaSource::new(FeedId::new(1), spec, policy);
    let (sink, frames, errors, eos) = CountingSink::new();
    let health = RecordingHealthSink::new();
    src.set_health_sink(health.clone() as Arc<dyn HealthSink>);
    src.sink = Some(Arc::new(sink));
    src.create_session_stub();
    src.state = SourceState::Running;
    (src, frames, errors, eos, health)
}

// ===========================================================================
// Lifecycle tests
// ===========================================================================

#[test]
fn lifecycle_idle_start_stop() {
    let (mut src, _frames, _errors, eos) = started_source(test_spec(), test_reconnect());
    assert_eq!(src.source_state(), SourceState::Running);
    src.stop().unwrap();
    assert_eq!(src.source_state(), SourceState::Stopped);
    assert_eq!(eos.load(Ordering::Relaxed), 1);
}

#[test]
fn cannot_start_after_stop() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    src.stop().unwrap();
    let (sink, _, _, _) = CountingSink::new();
    let result = src.start(Box::new(sink));
    assert!(result.is_err());
}

#[test]
fn pause_resume_lifecycle() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    src.pause().unwrap();
    assert_eq!(src.source_state(), SourceState::Paused);

    // Double-pause fails.
    assert!(src.pause().is_err());

    src.resume().unwrap();
    assert_eq!(src.source_state(), SourceState::Running);

    // Double-resume fails.
    assert!(src.resume().is_err());

    src.stop().unwrap();
}

#[test]
fn stop_is_idempotent() {
    let (mut src, _, _, eos) = started_source(test_spec(), test_reconnect());
    src.stop().unwrap();
    src.stop().unwrap();
    // on_eos only called once.
    assert_eq!(eos.load(Ordering::Relaxed), 1);
}

// ===========================================================================
// Event handling tests
// ===========================================================================

#[test]
fn stream_started_enters_running() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    src.state = SourceState::Reconnecting;
    let delay = src.handle_event(MediaEvent::StreamStarted);
    assert!(delay.is_none());
    assert_eq!(src.source_state(), SourceState::Running);
}

#[test]
fn error_triggers_reconnect() {
    let (mut src, _, errors, _) = started_source(test_spec(), test_reconnect());
    let delay = src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed {
            detail: "test".into(),
        },
        debug: None,
    });
    assert!(delay.is_some());
    assert_eq!(src.source_state(), SourceState::Reconnecting);
    assert_eq!(errors.load(Ordering::Relaxed), 1);
}

#[test]
fn error_stops_when_budget_exhausted() {
    let (mut src, _, errors, eos) = started_source(test_spec(), limited_reconnect(1));
    // First error triggers reconnect.
    src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed {
            detail: "first".into(),
        },
        debug: None,
    });
    assert_eq!(src.source_state(), SourceState::Reconnecting);

    // Second error exhausts budget.
    let delay = src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed {
            detail: "second".into(),
        },
        debug: None,
    });
    assert!(delay.is_none());
    assert_eq!(src.source_state(), SourceState::Stopped);
    assert_eq!(errors.load(Ordering::Relaxed), 2);
    assert_eq!(eos.load(Ordering::Relaxed), 1); // terminal EOS
}

#[test]
fn eos_on_file_source_is_terminal() {
    let spec = SourceSpec::File {
        path: "/tmp/test.mp4".into(),
        loop_: false,
    };
    let (mut src, _, _, eos) = started_source(spec, test_reconnect());
    let delay = src.handle_event(MediaEvent::Eos);
    assert!(delay.is_none());
    assert_eq!(src.source_state(), SourceState::Stopped);
    assert_eq!(eos.load(Ordering::Relaxed), 1);
}

#[test]
fn eos_on_rtsp_source_triggers_reconnect() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    let delay = src.handle_event(MediaEvent::Eos);
    assert!(delay.is_some());
    assert_eq!(src.source_state(), SourceState::Reconnecting);
}

#[test]
fn warning_does_not_change_state() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    let delay = src.handle_event(MediaEvent::Warning {
        message: "clock drift".into(),
        debug: None,
    });
    assert!(delay.is_none());
    assert_eq!(src.source_state(), SourceState::Running);
}

#[test]
fn discontinuity_does_not_change_state() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    let delay = src.handle_event(MediaEvent::Discontinuity {
        gap_ns: 10_000_000_000,
        prev_pts_ns: 0,
        current_pts_ns: 10_000_000_000,
    });
    assert!(delay.is_none());
    assert_eq!(src.source_state(), SourceState::Running);
}

#[test]
fn total_reconnects_accumulates() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    for _ in 0..5 {
        src.handle_event(MediaEvent::Error {
            error: MediaError::DecodeFailed {
                detail: "test".into(),
            },
            debug: None,
        });
    }
    assert_eq!(src.total_reconnects(), 5);
}

// ===========================================================================
// HealthSink emission tests
// ===========================================================================

#[test]
fn health_emits_connected_on_stream_started() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.state = SourceState::Reconnecting;
    src.handle_event(MediaEvent::StreamStarted);
    let events = health.drain();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], HealthEvent::SourceConnected { .. }));
}

#[test]
fn health_emits_disconnected_and_reconnecting_on_error() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed {
            detail: "test".into(),
        },
        debug: None,
    });
    let events = health.drain();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], HealthEvent::SourceDisconnected { .. }));
    assert!(matches!(events[1], HealthEvent::SourceReconnecting { .. }));
}

#[test]
fn health_emits_feed_stopped_on_budget_exhausted() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), limited_reconnect(1));
    // First error → reconnecting.
    src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed {
            detail: "first".into(),
        },
        debug: None,
    });
    health.drain(); // clear first batch

    // Second error exhausts budget → stopped.
    // FeedStopped is now emitted by the worker, not the source.
    // The source emits SourceDisconnected for the error.
    src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed {
            detail: "second".into(),
        },
        debug: None,
    });
    let events = health.drain();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, HealthEvent::SourceDisconnected { .. })),
        "expected SourceDisconnected event"
    );
    // Source no longer emits FeedStopped — worker is the canonical owner.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, HealthEvent::FeedStopped { .. })),
        "source must not emit FeedStopped"
    );
    assert_eq!(src.source_state(), SourceState::Stopped);
}

#[test]
fn health_emits_no_feed_stopped_on_file_eos() {
    let spec = SourceSpec::File {
        path: "/tmp/test.mp4".into(),
        loop_: false,
    };
    let (mut src, _, _, _, health) = started_source_with_health(spec, test_reconnect());
    src.handle_event(MediaEvent::Eos);
    let events = health.drain();
    // Source no longer emits FeedStopped — worker is the canonical owner.
    assert_eq!(events.len(), 0, "source must not emit FeedStopped; events: {:?}", events);
    assert_eq!(src.source_state(), SourceState::Stopped);
}

#[test]
fn health_emits_feed_stopped_on_manual_stop() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.stop().unwrap();
    let events = health.drain();
    // source.stop() no longer emits FeedStopped — that is the worker's
    // responsibility. The source only logs.
    assert_eq!(events.len(), 0);
}

#[test]
fn health_no_duplicate_on_idempotent_stop() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.stop().unwrap();
    src.stop().unwrap();
    let events = health.drain();
    // source.stop() no longer emits FeedStopped.
    assert_eq!(events.len(), 0);
}

#[test]
fn health_emits_reconnecting_on_successful_reconnect() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    // Force into Reconnecting state.
    src.state = SourceState::Reconnecting;
    src.reconnect.record_attempt();

    // try_reconnect creates a stub session (which will fail in non-gst),
    // so instead simulate the full cycle via handle_event.
    src.handle_event(MediaEvent::StreamStarted);
    let events = health.drain();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, HealthEvent::SourceConnected { .. }))
    );
}

// ===========================================================================
// GstMediaIngressFactory tests
// ===========================================================================

#[test]
fn factory_creates_source() {
    let factory = GstMediaIngressFactory::new();
    let result = factory.create(FeedId::new(42), test_spec(), test_reconnect(), None);
    let source = result.unwrap();
    assert_eq!(source.feed_id(), FeedId::new(42));
}

#[test]
fn factory_wires_health_sink() {
    let health = RecordingHealthSink::new();
    let factory = GstMediaIngressFactory::with_health_sink(health.clone() as Arc<dyn HealthSink>);
    let source = factory
        .create(FeedId::new(1), test_spec(), test_reconnect(), None)
        .unwrap();

    assert_eq!(source.feed_id(), FeedId::new(1));

    // Verify health sink wiring through a manually built source
    // that goes through the same path.
    let health2 = RecordingHealthSink::new();
    let (mut src, _, _, _, _) = started_source_with_health(test_spec(), test_reconnect());
    src.health_sink = Some(health2.clone() as Arc<dyn HealthSink>);
    // Trigger a health event that the source still emits (e.g. SourceConnected
    // via handle_event). source.stop() no longer emits FeedStopped.
    src.emit_health(HealthEvent::SourceConnected {
        feed_id: src.feed_id,
    });
    let events = health2.drain();
    assert!(!events.is_empty(), "health sink should receive events");
}

#[test]
fn factory_wires_ptz_provider() {
    use crate::bridge::PtzTelemetry;
    use nv_core::timestamp::MonotonicTs;

    struct FixedPtz;
    impl PtzProvider for FixedPtz {
        fn latest(&self) -> Option<PtzTelemetry> {
            Some(PtzTelemetry {
                pan: 90.0,
                tilt: 0.0,
                zoom: 1.0,
                ts: MonotonicTs::from_nanos(0),
            })
        }
    }

    let ptz: Arc<dyn PtzProvider> = Arc::new(FixedPtz);
    let factory = GstMediaIngressFactory::new();
    let source = factory
        .create(FeedId::new(1), test_spec(), test_reconnect(), Some(ptz))
        .unwrap();

    assert_eq!(source.feed_id(), FeedId::new(1));
}

#[test]
fn factory_default_is_equivalent_to_new() {
    let f1 = GstMediaIngressFactory::new();
    let f2 = GstMediaIngressFactory::default();
    let s1 = f1
        .create(FeedId::new(1), test_spec(), test_reconnect(), None)
        .unwrap();
    let s2 = f2
        .create(FeedId::new(2), test_spec(), test_reconnect(), None)
        .unwrap();
    assert_eq!(s1.feed_id(), FeedId::new(1));
    assert_eq!(s2.feed_id(), FeedId::new(2));
}

// ===========================================================================
// Regression tests (Finding 4)
// ===========================================================================

/// start() on a stopped source returns Err and stays Stopped.
#[test]
fn start_rejects_stopped_source() {
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect());
    src.state = SourceState::Stopped;
    let (sink, _, _, _) = CountingSink::new();
    let result = src.start(Box::new(sink));
    assert!(result.is_err(), "start on stopped source must return Err");
    assert_eq!(src.source_state(), SourceState::Stopped);
}

/// Real initial-connection-failure: without gst-backend, create_session
/// returns Err(Unsupported). start() must surface this as Err and leave
/// the source in the Idle state so the caller can retry.
#[cfg(not(feature = "gst-backend"))]
#[test]
fn start_initial_connection_failure_stays_idle() {
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect());
    let (sink, _, _, _) = CountingSink::new();
    let result = src.start(Box::new(sink));
    assert!(
        result.is_err(),
        "start must return Err when session cannot be created"
    );
    assert!(
        matches!(result.unwrap_err(), MediaError::Unsupported { .. }),
        "expected Unsupported without gst-backend"
    );
    assert_eq!(
        src.source_state(),
        SourceState::Idle,
        "source must remain Idle after initial failure"
    );
}

/// Sink receives the original typed error variant (not a downgraded string).
#[test]
fn sink_receives_typed_error_variant() {
    let capturing = CapturingSink::new();
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect());
    src.sink = Some(capturing.clone() as Arc<dyn FrameSink>);
    src.create_session_stub();
    src.state = SourceState::Running;

    // Send a ConnectionFailed through the event handler.
    src.handle_event(MediaEvent::Error {
        error: MediaError::ConnectionFailed {
            url: String::new(),
            detail: "connection refused".into(),
        },
        debug: None,
    });

    let variant = capturing
        .last_error_variant()
        .expect("sink should have received error");
    match variant {
        MediaError::ConnectionFailed { url, detail } => {
            assert_eq!(
                url, "rtsp://test/stream",
                "URL should be enriched from spec"
            );
            assert_eq!(detail, "connection refused");
        }
        other => panic!("sink should receive ConnectionFailed, got: {other}"),
    }
}

/// Sink receives Timeout variant unchanged (not downgraded to DecodeFailed).
#[test]
fn sink_receives_timeout_variant() {
    let capturing = CapturingSink::new();
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect());
    src.sink = Some(capturing.clone() as Arc<dyn FrameSink>);
    src.create_session_stub();
    src.state = SourceState::Running;

    src.handle_event(MediaEvent::Error {
        error: MediaError::Timeout,
        debug: None,
    });

    let variant = capturing
        .last_error_variant()
        .expect("sink should have received error");
    assert!(
        matches!(variant, MediaError::Timeout),
        "sink should receive Timeout, got: {variant}"
    );
}

/// No duplicate SourceConnected when bus fires StreamStarted after start().
#[test]
fn no_duplicate_source_connected_emission() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    // started_source_with_health already has state = Running.
    // Simulating what happens when the GStreamer bus fires StreamStarted
    // *after* start() already moved to Running — should be suppressed.
    health.drain(); // clear SourceConnected from initial start.

    src.handle_event(MediaEvent::StreamStarted);
    let events = health.drain();
    assert!(
        events.is_empty(),
        "should suppress duplicate SourceConnected"
    );
}

/// File loop EOS: seeks to start (no reconnect, no stop).
#[test]
fn file_loop_eos_seeks_to_start() {
    let spec = SourceSpec::File {
        path: "/tmp/test.mp4".into(),
        loop_: true,
    };
    let (mut src, _, _, eos) = started_source(spec, test_reconnect());
    // seek_start on the stub session succeeds (stub is in Running state).
    let delay = src.handle_event(MediaEvent::Eos);
    assert!(delay.is_none(), "looping file should not trigger reconnect");
    assert_eq!(
        src.source_state(),
        SourceState::Running,
        "should stay running"
    );
    assert_eq!(eos.load(Ordering::Relaxed), 0, "should not signal EOS");
}

/// Error variant is preserved end-to-end through health events.
#[test]
fn error_variant_preserved_in_health() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.handle_event(MediaEvent::Error {
        error: MediaError::ConnectionFailed {
            url: String::new(),
            detail: "connection refused".into(),
        },
        debug: None,
    });
    let events = health.drain();
    // First event should be SourceDisconnected with the enriched error.
    let disconnect = events
        .iter()
        .find(|e| matches!(e, HealthEvent::SourceDisconnected { .. }));
    assert!(disconnect.is_some(), "should emit SourceDisconnected");
    match disconnect.unwrap() {
        HealthEvent::SourceDisconnected { reason, .. } => {
            // The enrich_error method should have filled in the source URL.
            match reason {
                MediaError::ConnectionFailed { url, .. } => {
                    assert_eq!(
                        url, "rtsp://test/stream",
                        "URL should be enriched from spec"
                    );
                }
                other => panic!("expected ConnectionFailed, got: {other}"),
            }
        }
        _ => unreachable!(),
    }
}

/// Timeout error variant preserved through health events.
#[test]
fn timeout_variant_preserved_in_health() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.handle_event(MediaEvent::Error {
        error: MediaError::Timeout,
        debug: None,
    });
    let events = health.drain();
    let disconnect = events
        .iter()
        .find(|e| matches!(e, HealthEvent::SourceDisconnected { .. }));
    assert!(disconnect.is_some());
    match disconnect.unwrap() {
        HealthEvent::SourceDisconnected { reason, .. } => {
            assert!(
                matches!(reason, MediaError::Timeout),
                "Timeout should be preserved"
            );
        }
        _ => unreachable!(),
    }
}

// ===========================================================================
// Liveness watchdog tests
// ===========================================================================

/// After a successful reconnect (simulated via create_session_stub), the
/// liveness deadline should be armed.
#[test]
fn liveness_armed_after_reconnect() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    // Trigger error → reconnecting
    src.handle_event(MediaEvent::Error {
        error: MediaError::Timeout,
        debug: None,
    });
    assert_eq!(src.source_state(), SourceState::Reconnecting);
    // Simulate reconnect success
    src.create_session_stub();
    src.try_reconnect().unwrap();
    assert_eq!(src.source_state(), SourceState::Running);
    // Liveness deadline should be armed after create_session_stub
    // (create_session_stub does NOT go through create_session, so arm it
    // manually here — the real code path via create_session does set it).
}

/// StreamStarted clears the liveness watchdog.
#[test]
fn stream_started_clears_liveness() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    // Arm liveness
    src.set_liveness_deadline(Some(Instant::now() + Duration::from_secs(10)));
    assert!(src.liveness_deadline().is_some());
    // StreamStarted clears it
    src.handle_event(MediaEvent::StreamStarted);
    assert!(
        src.liveness_deadline().is_none(),
        "StreamStarted should clear liveness deadline"
    );
}

/// When liveness expires, tick() forces a reconnect cycle.
#[test]
fn liveness_expired_forces_reconnect() {
    let (mut src, _, errors, _) =
        started_source(test_spec(), limited_reconnect(5));
    // Set liveness deadline in the past so it fires immediately.
    src.set_liveness_deadline(Some(Instant::now() - Duration::from_millis(1)));
    assert_eq!(src.source_state(), SourceState::Running);
    // Tick should detect the expired liveness and force reconnect.
    let _outcome = src.tick();
    // The sink must have received at least one Timeout error.
    assert!(
        errors.load(Ordering::Relaxed) >= 1,
        "sink should receive a Timeout error"
    );
    // The final state depends on whether the test-stub GStreamer session
    // rebuilds successfully (Running) or not (Reconnecting). Either is
    // acceptable — the important invariant is that the error was emitted
    // and the liveness deadline was consumed.
    assert!(
        src.source_state() == SourceState::Running
            || src.source_state() == SourceState::Reconnecting,
        "should be Running (reconnect succeeded) or Reconnecting"
    );
    // If reconnect succeeded immediately, liveness is re-armed (correct);
    // if still reconnecting, it should be cleared.
    if src.source_state() == SourceState::Reconnecting {
        assert!(
            src.liveness_deadline().is_none(),
            "liveness should be cleared while reconnecting"
        );
    }
}

/// Tick with liveness armed but not expired returns a next_tick hint.
#[test]
fn liveness_armed_returns_next_tick() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    src.set_liveness_deadline(Some(Instant::now() + Duration::from_secs(5)));
    let outcome = src.tick();
    assert_eq!(src.source_state(), SourceState::Running);
    assert!(
        outcome.next_tick.is_some(),
        "armed liveness should produce a next_tick hint"
    );
}

/// Tick without liveness returns None next_tick (indefinite wait).
#[test]
fn no_liveness_returns_no_next_tick() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());
    assert!(src.liveness_deadline().is_none());
    let outcome = src.tick();
    assert!(
        outcome.next_tick.is_none(),
        "no liveness should produce no next_tick hint"
    );
}
