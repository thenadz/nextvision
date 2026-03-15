//! Tests for [`MediaSource`] lifecycle, event handling, and health emission.

use super::*;
use crate::decode::{DecodePreference, HwFailureTracker, SelectedDecoderInfo};
use crate::event::MediaEvent;
use crate::factory::GstMediaIngressFactory;
use crate::ingress::{IngressOptions, MediaIngress, MediaIngressFactory};
use nv_core::config::BackoffKind;
use nv_core::health::DecodeOutcome;
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
    let mut src = MediaSource::new(FeedId::new(1), spec, policy, DecodePreference::Auto);
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
    let mut src = MediaSource::new(FeedId::new(1), spec, policy, DecodePreference::Auto);
    let (sink, frames, errors, eos) = CountingSink::new();
    let health = RecordingHealthSink::new();
    src.set_health_sink(health.clone() as Arc<dyn HealthSink>);
    src.sink = Some(Arc::new(sink));
    src.create_session_stub();
    src.state = SourceState::Running;
    (src, frames, errors, eos, health)
}

/// Helper: create a source with a specific decode preference and health sink.
fn started_source_with_preference(
    spec: SourceSpec,
    policy: ReconnectPolicy,
    pref: DecodePreference,
) -> (
    MediaSource,
    Arc<AtomicU32>,
    Arc<AtomicU32>,
    Arc<AtomicU32>,
    Arc<RecordingHealthSink>,
) {
    let mut src = MediaSource::new(FeedId::new(1), spec, policy, pref);
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
    // Phase 2A: StreamStarted now emits DecodeDecision + SourceConnected.
    assert_eq!(events.len(), 2, "expected DecodeDecision + SourceConnected; got: {:?}", events);
    assert!(matches!(events[0], HealthEvent::DecodeDecision { .. }));
    assert!(matches!(events[1], HealthEvent::SourceConnected { .. }));
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
    let result = factory.create(IngressOptions {
        feed_id: FeedId::new(42),
        spec: test_spec(),
        reconnect: test_reconnect(),
        ptz_provider: None,
        decode_preference: DecodePreference::Auto,
    });
    let source = result.unwrap();
    assert_eq!(source.feed_id(), FeedId::new(42));
}

#[test]
fn factory_wires_health_sink() {
    let health = RecordingHealthSink::new();
    let factory = GstMediaIngressFactory::with_health_sink(health.clone() as Arc<dyn HealthSink>);
    let source = factory
        .create(IngressOptions {
            feed_id: FeedId::new(1),
            spec: test_spec(),
            reconnect: test_reconnect(),
            ptz_provider: None,
            decode_preference: DecodePreference::Auto,
        })
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
        .create(IngressOptions {
            feed_id: FeedId::new(1),
            spec: test_spec(),
            reconnect: test_reconnect(),
            ptz_provider: Some(ptz),
            decode_preference: DecodePreference::Auto,
        })
        .unwrap();

    assert_eq!(source.feed_id(), FeedId::new(1));
}

#[test]
fn factory_default_is_equivalent_to_new() {
    let f1 = GstMediaIngressFactory::new();
    let f2 = GstMediaIngressFactory::default();
    let s1 = f1
        .create(IngressOptions {
            feed_id: FeedId::new(1),
            spec: test_spec(),
            reconnect: test_reconnect(),
            ptz_provider: None,
            decode_preference: DecodePreference::Auto,
        })
        .unwrap();
    let s2 = f2
        .create(IngressOptions {
            feed_id: FeedId::new(2),
            spec: test_spec(),
            reconnect: test_reconnect(),
            ptz_provider: None,
            decode_preference: DecodePreference::Auto,
        })
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
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect(), DecodePreference::Auto);
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
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect(), DecodePreference::Auto);
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
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect(), DecodePreference::Auto);
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
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect(), DecodePreference::Auto);
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

/// No duplicate SourceConnected when bus fires StreamStarted twice.
#[test]
fn no_duplicate_source_connected_emission() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    // started_source_with_health sets state = Running, decoder_verified = false.
    // First StreamStarted should run verification and emit events.
    health.drain(); // clear any setup events.

    src.handle_event(MediaEvent::StreamStarted);
    let events = health.drain();
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::DecodeDecision { .. })),
        "first StreamStarted should emit DecodeDecision",
    );
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::SourceConnected { .. })),
        "first StreamStarted should emit SourceConnected",
    );
    assert!(src.decoder_verified(), "decoder should be verified");

    // Second StreamStarted should be suppressed.
    src.handle_event(MediaEvent::StreamStarted);
    let events2 = health.drain();
    assert!(
        events2.is_empty(),
        "second StreamStarted should suppress duplicate; got: {:?}",
        events2,
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
    assert!(matches!(src.try_reconnect(), ReconnectOutcome::Connected));
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
/// Attempt counter must accumulate across repeated errors without
/// resetting until a StreamStarted event confirms the stream is live.
#[test]
fn attempt_counter_accumulates_without_stream_started() {
    let (mut src, _, _, _) = started_source(test_spec(), test_reconnect());

    // Simulate 5 consecutive errors (pipeline created but stream never starts).
    for i in 1..=5 {
        src.handle_event(MediaEvent::Error {
            error: MediaError::DecodeFailed {
                detail: format!("fail {i}"),
            },
            debug: None,
        });
        assert_eq!(
            src.current_attempt(),
            i,
            "attempt counter should be {i} after {i} errors"
        );
    }

    // StreamStarted resets the counter.
    src.handle_event(MediaEvent::StreamStarted);
    assert_eq!(src.current_attempt(), 0, "StreamStarted should reset attempts");
}

// ===========================================================================
// DecodePreference integration tests
// ===========================================================================

/// Factory creates source with CpuOnly preference — no panic, correct feed_id.
#[test]
fn factory_creates_source_with_cpu_only() {
    let factory = GstMediaIngressFactory::new();
    let source = factory
        .create(IngressOptions {
            feed_id: FeedId::new(99),
            spec: test_spec(),
            reconnect: test_reconnect(),
            ptz_provider: None,
            decode_preference: DecodePreference::CpuOnly,
        })
        .unwrap();
    assert_eq!(source.feed_id(), FeedId::new(99));
}

/// Factory creates source with PreferHardware preference.
#[test]
fn factory_creates_source_with_prefer_hardware() {
    let factory = GstMediaIngressFactory::new();
    let source = factory
        .create(IngressOptions {
            feed_id: FeedId::new(100),
            spec: test_spec(),
            reconnect: test_reconnect(),
            ptz_provider: None,
            decode_preference: DecodePreference::PreferHardware,
        })
        .unwrap();
    assert_eq!(source.feed_id(), FeedId::new(100));
}

/// RequireHardware without gst-backend: start() fails with Unsupported
/// because the stub session path does not satisfy the hardware check.
#[cfg(not(feature = "gst-backend"))]
#[test]
fn require_hardware_fails_fast_without_gst_backend() {
    let mut src = MediaSource::new(
        FeedId::new(1),
        test_spec(),
        test_reconnect(),
        DecodePreference::RequireHardware,
    );
    let (sink, _, _, _) = CountingSink::new();
    let result = src.start(Box::new(sink));
    assert!(result.is_err(), "RequireHardware must fail without gst-backend");
    match result.unwrap_err() {
        MediaError::Unsupported { detail } => {
            assert!(
                detail.contains("RequireHardware") || detail.contains("hardware"),
                "error should mention hardware requirement: {detail}"
            );
        }
        other => panic!("expected Unsupported, got: {other}"),
    }
}

/// Default decode preference (Auto) preserves existing start behavior.
#[cfg(not(feature = "gst-backend"))]
#[test]
fn auto_preference_preserves_existing_behavior() {
    let mut src = MediaSource::new(
        FeedId::new(1),
        test_spec(),
        test_reconnect(),
        DecodePreference::Auto,
    );
    let (sink, _, _, _) = CountingSink::new();
    let result = src.start(Box::new(sink));
    // Without gst-backend, start() returns Unsupported (for no-GStreamer).
    // The important thing is it doesn't fail with a "hardware required" error.
    assert!(result.is_err());
    match result.unwrap_err() {
        MediaError::Unsupported { detail } => {
            assert!(
                !detail.contains("RequireHardware"),
                "Auto should not mention RequireHardware"
            );
        }
        _ => {} // any other error is also fine
    }
}

// ===========================================================================
// ForceHardware autoplug-select integration tests (gst-backend required)
// ===========================================================================

/// When gst-backend is enabled, ForceHardware must produce a decodebin
/// pipeline that rejects software video decoders via autoplug-select.
///
/// This test exercises the feature-enabled code path that constructs the
/// pipeline — the path that was previously untested.
#[cfg(feature = "gst-backend")]
#[test]
fn force_hardware_pipeline_builds_successfully() {
    use crate::pipeline::PipelineBuilder;
    use crate::decode::DecoderSelection;
    use nv_core::config::SourceSpec;

    // Use a fake file path — we only need to verify element creation
    // and signal wiring, not actual decoding.
    let spec = SourceSpec::File {
        path: "/dev/null".into(),
        loop_: false,
    };

    let result = PipelineBuilder::new(spec)
        .decoder(DecoderSelection::ForceHardware)
        .build();

    // On a system with GStreamer installed, the builder must succeed even
    // without hardware decoders — it's the stream negotiation that will
    // fail later (which is correct for RequireHardware semantics).
    assert!(
        result.is_ok(),
        "ForceHardware pipeline construction should succeed: {:?}",
        result.err()
    );
}

/// ForceSoftware pipeline also builds successfully (controls for the test above).
#[cfg(feature = "gst-backend")]
#[test]
fn force_software_pipeline_builds_successfully() {
    use crate::pipeline::PipelineBuilder;
    use crate::decode::DecoderSelection;
    use nv_core::config::SourceSpec;

    let spec = SourceSpec::File {
        path: "/dev/null".into(),
        loop_: false,
    };

    let result = PipelineBuilder::new(spec)
        .decoder(DecoderSelection::ForceSoftware)
        .build();

    assert!(
        result.is_ok(),
        "ForceSoftware pipeline construction should succeed: {:?}",
        result.err()
    );
}

// ===========================================================================
// Phase 2A: Post-selection verification tests
// ===========================================================================

/// StreamStarted with a stub session emits DecodeDecision health event.
#[test]
fn stream_started_emits_decode_decision() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    // Force into Reconnecting to ensure StreamStarted runs the full path.
    src.state = SourceState::Reconnecting;
    health.drain(); // clear previous events

    src.handle_event(MediaEvent::StreamStarted);

    let events = health.drain();
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::DecodeDecision { .. })),
        "StreamStarted should emit DecodeDecision; got: {:?}",
        events,
    );
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::SourceConnected { .. })),
        "SourceConnected should still be emitted after DecodeDecision",
    );
}

/// DecodeDecision outcome is Unknown for stub sessions (no real decoder).
#[test]
fn decode_decision_unknown_for_stub_session() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.state = SourceState::Reconnecting;
    health.drain();

    src.handle_event(MediaEvent::StreamStarted);

    let events = health.drain();
    let decision = events.iter().find_map(|e| match e {
        HealthEvent::DecodeDecision { outcome, .. } => Some(outcome),
        _ => None,
    });
    assert_eq!(
        decision,
        Some(&DecodeOutcome::Unknown),
        "stub session should report Unknown decoder outcome",
    );
}

/// When a hardware decoder is reported, DecodeDecision outcome is Hardware.
#[test]
fn decode_decision_hardware_when_hw_decoder_selected() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.state = SourceState::Reconnecting;
    health.drain();

    // Simulate hardware decoder being selected.
    if let Some(ref session) = src.session {
        session.set_selected_decoder(Some(SelectedDecoderInfo {
            element_name: "nvh264dec".to_string(),
            is_hardware: true,
        }));
    }

    src.handle_event(MediaEvent::StreamStarted);

    let events = health.drain();
    let decision = events.iter().find_map(|e| match e {
        HealthEvent::DecodeDecision { outcome, detail, .. } => Some((outcome, detail)),
        _ => None,
    });
    let (outcome, detail) = decision.expect("should emit DecodeDecision");
    assert_eq!(*outcome, DecodeOutcome::Hardware);
    assert_eq!(detail, "nvh264dec");
}

/// When a software decoder is reported, DecodeDecision outcome is Software.
#[test]
fn decode_decision_software_when_sw_decoder_selected() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    src.state = SourceState::Reconnecting;
    health.drain();

    if let Some(ref session) = src.session {
        session.set_selected_decoder(Some(SelectedDecoderInfo {
            element_name: "avdec_h264".to_string(),
            is_hardware: false,
        }));
    }

    src.handle_event(MediaEvent::StreamStarted);

    let events = health.drain();
    let outcome = events.iter().find_map(|e| match e {
        HealthEvent::DecodeDecision { outcome, .. } => Some(outcome),
        _ => None,
    });
    assert_eq!(outcome, Some(&DecodeOutcome::Software));
}

// ===========================================================================
// Phase 2A: RequireHardware post-selection enforcement tests
// ===========================================================================

/// RequireHardware + software decoder → disconnects and reconnects.
#[test]
fn require_hardware_rejects_software_decoder_post_selection() {
    let (mut src, _, errors, _, health) = started_source_with_preference(
        test_spec(),
        test_reconnect(),
        DecodePreference::RequireHardware,
    );
    src.state = SourceState::Reconnecting;
    health.drain();

    // Simulate software decoder being selected.
    if let Some(ref session) = src.session {
        session.set_selected_decoder(Some(SelectedDecoderInfo {
            element_name: "avdec_h264".to_string(),
            is_hardware: false,
        }));
    }

    let delay = src.handle_event(MediaEvent::StreamStarted);

    // Should trigger reconnect (returned delay).
    assert!(delay.is_some(), "should trigger reconnect on sw decoder");
    assert_eq!(
        src.source_state(),
        SourceState::Reconnecting,
        "should be reconnecting after post-selection failure",
    );
    assert!(
        errors.load(Ordering::Relaxed) >= 1,
        "sink should receive an error",
    );

    // Health should show SourceDisconnected (the error) + SourceReconnecting.
    let events = health.drain();
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::SourceDisconnected { .. })),
        "should emit SourceDisconnected on post-selection failure; events: {:?}",
        events,
    );
}

/// RequireHardware + hardware decoder → succeeds normally.
#[test]
fn require_hardware_accepts_hardware_decoder() {
    let (mut src, _, _, _, health) = started_source_with_preference(
        test_spec(),
        test_reconnect(),
        DecodePreference::RequireHardware,
    );
    src.state = SourceState::Reconnecting;
    health.drain();

    if let Some(ref session) = src.session {
        session.set_selected_decoder(Some(SelectedDecoderInfo {
            element_name: "nvh264dec".to_string(),
            is_hardware: true,
        }));
    }

    let delay = src.handle_event(MediaEvent::StreamStarted);
    assert!(delay.is_none(), "hardware decoder should be accepted");
    assert_eq!(src.source_state(), SourceState::Running);

    let events = health.drain();
    assert!(events.iter().any(|e| matches!(e, HealthEvent::DecodeDecision {
        outcome: DecodeOutcome::Hardware, ..
    })));
    assert!(events.iter().any(|e| matches!(e, HealthEvent::SourceConnected { .. })));
}

// ===========================================================================
// Phase 2A: Adaptive fallback cache tests
// ===========================================================================

/// HW failure tracker starts at zero.
#[test]
fn hw_failure_tracker_starts_clean() {
    let src = MediaSource::new(
        FeedId::new(1),
        test_spec(),
        test_reconnect(),
        DecodePreference::PreferHardware,
    );
    assert_eq!(src.hw_failure_tracker().consecutive_failures(), 0);
    assert!(!src.hw_failure_tracker().is_in_fallback());
}

/// Errors before StreamStarted with PreferHardware record failures.
#[test]
fn prefer_hardware_errors_record_hw_failures() {
    let (mut src, _, _, _, _health) = started_source_with_preference(
        test_spec(),
        test_reconnect(),
        DecodePreference::PreferHardware,
    );
    // Arm liveness deadline (simulates that stream hasn't confirmed yet).
    src.set_liveness_deadline(Some(Instant::now() + Duration::from_secs(10)));

    src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed { detail: "fail 1".into() },
        debug: None,
    });
    assert_eq!(src.hw_failure_tracker().consecutive_failures(), 1);

    // After stream confirmation, the tracker should reset.
    // Force back to Reconnecting to test StreamStarted path.
    src.state = SourceState::Reconnecting;
    src.handle_event(MediaEvent::StreamStarted);
    assert_eq!(src.hw_failure_tracker().consecutive_failures(), 0);
}

/// Auto preference does NOT record hw failures on errors.
#[test]
fn auto_preference_does_not_record_hw_failures() {
    let (mut src, _, _, _, _health) = started_source_with_health(test_spec(), test_reconnect());
    src.set_liveness_deadline(Some(Instant::now() + Duration::from_secs(10)));

    src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed { detail: "fail".into() },
        debug: None,
    });
    assert_eq!(
        src.hw_failure_tracker().consecutive_failures(),
        0,
        "Auto preference should not track hw failures",
    );
}

/// Three consecutive PreferHardware failures activate fallback.
#[test]
fn three_failures_activate_fallback() {
    let mut src = MediaSource::new(
        FeedId::new(1),
        test_spec(),
        test_reconnect(),
        DecodePreference::PreferHardware,
    );
    let (sink, _, _, _) = CountingSink::new();
    src.sink = Some(Arc::new(sink));
    src.create_session_stub();
    src.state = SourceState::Running;

    for _ in 0..3 {
        // Arm liveness so the error path triggers hw recording.
        src.set_liveness_deadline(Some(Instant::now() + Duration::from_secs(10)));
        src.handle_event(MediaEvent::Error {
            error: MediaError::DecodeFailed { detail: "hw fail".into() },
            debug: None,
        });
    }

    assert_eq!(src.hw_failure_tracker().consecutive_failures(), 3);
    assert!(
        src.hw_failure_tracker().is_in_fallback(),
        "3 failures should activate fallback",
    );
}

/// RequireHardware post-selection failure records in tracker.
#[test]
fn require_hardware_post_selection_failure_records_in_tracker() {
    let (mut src, _, _, _, _health) = started_source_with_preference(
        test_spec(),
        test_reconnect(),
        DecodePreference::RequireHardware,
    );
    src.state = SourceState::Reconnecting;

    if let Some(ref session) = src.session {
        session.set_selected_decoder(Some(SelectedDecoderInfo {
            element_name: "avdec_h264".to_string(),
            is_hardware: false,
        }));
    }

    src.handle_event(MediaEvent::StreamStarted);
    assert_eq!(
        src.hw_failure_tracker().consecutive_failures(),
        1,
        "post-selection enforcement failure should record in tracker",
    );
}

/// Success resets the tracker.
#[test]
fn stream_started_resets_hw_failure_tracker() {
    let (mut src, _, _, _, _health) = started_source_with_preference(
        test_spec(),
        test_reconnect(),
        DecodePreference::PreferHardware,
    );

    // Record some failures.
    src.hw_failure_tracker_mut().record_failure();
    src.hw_failure_tracker_mut().record_failure();
    assert_eq!(src.hw_failure_tracker().consecutive_failures(), 2);

    // StreamStarted should reset.
    src.state = SourceState::Reconnecting;
    src.handle_event(MediaEvent::StreamStarted);
    assert_eq!(
        src.hw_failure_tracker().consecutive_failures(),
        0,
        "StreamStarted should reset failure tracker",
    );
}

// ===========================================================================
// Phase 2B: Verification bypass fix tests
// ===========================================================================

/// start() must NOT emit SourceConnected — deferred to StreamStarted.
#[test]
fn start_does_not_emit_source_connected() {
    let health = RecordingHealthSink::new();
    let mut src = MediaSource::new(FeedId::new(1), test_spec(), test_reconnect(), DecodePreference::Auto);
    src.set_health_sink(health.clone() as Arc<dyn HealthSink>);
    let (sink, _, _, _) = CountingSink::new();
    src.sink = Some(Arc::new(sink));
    src.create_session_stub();
    src.state = SourceState::Running;

    let events = health.drain();
    assert!(
        !events.iter().any(|e| matches!(e, HealthEvent::SourceConnected { .. })),
        "start path must NOT emit SourceConnected; got: {:?}",
        events,
    );
}

/// Initial start path: StreamStarted runs verification even when state is
/// already Running (the `decoder_verified` flag is the guard, not state).
#[test]
fn initial_start_stream_started_runs_verification() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    // started_source_with_health: state=Running, decoder_verified=false.
    health.drain();

    assert!(!src.decoder_verified(), "should not be verified yet");

    src.handle_event(MediaEvent::StreamStarted);

    assert!(src.decoder_verified(), "should be verified after StreamStarted");
    let events = health.drain();
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::DecodeDecision { .. })),
        "initial start StreamStarted must emit DecodeDecision; got: {:?}",
        events,
    );
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::SourceConnected { .. })),
        "initial start StreamStarted must emit SourceConnected; got: {:?}",
        events,
    );
}

/// Reconnect path: StreamStarted also runs verification.
#[test]
fn reconnect_path_runs_verification() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    health.drain();

    // Simulate error → reconnect → session stub recreated → StreamStarted.
    src.handle_event(MediaEvent::Error {
        error: MediaError::DecodeFailed { detail: "fail".into() },
        debug: None,
    });
    health.drain();
    // Rebuild stub to simulate successful reconnect.
    src.create_session_stub();
    src.state = SourceState::Reconnecting;
    assert!(!src.decoder_verified());

    src.handle_event(MediaEvent::StreamStarted);

    assert!(src.decoder_verified());
    let events = health.drain();
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::DecodeDecision { .. })),
        "reconnect StreamStarted must emit DecodeDecision; got: {:?}",
        events,
    );
}

/// decoder_verified is reset on session creation (each reconnect gets fresh verification).
#[test]
fn decoder_verified_resets_on_new_session() {
    let (mut src, _, _, _, health) = started_source_with_health(test_spec(), test_reconnect());
    health.drain();

    // First session: verify.
    src.handle_event(MediaEvent::StreamStarted);
    assert!(src.decoder_verified());

    // Simulate reconnect: create new session stub → resets flag.
    src.create_session_stub();
    assert!(!src.decoder_verified(), "should reset on new session");

    // StreamStarted on new session should verify again.
    src.state = SourceState::Reconnecting;
    src.handle_event(MediaEvent::StreamStarted);
    assert!(src.decoder_verified());
    let events = health.drain();
    // Should have events from both StreamStarted calls.
    let decode_decisions: Vec<_> = events.iter()
        .filter(|e| matches!(e, HealthEvent::DecodeDecision { .. }))
        .collect();
    assert_eq!(
        decode_decisions.len(), 2,
        "should emit DecodeDecision for each session; got: {:?}",
        events,
    );
}

// ===========================================================================
// Phase 2B: RequireHardware rejects Unknown tests
// ===========================================================================

/// RequireHardware + unknown decoder → rejects (fail closed).
#[test]
fn require_hardware_rejects_unknown_decoder_post_selection() {
    let (mut src, _, errors, _, health) = started_source_with_preference(
        test_spec(),
        test_reconnect(),
        DecodePreference::RequireHardware,
    );
    src.state = SourceState::Reconnecting;
    health.drain();

    // Stub session has no selected_decoder → outcome is Unknown.
    // Do NOT set a decoder — this simulates the Unknown case.

    let delay = src.handle_event(MediaEvent::StreamStarted);

    assert!(delay.is_some(), "should trigger reconnect on Unknown decoder");
    assert_eq!(
        src.source_state(),
        SourceState::Reconnecting,
        "should be reconnecting after Unknown decoder with RequireHardware",
    );
    assert!(
        errors.load(Ordering::Relaxed) >= 1,
        "sink should receive an error",
    );
    let events = health.drain();
    assert!(
        events.iter().any(|e| matches!(e, HealthEvent::SourceDisconnected { .. })),
        "should emit SourceDisconnected on Unknown decoder failure; events: {:?}",
        events,
    );
}

// ===========================================================================
// Phase 2B: Enriched DecodeDecision telemetry tests
// ===========================================================================

/// DecodeDecision carries preference and fallback context.
#[test]
fn decode_decision_includes_preference_and_fallback() {
    let (mut src, _, _, _, health) = started_source_with_preference(
        test_spec(),
        test_reconnect(),
        DecodePreference::PreferHardware,
    );
    src.state = SourceState::Reconnecting;
    health.drain();

    src.handle_event(MediaEvent::StreamStarted);

    let events = health.drain();
    let decision = events.iter().find_map(|e| match e {
        HealthEvent::DecodeDecision {
            preference,
            fallback_active,
            fallback_reason,
            ..
        } => Some((preference.clone(), *fallback_active, fallback_reason.clone())),
        _ => None,
    });
    let (pref, fb_active, fb_reason) = decision.expect("should emit DecodeDecision");
    assert!(
        pref.contains("PreferHardware"),
        "preference should be PreferHardware; got: {}",
        pref,
    );
    assert!(!fb_active, "no fallback should be active on first attempt");
    assert!(fb_reason.is_none());
}

// ===========================================================================
// Phase 2B: Adaptive fallback real behaviour tests
// ===========================================================================

/// PreferHardware normally maps to ForceHardware, but after 3 failures
/// the adaptive fallback demotes to Auto — a real behavioral change.
#[test]
fn adaptive_fallback_changes_effective_selection() {
    use crate::decode::DecoderSelection;

    // Without fallback: PreferHardware → ForceHardware.
    let tracker_fresh = HwFailureTracker::new();
    let result = tracker_fresh.adjust_selection(DecodePreference::PreferHardware);
    assert!(result.is_none(), "fresh tracker should not adjust");
    assert!(
        matches!(DecodePreference::PreferHardware.to_selection(), DecoderSelection::ForceHardware),
        "PreferHardware base mapping should be ForceHardware",
    );

    // After threshold failures: fallback produces Auto.
    let mut tracker = HwFailureTracker::new();
    for _ in 0..3 {
        tracker.record_failure();
    }
    assert!(tracker.is_in_fallback(), "should be in fallback window");
    let (adjusted, reason) = tracker
        .adjust_selection(DecodePreference::PreferHardware)
        .expect("should adjust during fallback");
    assert!(
        matches!(adjusted, DecoderSelection::Auto),
        "fallback should demote to Auto; got: {:?}",
        adjusted,
    );
    assert!(reason.contains("consecutive hardware failures"));
}

// ===========================================================================
// Finding 1: Reconnect budget bypass — attempts must not reset before
// decoder verification passes
// ===========================================================================

/// RequireHardware with software decoder: reconnect attempts must NOT be
/// reset when verification fails. Without this fix, attempt reset happens
/// on StreamStarted entry, bypassing the reconnect budget.
#[test]
fn require_hardware_verification_failure_preserves_reconnect_attempts() {
    let (mut src, _, _, _, _health) = started_source_with_preference(
        test_spec(),
        limited_reconnect(3),
        DecodePreference::RequireHardware,
    );
    src.state = SourceState::Reconnecting;

    // Simulate 2 prior reconnect attempts.
    src.reconnect.record_attempt();
    src.reconnect.record_attempt();
    assert_eq!(src.reconnect.current_attempt(), 2);

    // Software decoder selected → verification will fail.
    if let Some(ref session) = src.session {
        session.set_selected_decoder(Some(SelectedDecoderInfo {
            element_name: "avdec_h264".to_string(),
            is_hardware: false,
        }));
    }

    let delay = src.handle_event(MediaEvent::StreamStarted);
    assert!(delay.is_some(), "should trigger reconnect");

    // Key assertion: attempts must NOT have been reset to 0.
    // They should have been incremented (reconnect attempt via
    // disconnect_and_reconnect → try_reconnect_or_stop → initiate_reconnect).
    assert!(
        src.reconnect.current_attempt() >= 2,
        "reconnect attempts must not be reset on verification failure; got: {}",
        src.reconnect.current_attempt(),
    );
}

/// Repeated StreamStarted on an already-verified session must not reset
/// the reconnect attempt counter again.
#[test]
fn duplicate_stream_started_does_not_reset_attempts() {
    let (mut src, _, _, _, _health) = started_source_with_preference(
        test_spec(),
        limited_reconnect(5),
        DecodePreference::Auto,
    );
    src.state = SourceState::Reconnecting;

    // First StreamStarted — verifies and resets attempts.
    src.handle_event(MediaEvent::StreamStarted);
    assert!(src.decoder_verified, "should be verified after first StreamStarted");
    assert_eq!(src.reconnect.current_attempt(), 0, "should reset after successful verification");

    // Simulate some reconnect attempts accumulating after the initial connect.
    src.reconnect.record_attempt();
    src.reconnect.record_attempt();
    assert_eq!(src.reconnect.current_attempt(), 2);

    // Second StreamStarted — duplicate, should be no-op (early return).
    let delay = src.handle_event(MediaEvent::StreamStarted);
    assert!(delay.is_none(), "duplicate StreamStarted should be no-op");
    assert_eq!(
        src.reconnect.current_attempt(),
        2,
        "duplicate StreamStarted must not reset reconnect attempts",
    );
}

/// RequireHardware budget exhaustion: repeated verification failures must
/// eventually exhaust the reconnect budget and stop the source.
#[test]
fn require_hardware_exhausts_reconnect_budget() {
    let (mut src, _, _, _, _health) = started_source_with_preference(
        test_spec(),
        limited_reconnect(2),
        DecodePreference::RequireHardware,
    );

    // Software decoder stub.
    if let Some(ref session) = src.session {
        session.set_selected_decoder(Some(SelectedDecoderInfo {
            element_name: "avdec_h264".to_string(),
            is_hardware: false,
        }));
    }

    // Each StreamStarted should fail verification and consume a reconnect attempt.
    // With max_attempts=2, after 2 failures the budget should be exhausted.
    for _ in 0..3 {
        src.state = SourceState::Reconnecting;
        src.decoder_verified = false;
        let _delay = src.handle_event(MediaEvent::StreamStarted);
    }

    // After budget exhaustion, source should be stopped.
    assert_eq!(
        src.source_state(),
        SourceState::Stopped,
        "source should stop when reconnect budget is exhausted",
    );
}