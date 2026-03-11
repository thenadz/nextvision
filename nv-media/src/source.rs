//! Source lifecycle management and reconnection.
//!
//! [`MediaSource`] owns the GStreamer session for a single feed and manages
//! its complete lifecycle: start, pause, resume, reconnect, stop.
//!
//! Implements the [`MediaIngress`] trait so the runtime interacts with
//! sources through the trait contract, not the concrete type.
//!
//! # Reconnection behavior
//!
//! When the pipeline reports a fatal error the source enters a reconnection
//! loop governed by a [`ReconnectPolicy`]:
//!
//! 1. The current GStreamer session is torn down.
//! 2. A backoff delay is computed (exponential or linear).
//! 3. After the delay, a new session is constructed.
//! 4. If construction succeeds, frames resume flowing.
//! 5. If it fails and `max_attempts` is exhausted, the source stops permanently.
//!
//! During reconnection, `on_error()` is called on the sink for each failure,
//! and `on_eos()` is called if the reconnection budget is exhausted.
//!
//! # PTS discontinuity detection
//!
//! PTS discontinuity detection is handled by the backend's appsink callback
//! thread, which pushes [`MediaEvent::Discontinuity`] events into a bounded
//! [`EventQueue`]. The source's [`poll_bus()`] drains these alongside
//! GStreamer bus messages.
//!
//! # Restart semantics
//!
//! A source that has been `stop()`-ed cannot be restarted — `stop()` is
//! terminal. A *reconnection* (which happens internally after a pipeline
//! error) is **not** the same as a restart: the source stays in the
//! `Reconnecting` state and re-enters `Running` when the new session
//! succeeds. From the runtime's perspective, the feed is still alive
//! during reconnection.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::MediaError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;

use crate::backend::{EventQueue, GstSession, SessionConfig};
use crate::ingress::PtzProvider;
use crate::decode::DecoderSelection;
use crate::event::MediaEvent;
use crate::ingress::{FrameSink, HealthSink, MediaIngress, SourceStatus};
use crate::pipeline::OutputFormat;
use crate::reconnect::ReconnectTracker;

/// Maximum time to wait for a stream to start producing frames after a
/// successful (re)connection before treating the session as dead. This
/// ensures the worker keeps polling the bus even when no frames arrive.
const LIVENESS_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Source state machine
// ---------------------------------------------------------------------------

/// Lifecycle state of the media source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceState {
    /// Created, not yet started.
    Idle,
    /// Session running, frames flowing.
    Running,
    /// Session paused — connection held open, no frames.
    Paused,
    /// Session failed, attempting reconnection.
    Reconnecting,
    /// Stopped — terminal state.
    Stopped,
}

// ---------------------------------------------------------------------------
// MediaSource
// ---------------------------------------------------------------------------

/// Handle to a running media source for a single feed.
///
/// Owns the GStreamer session (pipeline lifecycle) and manages reconnection
/// with configurable backoff. Implements [`MediaIngress`] so the runtime
/// layer interacts through the trait, not this concrete type.
pub struct MediaSource {
    feed_id: FeedId,
    spec: SourceSpec,
    state: SourceState,
    reconnect: ReconnectTracker,
    session: Option<GstSession>,
    sink: Option<Arc<dyn FrameSink>>,
    /// Optional health event sink — receives lifecycle events mapped from
    /// internal [`MediaEvent`]s. Set via [`set_health_sink()`].
    health_sink: Option<Arc<dyn HealthSink>>,
    /// Optional PTZ telemetry provider — passed through to the GStreamer
    /// session so frames can carry PTZ metadata.
    ptz_provider: Option<Arc<dyn PtzProvider>>,
    /// Shared event queue: the appsink callback pushes discontinuity events
    /// here, and [`poll_bus()`] drains them alongside GStreamer bus messages.
    event_queue: EventQueue,
    /// Earliest `Instant` at which the next reconnect attempt is allowed.
    /// Set when a backoff delay is computed; cleared on successful connect.
    reconnect_deadline: Option<Instant>,
    /// Liveness watchdog: deadline by which the new session must produce a
    /// `StreamStarted` event. Set after a successful (re)connect; cleared
    /// once the stream is confirmed flowing. If it expires, a reconnection
    /// is forced.
    liveness_deadline: Option<Instant>,
}

impl MediaSource {
    /// Create a new media source (does not start the pipeline).
    #[must_use]
    pub fn new(feed_id: FeedId, spec: SourceSpec, reconnect_policy: ReconnectPolicy) -> Self {
        Self {
            feed_id,
            spec,
            state: SourceState::Idle,
            reconnect: ReconnectTracker::new(reconnect_policy),
            session: None,
            sink: None,
            health_sink: None,
            ptz_provider: None,
            event_queue: Arc::new(std::sync::Mutex::new(VecDeque::new())),
            reconnect_deadline: None,
            liveness_deadline: None,
        }
    }

    /// Attach a health event sink for lifecycle reporting.
    ///
    /// The source emits [`HealthEvent`]s as it processes internal media
    /// events (connection, disconnection, reconnection attempts, etc.).
    /// Must be called before [`start()`](MediaIngress::start).
    pub fn set_health_sink(&mut self, sink: Arc<dyn HealthSink>) {
        self.health_sink = Some(sink);
    }

    /// Attach a PTZ telemetry provider.
    ///
    /// The provider is queried on every decoded frame to attach PTZ metadata.
    /// Must be called before [`start()`](MediaIngress::start).
    pub fn set_ptz_provider(&mut self, provider: Arc<dyn PtzProvider>) {
        self.ptz_provider = Some(provider);
    }

    /// Emit a health event if a health sink is attached.
    fn emit_health(&self, event: HealthEvent) {
        if let Some(ref hs) = self.health_sink {
            hs.emit(event);
        }
    }

    /// Enrich a [`MediaError`] with source context.
    ///
    /// For `ConnectionFailed` errors produced by `classify_bus_error`, the
    /// `url` field is empty because the bus message doesn't carry it. This
    /// method fills it in from the source spec so health consumers get a
    /// useful URL for diagnostics.
    fn enrich_error(&self, error: MediaError) -> MediaError {
        match error {
            MediaError::ConnectionFailed { url, detail } if url.is_empty() => {
                MediaError::ConnectionFailed {
                    url: self.source_url().unwrap_or_default(),
                    detail,
                }
            }
            other => other,
        }
    }

    /// Extract the URL (or path) from the source spec for diagnostics.
    fn source_url(&self) -> Option<String> {
        match &self.spec {
            SourceSpec::Rtsp { url, .. } => Some(url.clone()),
            SourceSpec::File { path, .. } => Some(path.display().to_string()),
            SourceSpec::V4l2 { device } => Some(device.clone()),
            SourceSpec::Custom { .. } => None,
        }
    }

    /// Attempt to create and start a new GStreamer session.
    ///
    /// On success the session is stored and PTS tracking is reset.
    /// On failure the session field is left as `None`.
    fn create_session(&mut self) -> Result<(), MediaError> {
        let config = SessionConfig {
            feed_id: self.feed_id,
            spec: self.spec.clone(),
            decoder: DecoderSelection::Auto,
            output_format: OutputFormat::default(),
            ptz_provider: self.ptz_provider.clone(),
        };
        let sink = self
            .sink
            .as_ref()
            .ok_or_else(|| MediaError::Unsupported {
                detail: "sink must be set before creating session".into(),
            })?
            .clone();

        // Clear the event queue before creating a new session so stale
        // discontinuity events from a previous session are not processed.
        if let Ok(mut q) = self.event_queue.lock() {
            q.clear();
        }

        let session = GstSession::start(config, sink, Arc::clone(&self.event_queue))?;
        self.session = Some(session);
        // Arm the liveness watchdog — the new session must produce a
        // StreamStarted within LIVENESS_TIMEOUT or we force a reconnect.
        self.liveness_deadline = Some(Instant::now() + LIVENESS_TIMEOUT);
        Ok(())
    }

    /// Test-only: create a stub session that reports as running.
    #[cfg(test)]
    fn create_session_stub(&mut self) {
        let config = SessionConfig {
            feed_id: self.feed_id,
            spec: self.spec.clone(),
            decoder: DecoderSelection::Auto,
            output_format: OutputFormat::default(),
            ptz_provider: None,
        };
        self.session = Some(GstSession::start_stub(config));
    }

    /// Initiate a reconnection cycle.
    ///
    /// Tears down the current session (if any), checks the policy budget,
    /// records the attempt, and returns the backoff delay.
    ///
    /// Returns `Ok(delay)` if another attempt is allowed, or `Err` if the
    /// reconnection budget is exhausted.
    fn initiate_reconnect(&mut self) -> Result<Duration, MediaError> {
        // Tear down current session.
        if let Some(ref mut session) = self.session {
            let _ = session.stop();
        }
        self.session = None;

        if !self.reconnect.can_retry() {
            return Err(MediaError::Unsupported {
                detail: format!(
                    "reconnection limit reached ({} attempts)",
                    self.reconnect.current_attempt()
                ),
            });
        }

        let delay = self.reconnect.backoff_delay();
        self.reconnect.record_attempt();
        self.state = SourceState::Reconnecting;
        Ok(delay)
    }

    /// Process a media event from the backend.
    ///
    /// Called by the source's event loop (driven by the runtime). The return
    /// value tells the caller whether a reconnection attempt should be
    /// scheduled after a delay.
    ///
    /// Returns `Some(delay)` if reconnection should be retried after waiting,
    /// or `None` if no action is needed.
    pub(crate) fn handle_event(&mut self, event: MediaEvent) -> Option<Duration> {
        match event {
            MediaEvent::StreamStarted => {
                // Stream is confirmed flowing — clear the liveness watchdog
                // and reset the reconnection attempt counter unconditionally.
                self.liveness_deadline = None;
                self.reconnect.reset_attempts();
                if self.state == SourceState::Running {
                    // Already running — suppress duplicate emission.
                    // This can happen when start() transitions to Running
                    // and the GStreamer bus subsequently fires StreamStarted.
                    return None;
                }
                self.state = SourceState::Running;
                self.reconnect_deadline = None;
                tracing::info!(feed_id = %self.feed_id, "stream started");
                self.emit_health(HealthEvent::SourceConnected {
                    feed_id: self.feed_id,
                });
                None
            }
            MediaEvent::Eos => {
                tracing::info!(feed_id = %self.feed_id, "end of stream");
                match &self.spec {
                    SourceSpec::File { loop_: true, .. } => {
                        // Looping file: seek back to start instead of
                        // reconnecting or stopping.
                        if let Some(ref mut session) = self.session {
                            if session.seek_start().is_ok() {
                                tracing::info!(
                                    feed_id = %self.feed_id,
                                    "file source looping — seeked to start"
                                );
                                return None;
                            }
                        }
                        // If seek failed, fall through to reconnect which
                        // will rebuild the pipeline from scratch.
                        tracing::warn!(
                            feed_id = %self.feed_id,
                            "file loop seek failed, falling back to reconnect"
                        );
                        self.try_reconnect_or_stop()
                    }
                    SourceSpec::File { loop_: false, .. } => {
                        // Non-looping file: EOS is terminal.
                        if let Some(ref sink) = self.sink {
                            sink.on_eos();
                        }
                        self.state = SourceState::Stopped;
                        // FeedStopped is emitted by the worker thread
                        // (the canonical owner of feed lifecycle events).
                        None
                    }
                    _ => {
                        // Live sources: attempt reconnection.
                        self.try_reconnect_or_stop()
                    }
                }
            }
            MediaEvent::Error { error, debug: debug_detail } => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    error = %error,
                    debug_detail = ?debug_detail,
                    "pipeline error"
                );
                // Clone the classified error so both health and sink receive
                // the original typed variant (not a lossy display string).
                let enriched = self.enrich_error(error);
                self.emit_health(HealthEvent::SourceDisconnected {
                    feed_id: self.feed_id,
                    reason: enriched.clone(),
                });
                if let Some(ref sink) = self.sink {
                    sink.on_error(enriched);
                }
                self.try_reconnect_or_stop()
            }
            MediaEvent::Warning { message, debug: debug_detail } => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    message = %message,
                    debug_detail = ?debug_detail,
                    "pipeline warning"
                );
                None
            }
            MediaEvent::Discontinuity {
                gap_ns,
                prev_pts_ns,
                current_pts_ns,
            } => {
                tracing::info!(
                    feed_id = %self.feed_id,
                    gap_ms = gap_ns / 1_000_000,
                    prev_pts_ms = prev_pts_ns / 1_000_000,
                    current_pts_ms = current_pts_ns / 1_000_000,
                    "PTS discontinuity detected"
                );
                None
            }
        }
    }

    /// Attempt reconnection or stop permanently if budget is exhausted.
    fn try_reconnect_or_stop(&mut self) -> Option<Duration> {
        match self.initiate_reconnect() {
            Ok(delay) => {
                tracing::info!(
                    feed_id = %self.feed_id,
                    attempt = self.reconnect.current_attempt(),
                    delay_ms = delay.as_millis() as u64,
                    "scheduling reconnection"
                );
                self.emit_health(HealthEvent::SourceReconnecting {
                    feed_id: self.feed_id,
                    attempt: self.reconnect.current_attempt(),
                });
                Some(delay)
            }
            Err(_) => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    total_attempts = self.reconnect.current_attempt(),
                    "reconnection budget exhausted, stopping"
                );
                self.state = SourceState::Stopped;
                if let Some(ref sink) = self.sink {
                    sink.on_eos();
                }
                // FeedStopped is emitted by the worker thread
                // (the canonical owner of feed lifecycle events).
                None
            }
        }
    }

    /// Poll the GStreamer bus for pending messages and process them.
    ///
    /// This is the operational connection between the bus message types
    /// and the source lifecycle FSM. Should be called periodically from the
    /// source's management loop (e.g., on a timer or after each frame batch).
    ///
    /// Returns `Some(delay)` if a reconnection should be scheduled after
    /// waiting, or `None` if nothing actionable occurred.
    pub fn poll_bus(&mut self) -> Option<Duration> {
        // Drain discontinuity events produced by the appsink callback thread.
        let queued_events: Vec<MediaEvent> = match self.event_queue.lock() {
            Ok(mut q) => q.drain(..).collect(),
            Err(_) => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    "event queue lock poisoned, skipping queued events"
                );
                Vec::new()
            }
        };

        if !queued_events.is_empty() {
            tracing::debug!(
                feed_id = %self.feed_id,
                count = queued_events.len(),
                "poll_bus: draining queued discontinuity events"
            );
        }

        for event in queued_events {
            if let Some(delay) = self.handle_event(event) {
                self.reconnect_deadline = Some(Instant::now() + delay);
                return Some(delay);
            }
        }

        // Drain GStreamer bus messages.
        let mut messages = Vec::new();
        if let Some(ref session) = self.session {
            while let Some(bus_msg) = session.poll_bus() {
                messages.push(bus_msg);
            }
        } else {
            tracing::debug!(
                feed_id = %self.feed_id,
                state = ?self.state,
                "poll_bus: no session — skipping bus drain"
            );
        }

        if !messages.is_empty() {
            tracing::trace!(
                feed_id = %self.feed_id,
                count = messages.len(),
                "poll_bus: draining GStreamer bus messages"
            );
        }

        let mut reconnect_delay = None;
        for bus_msg in messages {
            tracing::trace!(
                feed_id = %self.feed_id,
                bus_msg = ?bus_msg,
                "poll_bus: processing bus message"
            );
            if let Some(event) = bus_msg.into_media_event() {
                tracing::debug!(
                    feed_id = %self.feed_id,
                    event = ?event,
                    "poll_bus: mapped to media event"
                );
                if let Some(delay) = self.handle_event(event) {
                    self.reconnect_deadline = Some(Instant::now() + delay);
                    reconnect_delay = Some(delay);
                    break;
                }
            }
        }

        // If in Reconnecting state and no new delay was just scheduled,
        // attempt reconnection only if the backoff deadline has elapsed.
        if reconnect_delay.is_none() && self.state == SourceState::Reconnecting {
            let deadline_elapsed = self
                .reconnect_deadline
                .map_or(true, |d| Instant::now() >= d);
            tracing::debug!(
                feed_id = %self.feed_id,
                deadline_elapsed,
                reconnect_deadline = ?self.reconnect_deadline,
                "poll_bus: in Reconnecting state, checking deadline"
            );
            if deadline_elapsed {
                match self.try_reconnect() {
                    Ok(()) => {
                        tracing::info!(
                            feed_id = %self.feed_id,
                            "poll_bus: reconnection succeeded"
                        );
                        self.reconnect_deadline = None;
                    }
                    Err(Some(delay)) => {
                        tracing::debug!(
                            feed_id = %self.feed_id,
                            delay_ms = delay.as_millis() as u64,
                            "poll_bus: reconnection failed, scheduling retry"
                        );
                        self.reconnect_deadline = Some(Instant::now() + delay);
                        reconnect_delay = Some(delay);
                    }
                    Err(None) => {
                        tracing::warn!(
                            feed_id = %self.feed_id,
                            "poll_bus: reconnection budget exhausted, stopped"
                        );
                    }
                }
            }
        }

        reconnect_delay
    }

    /// Attempt a single reconnection: tear down the old session and create a new one.
    ///
    /// This method is meant to be called by the runtime after the backoff delay
    /// returned by [`handle_event()`] or [`poll_bus()`] has elapsed.
    ///
    /// Returns:
    /// - `Ok(())` if a new session was successfully created and is running.
    /// - `Err(delay)` if the session creation failed but another attempt is
    ///   allowed — the caller should wait `delay` before trying again.
    /// - `Err` where `self.state == Stopped` means the reconnection budget is
    ///   exhausted and the source is permanently stopped.
    pub fn try_reconnect(&mut self) -> Result<(), Option<Duration>> {
        if self.state == SourceState::Stopped {
            return Err(None);
        }
        if self.state != SourceState::Reconnecting {
            return Ok(());
        }

        match self.create_session() {
            Ok(()) => {
                self.state = SourceState::Running;
                // NOTE: attempt counter is NOT reset here. Pipeline
                // construction succeeding does not mean the stream is
                // actually flowing — the camera may still be unreachable
                // and GStreamer will surface a bus error shortly.
                // reset_attempts() happens in handle_event(StreamStarted)
                // once frames are confirmed flowing.
                tracing::info!(
                    feed_id = %self.feed_id,
                    "reconnected successfully"
                );
                // SourceConnected is emitted when handle_event receives
                // StreamStarted from the bus; no duplicate emission here.
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    error = %e,
                    attempt = self.reconnect.current_attempt(),
                    "reconnection attempt failed"
                );
                if let Some(ref sink) = self.sink {
                    sink.on_error(e);
                }
                // Check if we can retry
                match self.initiate_reconnect() {
                    Ok(delay) => {
                        self.emit_health(HealthEvent::SourceReconnecting {
                            feed_id: self.feed_id,
                            attempt: self.reconnect.current_attempt(),
                        });
                        Err(Some(delay))
                    }
                    Err(_) => {
                        self.state = SourceState::Stopped;
                        if let Some(ref sink) = self.sink {
                            sink.on_eos();
                        }
                        // FeedStopped is emitted by the worker thread
                        // (the canonical owner of feed lifecycle events).
                        Err(None)
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MediaIngress implementation
// ---------------------------------------------------------------------------

impl MediaIngress for MediaSource {
    fn start(&mut self, sink: Box<dyn FrameSink>) -> Result<(), MediaError> {
        match self.state {
            SourceState::Idle => {}
            SourceState::Stopped => {
                return Err(MediaError::Unsupported {
                    detail: "source has been stopped and cannot be restarted".into(),
                });
            }
            _ => {
                return Err(MediaError::Unsupported {
                    detail: "source is already running or reconnecting".into(),
                });
            }
        }

        self.sink = Some(Arc::from(sink));

        match self.create_session() {
            Ok(()) => {
                self.state = SourceState::Running;
                tracing::info!(feed_id = %self.feed_id, "source started");
                self.emit_health(HealthEvent::SourceConnected {
                    feed_id: self.feed_id,
                });
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    error = %e,
                    "initial connection failed"
                );
                // Return the error so the caller can decide whether to
                // retry. This matches the documented contract: initial
                // connection failure is surfaced as Err.
                Err(e)
            }
        }
    }

    fn stop(&mut self) -> Result<(), MediaError> {
        if self.state == SourceState::Stopped {
            return Ok(());
        }
        if let Some(ref mut session) = self.session {
            session.stop()?;
        }
        self.session = None;
        if let Some(ref sink) = self.sink {
            sink.on_eos();
        }
        self.state = SourceState::Stopped;
        // Note: FeedStopped health events are emitted by the worker thread
        // with the correct StopReason. The source does not emit them.
        tracing::info!(feed_id = %self.feed_id, "source stopped");
        Ok(())
    }

    fn pause(&mut self) -> Result<(), MediaError> {
        if self.state != SourceState::Running {
            return Err(MediaError::Unsupported {
                detail: "can only pause a running source".into(),
            });
        }
        if let Some(ref mut session) = self.session {
            session.pause()?;
        }
        self.state = SourceState::Paused;
        Ok(())
    }

    fn resume(&mut self) -> Result<(), MediaError> {
        if self.state != SourceState::Paused {
            return Err(MediaError::Unsupported {
                detail: "can only resume a paused source".into(),
            });
        }
        if let Some(ref mut session) = self.session {
            session.resume()?;
        }
        self.state = SourceState::Running;
        Ok(())
    }

    fn source_spec(&self) -> &SourceSpec {
        &self.spec
    }

    fn feed_id(&self) -> FeedId {
        self.feed_id
    }

    fn tick(&mut self) -> crate::ingress::TickOutcome {
        use crate::ingress::TickOutcome;

        // If already stopped or idle, return immediately.
        match self.state {
            SourceState::Stopped => return TickOutcome::stopped(),
            SourceState::Idle => return TickOutcome::running(),
            _ => {}
        }

        // Drive bus polling — this processes pending GStreamer messages and
        // pending discontinuity events, advancing the reconnection FSM.
        let _reconnect_delay = self.poll_bus();

        // Check liveness watchdog: if armed and expired, force reconnect.
        if self.state == SourceState::Running {
            if let Some(deadline) = self.liveness_deadline {
                if Instant::now() >= deadline {
                    tracing::warn!(
                        feed_id = %self.feed_id,
                        "liveness watchdog expired — no stream started, forcing reconnect"
                    );
                    self.liveness_deadline = None;
                    self.emit_health(HealthEvent::SourceDisconnected {
                        feed_id: self.feed_id,
                        reason: MediaError::Timeout,
                    });
                    if let Some(ref sink) = self.sink {
                        sink.on_error(MediaError::Timeout);
                    }
                    self.try_reconnect_or_stop();
                    // Re-poll to pick up the reconnection state.
                    let _delay = self.poll_bus();
                }
            }
        }

        // Map the resulting state and compute the next-tick hint.
        let outcome = match self.state {
            SourceState::Running | SourceState::Paused | SourceState::Idle => {
                // If liveness watchdog is armed, schedule a tick so the
                // worker keeps polling instead of waiting indefinitely.
                let next = self.liveness_deadline.map(|d| {
                    d.saturating_duration_since(Instant::now())
                });
                TickOutcome { status: SourceStatus::Running, next_tick: next }
            }
            SourceState::Reconnecting => {
                // Return the remaining backoff as the next-tick hint so
                // the worker can sleep precisely instead of polling.
                let remaining = self
                    .reconnect_deadline
                    .map(|d| d.saturating_duration_since(Instant::now()))
                    .unwrap_or(Duration::ZERO);
                TickOutcome::reconnecting(remaining)
            }
            SourceState::Stopped => TickOutcome::stopped(),
        };

        outcome
    }
}

// ---------------------------------------------------------------------------
// Tests (extracted to source_tests.rs for maintainability)
// ---------------------------------------------------------------------------

#[cfg(test)]
impl MediaSource {
    pub(crate) fn source_state(&self) -> SourceState {
        self.state
    }

    pub(crate) fn total_reconnects(&self) -> u32 {
        self.reconnect.total_reconnects()
    }

    pub(crate) fn liveness_deadline(&self) -> Option<Instant> {
        self.liveness_deadline
    }

    pub(crate) fn set_liveness_deadline(&mut self, deadline: Option<Instant>) {
        self.liveness_deadline = deadline;
    }

    pub(crate) fn current_attempt(&self) -> u32 {
        self.reconnect.current_attempt()
    }
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;


