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

mod event_handling;
mod ingress_impl;
mod polling;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::MediaError;
use nv_core::health::{DecodeOutcome, HealthEvent};
use nv_core::id::FeedId;

use crate::backend::{EventQueue, GstSession, SessionConfig};
use crate::decode::{DecodePreference, DecodePreferenceExt, HwFailureTracker};
use crate::hook::PostDecodeHook;
use crate::ingress::{FrameSink, HealthSink, PtzProvider};
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

/// Outcome of a [`MediaSource::try_reconnect`] call.
#[derive(Debug)]
pub(crate) enum ReconnectOutcome {
    /// A new session was created and is running.
    Connected,
    /// Session creation failed but another attempt is allowed after `delay`.
    Retry { delay: Duration },
    /// Reconnection budget exhausted or source stopped — no more retries.
    Exhausted,
}

// ---------------------------------------------------------------------------

/// Handle to a running media source for a single feed.
///
/// Owns the GStreamer session (pipeline lifecycle) and manages reconnection
/// with configurable backoff. Implements [`MediaIngress`] so the runtime
/// layer interacts through the trait, not this concrete type.
pub struct MediaSource {
    pub(super) feed_id: FeedId,
    pub(super) spec: SourceSpec,
    pub(super) state: SourceState,
    pub(super) reconnect: ReconnectTracker,
    pub(super) session: Option<GstSession>,
    pub(super) sink: Option<Arc<dyn FrameSink>>,
    /// Optional health event sink — receives lifecycle events mapped from
    /// internal [`MediaEvent`]s. Set via [`set_health_sink()`].
    pub(super) health_sink: Option<Arc<dyn HealthSink>>,
    /// Optional PTZ telemetry provider — passed through to the GStreamer
    /// session so frames can carry PTZ metadata.
    pub(super) ptz_provider: Option<Arc<dyn PtzProvider>>,
    /// Shared event queue: the appsink callback pushes discontinuity events
    /// here, and [`poll_bus()`] drains them alongside GStreamer bus messages.
    pub(super) event_queue: EventQueue,
    /// Earliest `Instant` at which the next reconnect attempt is allowed.
    /// Set when a backoff delay is computed; cleared on successful connect.
    pub(super) reconnect_deadline: Option<Instant>,
    /// Liveness watchdog: deadline by which the new session must produce a
    /// `StreamStarted` event. Set after a successful (re)connect; cleared
    /// once the stream is confirmed flowing. If it expires, a reconnection
    /// is forced.
    pub(super) liveness_deadline: Option<Instant>,
    /// User-facing decode preference — mapped to internal `DecoderSelection`
    /// when building each session.
    pub(super) decode_preference: DecodePreference,
    /// Cached capability probe (populated on first use to avoid repeated
    /// registry scans on reconnect).
    pub(super) cached_capabilities: Option<crate::decode::DecodeCapabilities>,
    /// Adaptive fallback cache — tracks consecutive hardware decoder
    /// failures to prevent reconnect thrash for `PreferHardware` feeds.
    pub(super) hw_failure_tracker: HwFailureTracker,
    /// Last confirmed decode decision — set once at StreamStarted.
    pub(super) last_decode_status: Option<(DecodeOutcome, String)>,
    /// Whether decoder verification has run for the current session.
    /// Reset to `false` on each `create_session()` / `create_session_stub()`.
    /// Set to `true` after `verify_decoder_selection()` completes.
    pub(super) decoder_verified: bool,
    /// Fallback reason captured during session creation (if the adaptive
    /// fallback cache overrode the decode preference).
    pub(super) session_fallback_reason: Option<String>,
    /// Optional post-decode hook — passed through to the pipeline builder.
    pub(super) post_decode_hook: Option<PostDecodeHook>,
    /// Maximum events buffered in the event queue before drops.
    pub(super) event_queue_capacity: usize,
}

impl MediaSource {
    /// Create a new media source (does not start the pipeline).
    #[must_use]
    pub fn new(
        feed_id: FeedId,
        spec: SourceSpec,
        reconnect_policy: ReconnectPolicy,
        decode_preference: DecodePreference,
    ) -> Self {
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
            decode_preference,
            cached_capabilities: None,
            hw_failure_tracker: HwFailureTracker::new(),
            last_decode_status: None,
            decoder_verified: false,
            session_fallback_reason: None,
            post_decode_hook: None,
            event_queue_capacity: crate::backend::EVENT_QUEUE_CAPACITY,
        }
    }

    /// Attach a health event sink for lifecycle reporting.
    ///
    /// The source emits [`HealthEvent`]s as it processes internal media
    /// events (connection, disconnection, reconnection attempts, etc.).
    /// Must be called before [`start()`](crate::ingress::MediaIngress::start).
    pub fn set_health_sink(&mut self, sink: Arc<dyn HealthSink>) {
        self.health_sink = Some(sink);
    }

    /// Attach a PTZ telemetry provider.
    ///
    /// The provider is queried on every decoded frame to attach PTZ metadata.
    /// Must be called before [`start()`](crate::ingress::MediaIngress::start).
    pub fn set_ptz_provider(&mut self, provider: Arc<dyn PtzProvider>) {
        self.ptz_provider = Some(provider);
    }

    /// Attach a post-decode hook.
    ///
    /// The hook is invoked once per session when the decoded stream's caps
    /// are known, and can inject a pipeline element between the decoder and
    /// the color-space converter.
    /// Must be called before [`start()`](crate::ingress::MediaIngress::start).
    pub fn set_post_decode_hook(&mut self, hook: PostDecodeHook) {
        self.post_decode_hook = Some(hook);
    }

    /// Emit a health event if a health sink is attached.
    pub(super) fn emit_health(&self, event: HealthEvent) {
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
    pub(super) fn enrich_error(&self, error: MediaError) -> MediaError {
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
    pub(super) fn create_session(&mut self) -> Result<(), MediaError> {
        // Lazily probe and cache capabilities so reconnect cycles
        // don't repeatedly scan the GStreamer registry.
        let caps = self
            .cached_capabilities
            .get_or_insert_with(crate::decode::discover_decode_capabilities);

        // RequireHardware fail-fast: reject before constructing the pipeline.
        if self.decode_preference.requires_hardware() && !caps.hardware_decode_available {
            let detail = if caps.backend_available {
                "DecodePreference::RequireHardware — no hardware video decoder found in backend registry"
            } else {
                "DecodePreference::RequireHardware — media backend is not available"
            };
            return Err(MediaError::Unsupported {
                detail: detail.into(),
            });
        }

        // PreferHardware: log when falling back to software so operators
        // can distinguish it from Auto.
        if self.decode_preference.prefers_hardware() && !caps.hardware_decode_available {
            tracing::info!(
                feed_id = %self.feed_id,
                "DecodePreference::PreferHardware — no hardware decoder detected, \
                 falling back to software decode",
            );
        }

        // Consult adaptive fallback cache — may temporarily override
        // selection for PreferHardware after repeated hw failures.
        let (selection, fallback_reason) = match self
            .hw_failure_tracker
            .adjust_selection(self.decode_preference)
        {
            Some((adjusted, reason)) => {
                tracing::info!(
                    feed_id = %self.feed_id,
                    reason = %reason,
                    "adaptive fallback: overriding decoder selection",
                );
                (adjusted, Some(reason))
            }
            None => (self.decode_preference.to_selection(), None),
        };

        // Persist fallback context so verify_decoder_selection() can
        // include it in the DecodeDecision health event.
        self.session_fallback_reason = fallback_reason.clone();
        // Mark decoder as not yet verified for this new session.
        self.decoder_verified = false;

        tracing::info!(
            feed_id = %self.feed_id,
            preference = ?self.decode_preference,
            selection = ?selection,
            hw_available = caps.hardware_decode_available,
            fallback_active = fallback_reason.is_some(),
            "creating session with decoder selection",
        );

        let config = SessionConfig {
            feed_id: self.feed_id,
            spec: self.spec.clone(),
            decoder: selection,
            output_format: OutputFormat::default(),
            ptz_provider: self.ptz_provider.clone(),
            post_decode_hook: self.post_decode_hook.clone(),
            event_queue_capacity: self.event_queue_capacity,
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
    pub(super) fn create_session_stub(&mut self) {
        let config = SessionConfig {
            feed_id: self.feed_id,
            spec: self.spec.clone(),
            decoder: self.decode_preference.to_selection(),
            output_format: OutputFormat::default(),
            ptz_provider: None,
            post_decode_hook: None,
            event_queue_capacity: self.event_queue_capacity,
        };
        self.session = Some(GstSession::start_stub(config));
        self.decoder_verified = false;
        self.session_fallback_reason = None;
    }

    /// Initiate a reconnection cycle.
    ///
    /// Tears down the current session (if any), checks the policy budget,
    /// records the attempt, and returns the backoff delay.
    ///
    /// Returns `Ok(delay)` if another attempt is allowed, or `Err` if the
    /// reconnection budget is exhausted.
    pub(super) fn initiate_reconnect(&mut self) -> Result<Duration, MediaError> {
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
}

// ---------------------------------------------------------------------------
// Test helpers and test module
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

    pub(crate) fn hw_failure_tracker(&self) -> &HwFailureTracker {
        &self.hw_failure_tracker
    }

    pub(crate) fn hw_failure_tracker_mut(&mut self) -> &mut HwFailureTracker {
        &mut self.hw_failure_tracker
    }

    pub(crate) fn decoder_verified(&self) -> bool {
        self.decoder_verified
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
