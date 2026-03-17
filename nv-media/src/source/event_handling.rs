use std::time::Duration;

use nv_core::error::MediaError;
use nv_core::health::{DecodeOutcome, HealthEvent};

use crate::decode::DecodePreferenceExt;
use crate::event::MediaEvent;

use super::{MediaSource, SourceState};

impl MediaSource {
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
                // Stream is confirmed flowing — clear the liveness watchdog.
                // Do NOT reset reconnect attempts yet: decoder verification
                // may still reject this stream (e.g., RequireHardware with a
                // software decoder). Resetting here would bypass the reconnect
                // budget.
                self.liveness_deadline = None;

                // Guard against duplicate verification within the same
                // session. This can happen when start() or try_reconnect()
                // transitions to Running and the GStreamer bus subsequently
                // fires StreamStarted — the decoder has already been
                // verified for this session. Attempts were already reset
                // when verification first succeeded.
                if self.decoder_verified {
                    return None;
                }

                self.state = SourceState::Running;
                self.reconnect_deadline = None;
                tracing::info!(feed_id = %self.feed_id, "stream started");

                // Verify decoder selection and emit DecodeDecision health
                // event. If post-selection enforcement fails (e.g.,
                // RequireHardware with a software decoder), this returns
                // Some(delay) to trigger reconnection — attempts are NOT
                // reset in that case.
                if let Some(delay) = self.verify_decoder_selection() {
                    return Some(delay);
                }

                // Verification passed — this is a genuinely accepted stream
                // start. Reset reconnect attempts now.
                self.reconnect.reset_attempts();

                self.emit_health(HealthEvent::SourceConnected {
                    feed_id: self.feed_id,
                });
                None
            }
            MediaEvent::Eos => {
                tracing::info!(feed_id = %self.feed_id, "end of stream");
                match &self.spec {
                    nv_core::config::SourceSpec::File { loop_: true, .. } => {
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
                    nv_core::config::SourceSpec::File { loop_: false, .. } => {
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
            MediaEvent::Error {
                error,
                debug: debug_detail,
            } => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    error = %error,
                    debug_detail = ?debug_detail,
                    "pipeline error"
                );
                // Track hardware failure for adaptive fallback when the
                // stream never confirmed (liveness watchdog still armed)
                // and the preference is PreferHardware.
                if self.liveness_deadline.is_some() && self.decode_preference.prefers_hardware() {
                    self.hw_failure_tracker.record_failure();
                }
                let enriched = self.enrich_error(error);
                self.disconnect_and_reconnect(enriched)
            }
            MediaEvent::Warning {
                message,
                debug: debug_detail,
            } => {
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

    /// Verify the decoder selected by the backend and emit diagnostics.
    ///
    /// Called once per session when `StreamStarted` confirms the stream is
    /// flowing. Returns `Some(delay)` if post-selection enforcement fails
    /// (e.g., `RequireHardware` with a software or unknown decoder), which
    /// signals the caller to schedule a reconnection.
    fn verify_decoder_selection(&mut self) -> Option<Duration> {
        let selected = self.session.as_ref().and_then(|s| s.selected_decoder());
        let outcome = match &selected {
            Some(info) if info.is_hardware => DecodeOutcome::Hardware,
            Some(_) => DecodeOutcome::Software,
            None => DecodeOutcome::Unknown,
        };
        let element_name = selected
            .as_ref()
            .map(|s| s.element_name.as_str())
            .unwrap_or("unknown");

        // Post-selection enforcement for RequireHardware.
        // Reject both Software and Unknown — fail closed when we cannot
        // confirm hardware acceleration.
        if self.decode_preference.requires_hardware()
            && matches!(outcome, DecodeOutcome::Software | DecodeOutcome::Unknown)
        {
            let detail = format!(
                "RequireHardware: effective decoder '{}' is {} — aborting",
                element_name, outcome,
            );
            tracing::error!(feed_id = %self.feed_id, detail = %detail);
            self.hw_failure_tracker.record_failure();
            return self.disconnect_and_reconnect(MediaError::Unsupported { detail });
        }

        // Track success for adaptive fallback cache.
        self.hw_failure_tracker.record_success();

        let fallback_active = self.session_fallback_reason.is_some();

        tracing::info!(
            feed_id = %self.feed_id,
            preference = ?self.decode_preference,
            outcome = ?outcome,
            decoder = %element_name,
            fallback_active,
            "decode decision verified",
        );

        let detail_string = element_name.to_string();
        self.last_decode_status = Some((outcome, detail_string.clone()));
        self.decoder_verified = true;

        self.emit_health(HealthEvent::DecodeDecision {
            feed_id: self.feed_id,
            outcome,
            preference: self.decode_preference,
            fallback_active,
            fallback_reason: self.session_fallback_reason.clone(),
            detail: detail_string,
        });

        None
    }

    /// Emit `SourceDisconnected`, notify the sink, and attempt reconnection.
    ///
    /// Shared path for pipeline errors and liveness watchdog expiry.
    pub(super) fn disconnect_and_reconnect(&mut self, error: MediaError) -> Option<Duration> {
        self.emit_health(HealthEvent::SourceDisconnected {
            feed_id: self.feed_id,
            reason: error.clone(),
        });
        if let Some(ref sink) = self.sink {
            sink.on_error(error);
        }
        self.try_reconnect_or_stop()
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
}
