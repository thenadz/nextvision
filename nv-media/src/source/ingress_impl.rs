use std::sync::Arc;
use std::time::{Duration, Instant};

use nv_core::config::SourceSpec;
use nv_core::error::MediaError;
use nv_core::health::DecodeOutcome;
use nv_core::id::FeedId;

use crate::ingress::{FrameSink, MediaIngress, SourceStatus};

use super::{MediaSource, SourceState};

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
                // SourceConnected and DecodeDecision are deferred until
                // handle_event(StreamStarted) confirms the stream is
                // actually flowing and decoder verification succeeds.
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
                    self.disconnect_and_reconnect(MediaError::Timeout);
                    // Re-poll to pick up the reconnection state.
                    let _delay = self.poll_bus();
                }
            }
        }

        // Map the resulting state and compute the next-tick hint.
        match self.state {
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
        }
    }

    fn decode_status(&self) -> Option<(DecodeOutcome, String)> {
        self.last_decode_status.clone()
    }
}
