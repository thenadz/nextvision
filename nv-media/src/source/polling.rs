use std::time::{Duration, Instant};

use nv_core::health::HealthEvent;

use crate::event::MediaEvent;

use super::{MediaSource, ReconnectOutcome, SourceState};

impl MediaSource {
    /// Poll the GStreamer bus for pending messages and process them.
    ///
    /// This is the operational connection between the bus message types
    /// and the source lifecycle FSM. Should be called periodically from the
    /// source's management loop (e.g., on a timer or after each frame batch).
    ///
    /// Returns `Some(delay)` if a reconnection should be scheduled after
    /// waiting, or `None` if nothing actionable occurred.
    pub fn poll_bus(&mut self) -> Option<Duration> {
        if let Some(delay) = self.drain_queued_events() {
            return Some(delay);
        }

        let reconnect_delay = self.drain_bus_messages();

        if reconnect_delay.is_none() {
            return self.check_reconnect();
        }

        reconnect_delay
    }

    /// Drain discontinuity events produced by the appsink callback thread.
    ///
    /// Returns `Some(delay)` if an event triggered a reconnection.
    fn drain_queued_events(&mut self) -> Option<Duration> {
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
                "draining queued discontinuity events"
            );
        }

        for event in queued_events {
            if let Some(delay) = self.handle_event(event) {
                self.reconnect_deadline = Some(Instant::now() + delay);
                return Some(delay);
            }
        }

        None
    }

    /// Drain GStreamer bus messages and process any media events.
    ///
    /// Returns `Some(delay)` if a message triggered a reconnection.
    fn drain_bus_messages(&mut self) -> Option<Duration> {
        let mut messages = Vec::new();
        if let Some(ref session) = self.session {
            while let Some(bus_msg) = session.poll_bus() {
                messages.push(bus_msg);
            }
        } else {
            tracing::debug!(
                feed_id = %self.feed_id,
                state = ?self.state,
                "no session — skipping bus drain"
            );
        }

        if !messages.is_empty() {
            tracing::trace!(
                feed_id = %self.feed_id,
                count = messages.len(),
                "draining GStreamer bus messages"
            );
        }

        for bus_msg in messages {
            tracing::trace!(
                feed_id = %self.feed_id,
                bus_msg = ?bus_msg,
                "processing bus message"
            );
            if let Some(event) = bus_msg.into_media_event() {
                tracing::debug!(
                    feed_id = %self.feed_id,
                    event = ?event,
                    "mapped to media event"
                );
                if let Some(delay) = self.handle_event(event) {
                    self.reconnect_deadline = Some(Instant::now() + delay);
                    return Some(delay);
                }
            }
        }

        None
    }

    /// If in Reconnecting state, attempt reconnection when the backoff
    /// deadline has elapsed.
    ///
    /// Returns `Some(delay)` if a retry was scheduled, `None` otherwise.
    fn check_reconnect(&mut self) -> Option<Duration> {
        if self.state != SourceState::Reconnecting {
            return None;
        }

        let deadline_elapsed = self.reconnect_deadline.is_none_or(|d| Instant::now() >= d);
        tracing::debug!(
            feed_id = %self.feed_id,
            deadline_elapsed,
            reconnect_deadline = ?self.reconnect_deadline,
            "in Reconnecting state, checking deadline"
        );
        if !deadline_elapsed {
            return None;
        }

        match self.try_reconnect() {
            ReconnectOutcome::Connected => {
                tracing::info!(
                    feed_id = %self.feed_id,
                    "reconnection succeeded"
                );
                self.reconnect_deadline = None;
                None
            }
            ReconnectOutcome::Retry { delay } => {
                tracing::debug!(
                    feed_id = %self.feed_id,
                    delay_ms = delay.as_millis() as u64,
                    "reconnection failed, scheduling retry"
                );
                self.reconnect_deadline = Some(Instant::now() + delay);
                Some(delay)
            }
            ReconnectOutcome::Exhausted => {
                tracing::warn!(
                    feed_id = %self.feed_id,
                    "reconnection budget exhausted, stopped"
                );
                None
            }
        }
    }

    /// Attempt a single reconnection: tear down the old session and create a new one.
    ///
    /// This method is meant to be called by the runtime after the backoff delay
    /// returned by [`handle_event()`] or [`poll_bus()`] has elapsed.
    pub(crate) fn try_reconnect(&mut self) -> ReconnectOutcome {
        if self.state == SourceState::Stopped {
            return ReconnectOutcome::Exhausted;
        }
        if self.state != SourceState::Reconnecting {
            return ReconnectOutcome::Connected;
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
                ReconnectOutcome::Connected
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
                        ReconnectOutcome::Retry { delay }
                    }
                    Err(_) => {
                        self.state = SourceState::Stopped;
                        if let Some(ref sink) = self.sink {
                            sink.on_eos();
                        }
                        // FeedStopped is emitted by the worker thread
                        // (the canonical owner of feed lifecycle events).
                        ReconnectOutcome::Exhausted
                    }
                }
            }
        }
    }
}
