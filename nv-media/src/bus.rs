//! GStreamer bus message types and mapping.
//!
//! This module defines a library-internal mirror of the GStreamer bus messages
//! that matter for source lifecycle. The actual GStreamer bus polling lives in
//! [`backend`](crate::backend) behind `#[cfg(feature = "gst-backend")]`; this
//! module provides the intermediate types and the mapping to [`MediaEvent`].
//!
//! Keeping the mapping as pure functions over library-owned enums makes it
//! easy to test without linking GStreamer.

use crate::event::MediaEvent;
use nv_core::error::MediaError;

/// Library-internal representation of a GStreamer bus message.
///
/// Constructed by the backend from actual `gst::Message` values.
/// Processed by the source layer's event loop via [`into_media_event`](Self::into_media_event).
#[derive(Debug)]
pub(crate) enum BusMessage {
    /// Pipeline reached end of stream.
    Eos,

    /// Fatal pipeline error — the pipeline cannot continue.
    Error {
        message: String,
        debug: Option<String>,
    },

    /// Non-fatal pipeline warning.
    Warning {
        message: String,
        debug: Option<String>,
    },

    /// A pipeline element changed state.
    StateChanged {
        old: ElementState,
        new: ElementState,
    },

    /// A new stream started (stream-start message from the source element).
    StreamStart,

    /// Pipeline latency changed — an element requested latency recalculation.
    Latency,

    /// Buffering progress (e.g., for network sources with jitter buffers).
    Buffering { percent: u32 },
}

/// GStreamer element state (library-internal mirror).
///
/// Mirrors `gst::State` without depending on `gstreamer-rs` at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementState {
    Null,
    Ready,
    Paused,
    Playing,
}

impl BusMessage {
    /// Map a bus message to a [`MediaEvent`] for the source lifecycle FSM.
    ///
    /// Returns `None` for messages that do not affect source lifecycle
    /// (e.g., latency notifications, buffering progress).
    pub fn into_media_event(self) -> Option<MediaEvent> {
        match self {
            Self::Eos => Some(MediaEvent::Eos),

            Self::Error { message, debug } => {
                let error = classify_bus_error(&message);
                Some(MediaEvent::Error { error, debug })
            }

            Self::Warning { message, debug } => Some(MediaEvent::Warning { message, debug }),

            Self::StreamStart => Some(MediaEvent::StreamStarted),

            // State changes, latency, and buffering are informational —
            // they don't drive the source FSM directly.
            Self::StateChanged { .. } | Self::Latency | Self::Buffering { .. } => None,
        }
    }
}

/// Classify a GStreamer error message into the appropriate [`MediaError`] variant.
fn classify_bus_error(message: &str) -> MediaError {
    let lower = message.to_lowercase();
    // Check timeout before connection — "connection timed out" should be Timeout
    if lower.contains("timeout") || lower.contains("timed out") {
        MediaError::Timeout
    } else if lower.contains("could not open resource")
        || lower.contains("connection refused")
        || lower.contains("not found")
        || lower.contains("connect")
        || lower.contains("resolve")
        || lower.contains("unauthorized")
    {
        MediaError::ConnectionFailed {
            url: String::new(),
            detail: message.to_string(),
        }
    } else if lower.contains("not supported")
        || lower.contains("unsupported")
        || lower.contains("no decoder")
        || lower.contains("missing plugin")
    {
        MediaError::Unsupported {
            detail: message.to_string(),
        }
    } else {
        MediaError::DecodeFailed {
            detail: message.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_maps_to_eos() {
        let evt = BusMessage::Eos.into_media_event().unwrap();
        assert!(matches!(evt, MediaEvent::Eos));
    }

    #[test]
    fn error_maps_to_error() {
        let evt = BusMessage::Error {
            message: "decode failure".into(),
            debug: Some("element foobar".into()),
        }
        .into_media_event()
        .unwrap();

        match evt {
            MediaEvent::Error { error, debug } => {
                assert!(error.to_string().contains("decode failure"));
                assert_eq!(debug.as_deref(), Some("element foobar"));
            }
            _ => panic!("expected Error event"),
        }
    }

    #[test]
    fn connection_error_classified_correctly() {
        let evt = BusMessage::Error {
            message: "Could not open resource for reading".into(),
            debug: None,
        }
        .into_media_event()
        .unwrap();

        match evt {
            MediaEvent::Error { error, .. } => {
                assert!(matches!(error, MediaError::ConnectionFailed { .. }));
            }
            _ => panic!("expected Error event"),
        }
    }

    #[test]
    fn timeout_error_classified_correctly() {
        let evt = BusMessage::Error {
            message: "connection timed out".into(),
            debug: None,
        }
        .into_media_event()
        .unwrap();

        match evt {
            MediaEvent::Error { error, .. } => {
                assert!(matches!(error, MediaError::Timeout));
            }
            _ => panic!("expected Error event"),
        }
    }

    #[test]
    fn unsupported_error_classified_correctly() {
        let evt = BusMessage::Error {
            message: "no decoder available for type".into(),
            debug: None,
        }
        .into_media_event()
        .unwrap();

        match evt {
            MediaEvent::Error { error, .. } => {
                assert!(matches!(error, MediaError::Unsupported { .. }));
            }
            _ => panic!("expected Error event"),
        }
    }

    #[test]
    fn warning_maps_to_warning() {
        let evt = BusMessage::Warning {
            message: "clock drift".into(),
            debug: None,
        }
        .into_media_event()
        .unwrap();
        assert!(matches!(evt, MediaEvent::Warning { .. }));
    }

    #[test]
    fn stream_start_maps_to_started() {
        let evt = BusMessage::StreamStart.into_media_event().unwrap();
        assert!(matches!(evt, MediaEvent::StreamStarted));
    }

    #[test]
    fn state_changed_maps_to_none() {
        let evt = BusMessage::StateChanged {
            old: ElementState::Ready,
            new: ElementState::Playing,
        }
        .into_media_event();
        assert!(evt.is_none());
    }

    #[test]
    fn latency_maps_to_none() {
        assert!(BusMessage::Latency.into_media_event().is_none());
    }

    #[test]
    fn buffering_maps_to_none() {
        assert!(
            BusMessage::Buffering { percent: 50 }
                .into_media_event()
                .is_none()
        );
    }
}
