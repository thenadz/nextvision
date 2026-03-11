//! GStreamer session adapter — the safe boundary around GStreamer internals.
//!
//! [`GstSession`] owns a single GStreamer pipeline for one feed. It is the
//! **sole location** where GStreamer API calls are made (together with
//! helpers in [`pipeline`](crate::pipeline) and [`bridge`](crate::bridge)).
//!
//! # Adapter boundary
//!
//! No GStreamer types cross this module's `pub(crate)` interface. Callers
//! interact with [`SessionConfig`], [`BusMessage`], and [`MediaEvent`] —
//! all library-defined types.
//!
//! # Feature gating
//!
//! When the `gst-backend` cargo feature is **enabled**, `GstSession::start()`
//! builds a real GStreamer pipeline, wires the appsink callback, and starts
//! the bus watch. When the feature is **disabled**, `start()` returns
//! `MediaError::Unsupported`. This allows the rest of the crate (types,
//! state machines, event mapping) to compile and be tested without
//! GStreamer development libraries.
//!
//! # Lifecycle
//!
//! ```text
//! start() ──► Running ──► pause() ──► Paused ──► resume() ──► Running
//!                │                                   │
//!                └───────── stop() ──────────────────┘──► Stopped
//! ```
//!
//! `Drop` calls `stop()` for best-effort cleanup.

use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use nv_core::config::SourceSpec;
use nv_core::error::MediaError;
use nv_core::id::FeedId;

use crate::bus::BusMessage;
use crate::clock::PtsTracker;
use crate::decode::DecoderSelection;
use crate::event::MediaEvent;
use crate::ingress::{FrameSink, PtzProvider};
use crate::pipeline::{OutputFormat, PipelineBuilder};

/// Thread-safe queue for media events produced asynchronously (e.g., from
/// the appsink callback thread). Drained by [`MediaSource::poll_bus()`].
pub(crate) type EventQueue = Arc<Mutex<VecDeque<MediaEvent>>>;

/// Maximum number of events buffered in the [`EventQueue`] before new events
/// are dropped. Prevents unbounded memory growth when the bus poller is slow.
pub(crate) const EVENT_QUEUE_CAPACITY: usize = 64;

/// Configuration for a GStreamer session.
pub(crate) struct SessionConfig {
    pub feed_id: FeedId,
    pub spec: SourceSpec,
    pub decoder: DecoderSelection,
    pub output_format: OutputFormat,
    /// Optional PTZ telemetry provider — queried per frame.
    pub ptz_provider: Option<Arc<dyn PtzProvider>>,
}

impl std::fmt::Debug for SessionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionConfig")
            .field("feed_id", &self.feed_id)
            .field("spec", &self.spec)
            .field("decoder", &self.decoder)
            .field("output_format", &self.output_format)
            .field("ptz_provider", &self.ptz_provider.as_ref().map(|_| ".."))
            .finish()
    }
}

/// State of a GStreamer session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionState {
    /// Pipeline is in the Playing state, frames are flowing.
    Running,
    /// Pipeline is in the Paused state, connection held open.
    Paused,
    /// Pipeline has been torn down.
    Stopped,
}

/// A running GStreamer session for a single feed.
///
/// Owns the GStreamer pipeline, appsink, and bus watch. Frames are delivered
/// to the [`FrameSink`] via the appsink callback. Bus messages are available
/// through [`poll_bus()`](Self::poll_bus).
///
/// All GStreamer types are encapsulated — nothing leaks beyond this struct.
///
/// # Thread safety
///
/// `GstSession` is `Send` but not `Sync`. It is used from the source's
/// management thread. The appsink callback runs on GStreamer's streaming
/// thread and delivers frames through the `Arc<dyn FrameSink>`.
pub(crate) struct GstSession {
    #[allow(dead_code)] // used only via Debug impl
    feed_id: FeedId,
    #[allow(dead_code)]
    config: SessionConfig,
    state: SessionState,
    /// Monotonic frame sequence counter (shared with appsink callback).
    #[allow(dead_code)] // held to keep Arc alive for appsink callback
    frame_seq: Arc<AtomicU64>,

    // GStreamer handles — only present when the feature is enabled.
    #[cfg(feature = "gst-backend")]
    pipeline: gstreamer::Pipeline,
    #[cfg(feature = "gst-backend")]
    bus: gstreamer::Bus,
    #[cfg(feature = "gst-backend")]
    _appsink: gstreamer_app::AppSink,
}

// SAFETY: GstSession is used from the source management thread.
// The GStreamer pipeline and its elements are thread-safe by design.
unsafe impl Send for GstSession {}

impl GstSession {
    /// Build and start a GStreamer pipeline for the given configuration.
    ///
    /// The appsink callback delivers decoded frames to `sink`. Bus messages
    /// are available via [`poll_bus()`](Self::poll_bus).
    ///
    /// # GStreamer pipeline wiring
    ///
    /// 1. `gst::init()` (idempotent)
    /// 2. Build pipeline from `config.spec` using [`PipelineBuilder`]
    /// 3. Wire appsink callback:
    ///    `GstSample` → [`bridge::bridge_gst_sample()`](crate::bridge::bridge_gst_sample) → `sink.on_frame()`
    /// 4. Set pipeline to `Playing`
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if the pipeline cannot be constructed or started.
    #[cfg(feature = "gst-backend")]
    pub fn start(
        config: SessionConfig,
        sink: Arc<dyn FrameSink>,
        event_queue: EventQueue,
    ) -> Result<Self, MediaError> {
        use gstreamer as gst;
        use gstreamer::prelude::*;

        // Idempotent initialization
        gst::init().map_err(|e| MediaError::Unsupported {
            detail: format!("GStreamer init failed: {e}"),
        })?;

        // Build the pipeline
        let built = PipelineBuilder::new(config.spec.clone())
            .decoder(config.decoder.clone())
            .output_format(config.output_format)
            .build()?;

        let feed_id = config.feed_id;
        let output_format = built.output_format;
        let frame_seq = Arc::new(AtomicU64::new(0));

        // Shared PTS tracker for discontinuity detection on the appsink thread.
        let pts_tracker = Arc::new(Mutex::new(PtsTracker::new()));

        // PTZ provider (if any) — queried on each frame in the appsink callback.
        let ptz = config.ptz_provider.clone();

        // Wire the appsink new-sample callback
        let seq_counter = Arc::clone(&frame_seq);
        let sink_clone = Arc::clone(&sink);
        let sink_wake = Arc::clone(&sink);
        let pts_clone = Arc::clone(&pts_tracker);
        let eq_clone = Arc::clone(&event_queue);
        built.appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    let ptz_telemetry = ptz.as_ref().and_then(|p| p.latest());
                    match crate::bridge::bridge_gst_sample(
                        feed_id,
                        &seq_counter,
                        output_format,
                        &sample,
                        ptz_telemetry,
                    ) {
                        Ok(frame) => {
                            // Observe PTS for discontinuity detection
                            let pts_ns = frame.ts().as_nanos();
                            if let Ok(mut tracker) = pts_clone.lock() {
                                let result = tracker.observe(pts_ns);
                                if let crate::clock::PtsResult::Discontinuity {
                                    gap_ns,
                                    prev_ns,
                                    current_ns,
                                } = result
                                {
                                    if let Ok(mut q) = eq_clone.lock() {
                                        if q.len() < EVENT_QUEUE_CAPACITY {
                                            q.push_back(MediaEvent::Discontinuity {
                                                gap_ns,
                                                prev_pts_ns: prev_ns,
                                                current_pts_ns: current_ns,
                                            });
                                        } else {
                                            tracing::warn!(
                                                feed_id = %feed_id,
                                                "event queue full, dropping discontinuity event"
                                            );
                                        }
                                    } else {
                                        tracing::warn!(
                                            feed_id = %feed_id,
                                            "event queue lock poisoned, dropping discontinuity event"
                                        );
                                    }
                                }
                            } else {
                                tracing::warn!(
                                    feed_id = %feed_id,
                                    "PTS tracker lock poisoned, skipping discontinuity detection"
                                );
                            }
                            sink_clone.on_frame(frame);
                            Ok(gst::FlowSuccess::Ok)
                        }
                        Err(e) => {
                            tracing::warn!(
                                feed_id = %feed_id,
                                error = %e,
                                "bridge_gst_sample failed, dropping frame"
                            );
                            // Don't propagate as EOS — just drop this frame
                            Ok(gst::FlowSuccess::Ok)
                        }
                    }
                })
                .eos(move |_appsink| {
                    // Wake the consumer so the worker ticks the source.
                    // The source FSM (poll_bus/handle_event) decides whether
                    // to reconnect or stop. Do NOT close the queue here.
                    sink.wake();
                })
                .build(),
        );

        // Install a bus sync handler so actionable messages (Error, EOS)
        // immediately wake the consumer thread. The sync handler runs on
        // the thread that posted the message \u2014 wake_consumer() is safe
        // from any thread (condvar notify + atomic flag).
        //
        // Messages are kept on the bus (BusSyncReply::Pass) for
        // poll_bus() to process through the normal source FSM path.
        let sink_bus = Arc::downgrade(&sink_wake);
        built.bus.set_sync_handler(move |_bus, msg| {
            use gstreamer::MessageView;
            match msg.view() {
                MessageView::Error(_) | MessageView::Eos(_) => {
                    if let Some(s) = sink_bus.upgrade() {
                        s.wake();
                    }
                }
                _ => {}
            }
            gst::BusSyncReply::Pass
        });

        // Start the pipeline
        built
            .pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to set pipeline to Playing: {e}"),
            })?;

        tracing::info!(
            feed_id = %feed_id,
            source = ?config.spec,
            "GStreamer pipeline started"
        );

        Ok(Self {
            feed_id,
            config,
            state: SessionState::Running,
            frame_seq,
            pipeline: built.pipeline,
            bus: built.bus,
            _appsink: built.appsink,
        })
    }

    /// Start stub when GStreamer is not linked.
    ///
    /// # Errors
    ///
    /// Always returns `MediaError::Unsupported`.
    #[cfg(not(feature = "gst-backend"))]
    pub fn start(
        config: SessionConfig,
        _sink: Arc<dyn FrameSink>,
        _event_queue: EventQueue,
    ) -> Result<Self, MediaError> {
        let _ = config;
        Err(MediaError::Unsupported {
            detail: "GStreamer backend not linked (enable the `gst-backend` feature)".into(),
        })
    }

    /// Create a running stub session for unit tests.
    ///
    /// No GStreamer pipeline exists — the session is just a state handle.
    /// Used by `source.rs` tests to exercise the lifecycle FSM without
    /// requiring GStreamer development libraries.
    #[cfg(test)]
    pub fn start_stub(config: SessionConfig) -> Self {
        let feed_id = config.feed_id;
        Self {
            feed_id,
            config,
            state: SessionState::Running,
            frame_seq: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "gst-backend")]
            pipeline: {
                gstreamer::init().unwrap();
                gstreamer::Pipeline::new()
            },
            #[cfg(feature = "gst-backend")]
            bus: {
                use gstreamer::prelude::*;
                gstreamer::Pipeline::new().bus().unwrap()
            },
            #[cfg(feature = "gst-backend")]
            _appsink: { gstreamer_app::AppSink::builder().build() },
        }
    }

    /// Poll the bus for a pending message (non-blocking).
    ///
    /// Returns `None` if no message is available. Maps GStreamer bus
    /// messages to [`BusMessage`].
    #[cfg(feature = "gst-backend")]
    pub fn poll_bus(&self) -> Option<BusMessage> {
        use gstreamer::MessageView;
        use gstreamer::prelude::*;

        let msg = self.bus.pop()?;
        let pipeline_obj: &gstreamer::Object = self.pipeline.upcast_ref();
        match msg.view() {
            MessageView::Eos(_) => Some(BusMessage::Eos),
            MessageView::Error(e) => {
                let debug = e.debug().map(|d| d.to_string());
                Some(BusMessage::Error {
                    message: e.error().to_string(),
                    debug,
                })
            }
            MessageView::Warning(w) => {
                let debug = w.debug().map(|d| d.to_string());
                Some(BusMessage::Warning {
                    message: w.error().to_string(),
                    debug,
                })
            }
            MessageView::StateChanged(sc) => {
                // Only report state changes from the pipeline element itself
                if msg.src() == Some(pipeline_obj) {
                    Some(BusMessage::StateChanged {
                        old: map_gst_state(sc.old()),
                        new: map_gst_state(sc.current()),
                    })
                } else {
                    None
                }
            }
            MessageView::StreamStart(_) => Some(BusMessage::StreamStart),
            MessageView::Latency(_) => Some(BusMessage::Latency),
            MessageView::Buffering(b) => Some(BusMessage::Buffering {
                percent: b.percent() as u32,
            }),
            _ => None,
        }
    }

    /// Stub poll when GStreamer is not linked.
    #[cfg(not(feature = "gst-backend"))]
    pub fn poll_bus(&self) -> Option<BusMessage> {
        None
    }

    /// Set the pipeline to the Paused state.
    ///
    /// # Errors
    ///
    /// Returns `MediaError::Unsupported` if not in `Running` state.
    pub fn pause(&mut self) -> Result<(), MediaError> {
        if self.state != SessionState::Running {
            return Err(MediaError::Unsupported {
                detail: "can only pause a running session".into(),
            });
        }
        #[cfg(feature = "gst-backend")]
        {
            use gstreamer::prelude::*;
            self.pipeline
                .set_state(gstreamer::State::Paused)
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to pause pipeline: {e}"),
                })?;
        }
        self.state = SessionState::Paused;
        Ok(())
    }

    /// Set the pipeline back to the Playing state.
    ///
    /// # Errors
    ///
    /// Returns `MediaError::Unsupported` if not in `Paused` state.
    pub fn resume(&mut self) -> Result<(), MediaError> {
        if self.state != SessionState::Paused {
            return Err(MediaError::Unsupported {
                detail: "can only resume a paused session".into(),
            });
        }
        #[cfg(feature = "gst-backend")]
        {
            use gstreamer::prelude::*;
            self.pipeline
                .set_state(gstreamer::State::Playing)
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to resume pipeline: {e}"),
                })?;
        }
        self.state = SessionState::Running;
        Ok(())
    }

    /// Tear down the pipeline and release all GStreamer resources.
    ///
    /// Idempotent — calling `stop()` on a stopped session is a no-op.
    pub fn stop(&mut self) -> Result<(), MediaError> {
        if self.state == SessionState::Stopped {
            return Ok(());
        }
        #[cfg(feature = "gst-backend")]
        {
            use gstreamer::prelude::*;
            let _ = self.pipeline.set_state(gstreamer::State::Null);
        }
        self.state = SessionState::Stopped;
        Ok(())
    }

    /// Seek the pipeline back to the beginning.
    ///
    /// Used for looping file sources on EOS. The pipeline stays in Playing
    /// state and frames resume from the start of the file.
    ///
    /// # Errors
    ///
    /// Returns `MediaError::Unsupported` if the seek fails or the session is
    /// not in the Running state.
    pub fn seek_start(&mut self) -> Result<(), MediaError> {
        if self.state != SessionState::Running {
            return Err(MediaError::Unsupported {
                detail: "can only seek a running session".into(),
            });
        }
        #[cfg(feature = "gst-backend")]
        {
            use gstreamer::prelude::*;
            self.pipeline
                .seek_simple(
                    gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::KEY_UNIT,
                    gstreamer::ClockTime::ZERO,
                )
                .map_err(|_| MediaError::Unsupported {
                    detail: "pipeline seek to start failed".into(),
                })?;
        }
        Ok(())
    }

}

#[cfg(test)]
impl GstSession {
    pub fn state(&self) -> SessionState {
        self.state
    }
}

/// Map GStreamer element state to our library-internal mirror type.
#[cfg(feature = "gst-backend")]
fn map_gst_state(state: gstreamer::State) -> crate::bus::ElementState {
    match state {
        gstreamer::State::Null | gstreamer::State::Ready => crate::bus::ElementState::Ready,
        gstreamer::State::Paused => crate::bus::ElementState::Paused,
        gstreamer::State::Playing => crate::bus::ElementState::Playing,
        _ => crate::bus::ElementState::Null,
    }
}

impl Drop for GstSession {
    fn drop(&mut self) {
        if self.state != SessionState::Stopped {
            let _ = self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::config::RtspTransport;

    fn test_config() -> SessionConfig {
        SessionConfig {
            feed_id: FeedId::new(1),
            spec: SourceSpec::Rtsp {
                url: "rtsp://test/stream".into(),
                transport: RtspTransport::Tcp,
            },
            decoder: DecoderSelection::Auto,
            output_format: OutputFormat::default(),
            ptz_provider: None,
        }
    }

    #[test]
    fn stub_session_starts_running() {
        let s = GstSession::start_stub(test_config());
        assert_eq!(s.state(), SessionState::Running);
    }

    #[test]
    fn pause_resume_cycle() {
        let mut s = GstSession::start_stub(test_config());
        s.pause().unwrap();
        assert_eq!(s.state(), SessionState::Paused);
        s.resume().unwrap();
        assert_eq!(s.state(), SessionState::Running);
    }

    #[test]
    fn stop_is_terminal() {
        let mut s = GstSession::start_stub(test_config());
        s.stop().unwrap();
        assert_eq!(s.state(), SessionState::Stopped);
        // Idempotent
        s.stop().unwrap();
    }

    #[test]
    fn cannot_pause_when_not_running() {
        let mut s = GstSession::start_stub(test_config());
        s.stop().unwrap();
        assert!(s.pause().is_err());
    }

    #[test]
    fn cannot_resume_when_not_paused() {
        let mut s = GstSession::start_stub(test_config());
        assert!(s.resume().is_err());
    }

    #[test]
    fn drop_stops_session() {
        let s = GstSession::start_stub(test_config());
        assert_eq!(s.state(), SessionState::Running);
        drop(s); // should not panic
    }

    /// Compile-time assertion: `GstSession` is `Send`.
    const _: () = {
        const fn assert_send<T: Send>() {}
        assert_send::<GstSession>();
    };
}
