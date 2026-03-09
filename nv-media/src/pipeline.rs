//! GStreamer pipeline construction and appsink wiring.
//!
//! Translates a [`SourceSpec`](nv_core::SourceSpec) into a GStreamer pipeline
//! with an `appsink` for frame extraction.
//!
//! # Pipeline topology
//!
//! The builder produces a pipeline of the form:
//!
//! ```text
//! source → [depay] → [parse] → decode → videoconvert → appsink
//! ```
//!
//! - **source**: chosen from the `SourceSpec` variant (rtspsrc, filesrc, v4l2src, or custom).
//! - **depay/parse**: inserted automatically for RTSP streams based on the negotiated codec.
//! - **decode**: chosen by [`DecoderSelection`](crate::decode::DecoderSelection) — hardware
//!   accelerated when available, software fallback otherwise.
//! - **videoconvert**: converts to the requested [`OutputFormat`].
//! - **appsink**: the extraction point. Configured with caps matching `OutputFormat`.
//!
//! # Feature gating
//!
//! The [`PipelineBuilder::build()`] method is only fully functional when the
//! `gst-backend` cargo feature is enabled. All configuration types compile
//! unconditionally.

use nv_core::config::{RtspTransport, SourceSpec};
use nv_core::error::MediaError;
use nv_frame::PixelFormat;

use crate::decode::DecoderSelection;

/// Target pixel format for the appsink output.
///
/// This controls the `video/x-raw,format=...` caps set on the appsink.
/// The `videoconvert` element handles the conversion from whatever the
/// decoder outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    /// 8-bit RGB (3 bytes per pixel).
    Rgb,
    /// 8-bit BGR (3 bytes per pixel, OpenCV-native ordering).
    Bgr,
    /// 8-bit RGBA (4 bytes per pixel).
    Rgba,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Rgb
    }
}

impl OutputFormat {
    /// GStreamer caps format string (e.g., `"RGB"`, `"BGR"`, `"RGBA"`).
    pub fn gst_format_str(self) -> &'static str {
        match self {
            Self::Rgb => "RGB",
            Self::Bgr => "BGR",
            Self::Rgba => "RGBA",
        }
    }

    /// Map to the library's [`PixelFormat`].
    pub fn to_pixel_format(self) -> PixelFormat {
        match self {
            Self::Rgb => PixelFormat::Rgb8,
            Self::Bgr => PixelFormat::Bgr8,
            Self::Rgba => PixelFormat::Rgba8,
        }
    }
}

/// Internal pipeline builder — constructs GStreamer elements from a [`SourceSpec`].
///
/// This type is `pub(crate)` and does not appear in the public API.
pub(crate) struct PipelineBuilder {
    spec: SourceSpec,
    decoder: DecoderSelection,
    output_format: OutputFormat,
    /// Latency hint in milliseconds for live sources (jitter buffer).
    latency_ms: u32,
}

/// Default latency hint for RTSP jitter buffers.
const DEFAULT_LATENCY_MS: u32 = 200;

impl PipelineBuilder {
    /// Create a builder for the given source specification.
    pub fn new(spec: SourceSpec) -> Self {
        Self {
            spec,
            decoder: DecoderSelection::default(),
            output_format: OutputFormat::default(),
            latency_ms: DEFAULT_LATENCY_MS,
        }
    }

    /// Override decoder selection.
    pub fn decoder(mut self, decoder: DecoderSelection) -> Self {
        self.decoder = decoder;
        self
    }

    /// Set the target output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set the latency hint for live source jitter buffers (milliseconds).
    pub fn latency_ms(mut self, ms: u32) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Determine the GStreamer source element name for the configured spec.
    pub fn source_element_name(&self) -> &'static str {
        match &self.spec {
            SourceSpec::Rtsp { .. } => "rtspsrc",
            SourceSpec::File { .. } => "filesrc",
            SourceSpec::V4l2 { .. } => "v4l2src",
            SourceSpec::Custom { .. } => "custom",
        }
    }

    /// RTSP transport property value, if applicable.
    fn rtsp_protocols(&self) -> Option<&'static str> {
        match &self.spec {
            SourceSpec::Rtsp { transport, .. } => Some(match transport {
                RtspTransport::Tcp => "tcp",
                RtspTransport::UdpUnicast => "udp",
            }),
            _ => None,
        }
    }

    /// Construct the GStreamer pipeline, appsink, and bus handle.
    ///
    /// # Pipeline construction steps
    ///
    /// 1. Create source element from `SourceSpec` and set properties.
    /// 2. Create `decodebin` + `videoconvert` + `appsink`.
    /// 3. Set appsink caps to `video/x-raw,format=<OutputFormat>`.
    /// 4. Wire dynamic pads (RTSP source → decodebin, decodebin → videoconvert).
    /// 5. Return the assembled pipeline, appsink, and bus.
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if element creation or linking fails.
    #[cfg(feature = "gst-backend")]
    pub fn build(self) -> Result<BuiltPipeline, MediaError> {
        use gstreamer as gst;
        use gstreamer::prelude::*;
        use gstreamer_app as gst_app;

        let pipeline = gst::Pipeline::new();

        // --- Source element ---
        let source = match &self.spec {
            SourceSpec::Rtsp { url, transport } => {
                gst::ElementFactory::make("rtspsrc")
                    .property("location", url.as_str())
                    .property("latency", self.latency_ms)
                    .property(
                        "protocols",
                        match transport {
                            RtspTransport::Tcp => "tcp",
                            RtspTransport::UdpUnicast => "udp",
                        },
                    )
                    .build()
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to create rtspsrc: {e}"),
                    })?
            }
            SourceSpec::File { path, loop_: _ } => {
                gst::ElementFactory::make("filesrc")
                    .property("location", path.to_string_lossy().as_ref())
                    .build()
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to create filesrc: {e}"),
                    })?
            }
            SourceSpec::V4l2 { device } => {
                gst::ElementFactory::make("v4l2src")
                    .property("device", device.as_str())
                    .build()
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to create v4l2src: {e}"),
                    })?
            }
            SourceSpec::Custom { gst_launch_fragment } => {
                gst::parse::bin_from_description(gst_launch_fragment, true)
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to parse custom pipeline fragment: {e}"),
                    })?
                    .upcast()
            }
        };

        // --- Decode element(s) ---
        // When `decoder` is `Auto`, we use `decodebin` which auto-selects the
        // best decoder. For `ForceSoftware`, we set `force-sw-decoders` on
        // decodebin. For `Named`, we try the specific element first and fall
        // back to decodebin on failure.
        let (decode_element, uses_decodebin) = match &self.decoder {
            DecoderSelection::Auto => {
                let db = gst::ElementFactory::make("decodebin")
                    .build()
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to create decodebin: {e}"),
                    })?;
                (db, true)
            }
            DecoderSelection::ForceSoftware => {
                let db = gst::ElementFactory::make("decodebin")
                    .property("force-sw-decoders", true)
                    .build()
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to create decodebin (force-sw): {e}"),
                    })?;
                (db, true)
            }
            DecoderSelection::Named(name) => {
                match gst::ElementFactory::make(name.as_str()).build() {
                    Ok(elem) => (elem, false),
                    Err(_) => {
                        tracing::warn!(
                            decoder = %name,
                            "named decoder not available, falling back to decodebin"
                        );
                        let db = gst::ElementFactory::make("decodebin")
                            .build()
                            .map_err(|e| MediaError::Unsupported {
                                detail: format!("failed to create decodebin (fallback): {e}"),
                            })?;
                        (db, true)
                    }
                }
            }
        };

        // --- Videoconvert + Appsink ---
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to create videoconvert: {e}"),
            })?;

        let caps_str = format!("video/x-raw,format={}", self.output_format.gst_format_str());
        let appsink_caps: gst::Caps = caps_str.parse().map_err(|_| {
            MediaError::Unsupported {
                detail: format!("invalid appsink caps: {caps_str}"),
            }
        })?;

        let appsink = gst_app::AppSink::builder()
            .caps(&appsink_caps)
            .max_buffers(2)
            .drop(true)
            .build();

        // --- Assemble pipeline ---
        let appsink_element: &gst::Element = appsink.upcast_ref();
        pipeline
            .add_many([&source, &decode_element, &videoconvert, appsink_element])
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to add elements to pipeline: {e}"),
            })?;

        let is_rtsp = matches!(&self.spec, SourceSpec::Rtsp { .. });

        if uses_decodebin {
            // decodebin uses dynamic pads — wire them up with signals.

            // Non-RTSP: static link source → decodebin
            if !is_rtsp {
                source.link(&decode_element).map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to link source → decodebin: {e}"),
                })?;
            }

            // Static link: videoconvert → appsink
            videoconvert
                .link(appsink_element)
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to link videoconvert → appsink: {e}"),
                })?;

            // Dynamic pad: decodebin → videoconvert (when video pad appears)
            let vc_weak = videoconvert.downgrade();
            decode_element.connect_pad_added(move |_element, pad| {
                let Some(vc) = vc_weak.upgrade() else { return };
                let caps = pad.current_caps().unwrap_or_else(|| pad.query_caps(None));
                if let Some(structure) = caps.structure(0) {
                    if structure.name().starts_with("video/") {
                        if let Some(sink_pad) = vc.static_pad("sink") {
                            if !sink_pad.is_linked() {
                                let _ = pad.link(&sink_pad);
                            }
                        }
                    }
                }
            });

            // RTSP: dynamic pad rtspsrc → decodebin
            if is_rtsp {
                let db_weak = decode_element.downgrade();
                source.connect_pad_added(move |_element, pad| {
                    let Some(db) = db_weak.upgrade() else { return };
                    if let Some(sink_pad) = db.static_pad("sink") {
                        if !sink_pad.is_linked() {
                            let _ = pad.link(&sink_pad);
                        }
                    }
                });
            }
        } else {
            // Named decoder — static pads. Link: source → decoder → videoconvert → appsink.
            // For RTSP, we still need dynamic pad handling from rtspsrc.
            if is_rtsp {
                let dec_weak = decode_element.downgrade();
                source.connect_pad_added(move |_element, pad| {
                    let Some(dec) = dec_weak.upgrade() else { return };
                    if let Some(sink_pad) = dec.static_pad("sink") {
                        if !sink_pad.is_linked() {
                            let _ = pad.link(&sink_pad);
                        }
                    }
                });
            } else {
                source.link(&decode_element).map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to link source → decoder: {e}"),
                })?;
            }

            decode_element.link(&videoconvert).map_err(|e| MediaError::Unsupported {
                detail: format!("failed to link decoder → videoconvert: {e}"),
            })?;

            videoconvert
                .link(appsink_element)
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to link videoconvert → appsink: {e}"),
                })?;
        }

        let bus = pipeline.bus().ok_or_else(|| MediaError::Unsupported {
            detail: "pipeline has no bus".into(),
        })?;

        Ok(BuiltPipeline {
            pipeline,
            appsink,
            bus,
            output_format: self.output_format,
        })
    }

    /// Stub build when GStreamer is not linked.
    #[cfg(not(feature = "gst-backend"))]
    pub fn build(self) -> Result<BuiltPipeline, MediaError> {
        let _ = self;
        Err(MediaError::Unsupported {
            detail: "GStreamer backend not linked (enable the `gst-backend` feature)".into(),
        })
    }
}

/// Result of a successful pipeline build.
///
/// Contains the assembled GStreamer pipeline, appsink extraction point,
/// bus handle, and the negotiated output format. Owned by [`GstSession`](crate::backend::GstSession).
#[cfg(feature = "gst-backend")]
pub(crate) struct BuiltPipeline {
    pub pipeline: gstreamer::Pipeline,
    pub appsink: gstreamer_app::AppSink,
    pub bus: gstreamer::Bus,
    pub output_format: OutputFormat,
}

/// Stub for non-GStreamer builds (never constructed).
#[cfg(not(feature = "gst-backend"))]
pub(crate) struct BuiltPipeline {
    _private: (),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_element_for_rtsp() {
        let b = PipelineBuilder::new(SourceSpec::rtsp("rtsp://test"));
        assert_eq!(b.source_element_name(), "rtspsrc");
    }

    #[test]
    fn source_element_for_file() {
        let b = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"));
        assert_eq!(b.source_element_name(), "filesrc");
    }

    #[test]
    fn source_element_for_v4l2() {
        let b = PipelineBuilder::new(SourceSpec::V4l2 {
            device: "/dev/video0".into(),
        });
        assert_eq!(b.source_element_name(), "v4l2src");
    }

    #[test]
    fn rtsp_protocols_tcp() {
        let b = PipelineBuilder::new(SourceSpec::Rtsp {
            url: "rtsp://test".into(),
            transport: RtspTransport::Tcp,
        });
        assert_eq!(b.rtsp_protocols(), Some("tcp"));
    }

    #[test]
    fn rtsp_protocols_udp() {
        let b = PipelineBuilder::new(SourceSpec::Rtsp {
            url: "rtsp://test".into(),
            transport: RtspTransport::UdpUnicast,
        });
        assert_eq!(b.rtsp_protocols(), Some("udp"));
    }

    #[test]
    fn non_rtsp_has_no_protocols() {
        let b = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"));
        assert_eq!(b.rtsp_protocols(), None);
    }

    #[test]
    fn output_format_gst_strings() {
        assert_eq!(OutputFormat::Rgb.gst_format_str(), "RGB");
        assert_eq!(OutputFormat::Bgr.gst_format_str(), "BGR");
        assert_eq!(OutputFormat::Rgba.gst_format_str(), "RGBA");
    }

    #[test]
    fn output_format_to_pixel_format() {
        assert_eq!(OutputFormat::Rgb.to_pixel_format(), PixelFormat::Rgb8);
        assert_eq!(OutputFormat::Bgr.to_pixel_format(), PixelFormat::Bgr8);
        assert_eq!(OutputFormat::Rgba.to_pixel_format(), PixelFormat::Rgba8);
    }

    #[test]
    fn builder_chaining() {
        let b = PipelineBuilder::new(SourceSpec::rtsp("rtsp://test"))
            .decoder(DecoderSelection::ForceSoftware)
            .output_format(OutputFormat::Bgr)
            .latency_ms(500);
        assert_eq!(b.output_format, OutputFormat::Bgr);
        assert_eq!(b.latency_ms, 500);
        assert!(matches!(b.decoder, DecoderSelection::ForceSoftware));
    }
}

