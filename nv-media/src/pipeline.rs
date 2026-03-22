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

use crate::decode::{DecoderSelection, SelectedDecoderSlot};
use crate::hook::PostDecodeHook;

/// Target pixel format for the appsink output.
///
/// This controls the `video/x-raw,format=...` caps set on the appsink.
/// The `videoconvert` element handles the conversion from whatever the
/// decoder outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    /// 8-bit RGB (3 bytes per pixel).
    #[default]
    Rgb,
    /// 8-bit BGR (3 bytes per pixel, OpenCV-native ordering).
    #[allow(dead_code)] // used in gst-backend format negotiation
    Bgr,
    /// 8-bit RGBA (4 bytes per pixel).
    #[allow(dead_code)] // used in gst-backend format negotiation
    Rgba,
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
    /// Optional post-decode hook for injecting elements.
    post_decode_hook: Option<PostDecodeHook>,
}

/// Default latency hint for RTSP jitter buffers.
const DEFAULT_LATENCY_MS: u32 = 200;

/// Default TCP timeout for RTSP sources, in microseconds.
///
/// Controls how long the TCP interleaved connection waits before
/// declaring a timeout on a stalled connection (e.g., network outage).
/// 10 seconds balances normal jitter tolerance against prompt failure
/// detection. GStreamer posts a bus Error when this fires, which the
/// source FSM maps to a reconnection attempt.
const DEFAULT_RTSP_TCP_TIMEOUT_US: u64 = 10_000_000;

impl PipelineBuilder {
    /// Create a builder for the given source specification.
    pub fn new(spec: SourceSpec) -> Self {
        Self {
            spec,
            decoder: DecoderSelection::default(),
            output_format: OutputFormat::default(),
            latency_ms: DEFAULT_LATENCY_MS,
            post_decode_hook: None,
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

    /// Set the post-decode hook.
    pub fn post_decode_hook(mut self, hook: Option<PostDecodeHook>) -> Self {
        self.post_decode_hook = hook;
        self
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

        // GstAutoplugSelectResult is a C GEnum registered by decodebin.
        // gstreamer-rs doesn't expose it as a Rust type. We construct the
        // properly-typed glib::Value manually so that the glib signal
        // marshaller accepts it (glib-rs >= 0.22 validates return types).
        //
        //   GST_AUTOPLUG_SELECT_TRY    = 0
        //   GST_AUTOPLUG_SELECT_EXPOSE = 1
        //   GST_AUTOPLUG_SELECT_SKIP   = 2
        fn autoplug_select_result(discriminant: i32) -> gst::glib::Value {
            use gst::glib::translate::ToGlibPtrMut;
            let Some(gtype) = gst::glib::Type::from_name("GstAutoplugSelectResult") else {
                // decodebin is not loaded — fall back to the raw integer value.
                // GStreamer will coerce it, but may log a type warning.
                tracing::error!(
                    "GstAutoplugSelectResult GType not registered — is decodebin loaded?"
                );
                return discriminant.into();
            };
            unsafe {
                let mut value = gst::glib::Value::from_type(gtype);
                gst::glib::gobject_ffi::g_value_set_enum(
                    value.to_glib_none_mut().0,
                    discriminant,
                );
                value
            }
        }
        const AUTOPLUG_TRY: i32 = 0;
        const AUTOPLUG_SKIP: i32 = 2;

        let pipeline = gst::Pipeline::new();

        // --- Source element ---
        let source = match &self.spec {
            SourceSpec::Rtsp { url, transport } => gst::ElementFactory::make("rtspsrc")
                .property("location", url.as_str())
                .property("latency", self.latency_ms)
                .property("tcp-timeout", DEFAULT_RTSP_TCP_TIMEOUT_US)
                .property_from_str(
                    "protocols",
                    match transport {
                        RtspTransport::Tcp => "tcp",
                        RtspTransport::UdpUnicast => "udp-unicast",
                    },
                )
                .build()
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to create rtspsrc: {e}"),
                })?,
            SourceSpec::File { path, loop_: _ } => gst::ElementFactory::make("filesrc")
                .property("location", path.to_string_lossy().as_ref())
                .build()
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to create filesrc: {e}"),
                })?,
            SourceSpec::V4l2 { device } => gst::ElementFactory::make("v4l2src")
                .property("device", device.as_str())
                .build()
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to create v4l2src: {e}"),
                })?,
            SourceSpec::Custom { pipeline_fragment } => {
                gst::parse::bin_from_description(pipeline_fragment, true)
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
        let selected_decoder: SelectedDecoderSlot =
            std::sync::Arc::new(std::sync::Mutex::new(None));

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
                    .build()
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to create decodebin (force-sw): {e}"),
                    })?;
                // Reject hardware video decoders via autoplug-select so
                // decodebin only considers software decoders. This avoids
                // relying on the `force-sw-decoders` property which requires
                // GStreamer >= 1.18 (Jetson ships 1.16).
                db.connect("autoplug-select", false, |values| {
                    use crate::decode::is_hardware_video_decoder;

                    let factory: gst::ElementFactory = match values
                        .get(3)
                        .and_then(|v| v.get::<gst::ElementFactory>().ok())
                    {
                        Some(f) => f,
                        None => return Some(autoplug_select_result(AUTOPLUG_TRY)),
                    };
                    let klass: String = factory.metadata("klass").unwrap_or_default().into();
                    let name = factory.name();

                    if is_hardware_video_decoder(&klass, name.as_str()) {
                        tracing::debug!(
                            element = %name,
                            klass = %klass,
                            "autoplug-select: skipping hardware video decoder \
                             (ForceSoftware mode)",
                        );
                        return Some(autoplug_select_result(AUTOPLUG_SKIP));
                    }
                    Some(autoplug_select_result(AUTOPLUG_TRY))
                });
                (db, true)
            }
            DecoderSelection::ForceHardware => {
                let db = gst::ElementFactory::make("decodebin")
                    .build()
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to create decodebin (force-hw): {e}"),
                    })?;
                // Reject software video decoders at the autoplug level so
                // decodebin only considers hardware decoders. If none can
                // handle the stream, decodebin will post an error — which is
                // the correct behaviour for RequireHardware.
                //
                // Classification is delegated to `is_hardware_video_decoder`
                // (the single source of truth shared with capability discovery).
                db.connect("autoplug-select", false, |values| {
                    use crate::decode::is_hardware_video_decoder;

                    // values: &[decodebin, pad, caps, factory]
                    let factory: gst::ElementFactory = match values
                        .get(3)
                        .and_then(|v| v.get::<gst::ElementFactory>().ok())
                    {
                        Some(f) => f,
                        None => {
                            tracing::warn!(
                                "autoplug-select: malformed callback payload, \
                                 allowing element (safe default)",
                            );
                            return Some(autoplug_select_result(AUTOPLUG_TRY));
                        }
                    };
                    let klass: String = factory.metadata("klass").unwrap_or_default().into();
                    let name = factory.name();

                    // Non-video-decoder elements (demuxers, parsers, etc.)
                    // are always allowed through.
                    if !is_hardware_video_decoder(&klass, name.as_str()) {
                        let is_video_decoder = klass.contains("Decoder") && klass.contains("Video");
                        if !is_video_decoder {
                            return Some(autoplug_select_result(AUTOPLUG_TRY));
                        }
                        tracing::debug!(
                            element = %name,
                            klass = %klass,
                            "autoplug-select: skipping software video decoder \
                             (ForceHardware mode)",
                        );
                        return Some(autoplug_select_result(AUTOPLUG_SKIP));
                    }
                    tracing::debug!(
                        element = %name,
                        klass = %klass,
                        "autoplug-select: accepting hardware video decoder",
                    );
                    Some(autoplug_select_result(AUTOPLUG_TRY))
                });
                (db, true)
            }
            DecoderSelection::Named(name) => {
                match gst::ElementFactory::make(name.as_str()).build() {
                    Ok(elem) => {
                        // Populate selected decoder directly — no decodebin.
                        if let Some(factory) = elem.factory() {
                            use crate::decode::{SelectedDecoderInfo, is_hardware_video_decoder};
                            let klass: String =
                                factory.metadata("klass").unwrap_or_default().into();
                            let is_hw = is_hardware_video_decoder(&klass, name.as_str());
                            if let Ok(mut slot) = selected_decoder.lock() {
                                *slot = Some(SelectedDecoderInfo {
                                    element_name: name.clone(),
                                    is_hardware: is_hw,
                                });
                            }
                        }
                        (elem, false)
                    }
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

        // Wire `element-added` on decodebin to capture the effective
        // video decoder. For non-decodebin pipelines (Named) the slot
        // is populated directly in the match arm above.
        if uses_decodebin {
            let slot_clone = std::sync::Arc::clone(&selected_decoder);
            decode_element.connect("element-added", false, move |values| {
                use crate::decode::{SelectedDecoderInfo, is_hardware_video_decoder};
                let element: gst::Element =
                    match values.get(1).and_then(|v| v.get::<gst::Element>().ok()) {
                        Some(e) => e,
                        None => {
                            tracing::warn!("element-added: malformed callback payload, ignoring",);
                            return None;
                        }
                    };
                if let Some(factory) = element.factory() {
                    let klass: String = factory.metadata("klass").unwrap_or_default().into();
                    if klass.contains("Decoder") && klass.contains("Video") {
                        let name = factory.name().to_string();
                        let is_hw = is_hardware_video_decoder(&klass, &name);
                        if let Ok(mut slot) = slot_clone.lock() {
                            *slot = Some(SelectedDecoderInfo {
                                element_name: name,
                                is_hardware: is_hw,
                            });
                        }
                    }
                }
                None
            });
        }

        // --- Videoconvert + Appsink ---
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to create videoconvert: {e}"),
            })?;

        let caps_str = format!("video/x-raw,format={}", self.output_format.gst_format_str());
        let appsink_caps: gst::Caps = caps_str.parse().map_err(|_| MediaError::Unsupported {
            detail: format!("invalid appsink caps: {caps_str}"),
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
                source
                    .link(&decode_element)
                    .map_err(|e| MediaError::Unsupported {
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
            //
            // When a post-decode hook is provided, it is consulted to decide
            // whether an additional element should be inserted between the
            // decoder output pad and videoconvert. This supports platforms
            // where the hardware decoder outputs memory types that the
            // standard videoconvert cannot accept (e.g., NVMM on Jetson).
            let vc_weak = videoconvert.downgrade();
            let pipeline_weak = pipeline.downgrade();
            let hook = self.post_decode_hook.clone();
            decode_element.connect_pad_added(move |_element, pad| {
                let Some(vc) = vc_weak.upgrade() else { return };
                let caps = pad.current_caps().unwrap_or_else(|| pad.query_caps(None));
                if let Some(structure) = caps.structure(0) {
                    if structure.name().starts_with("video/") {
                        // Determine the link target: either videoconvert
                        // directly, or a hook-injected bridge element.
                        let target = if let Some(ref hook) = hook {
                            // Use the GStreamer caps features API to extract
                            // the memory type (e.g., "NVMM") instead of
                            // parsing the caps string representation.
                            let memory_type = caps.features(0).and_then(|features| {
                                (0..features.size()).find_map(|i| {
                                    features
                                        .nth(i)
                                        .and_then(|f| {
                                            f.as_str()
                                                .strip_prefix("memory:")
                                                .map(String::from)
                                        })
                                })
                            });
                            let format = structure
                                .get::<&str>("format")
                                .ok()
                                .map(String::from);
                            let info = crate::hook::DecodedStreamInfo {
                                media_type: structure.name().to_string(),
                                memory_type,
                                format,
                            };
                            match hook(&info) {
                                Some(element_name) => {
                                    if let Some(pipeline) = pipeline_weak.upgrade() {
                                        match gst::ElementFactory::make(element_name.as_str()).build() {
                                            Ok(bridge) => {
                                                if pipeline.add(&bridge).is_ok() {
                                                    if bridge.sync_state_with_parent().is_ok()
                                                        && bridge.link(&vc).is_ok()
                                                    {
                                                        tracing::info!(
                                                            element = %element_name,
                                                            "post-decode hook: inserted bridge element"
                                                        );
                                                        bridge
                                                    } else {
                                                        tracing::warn!(
                                                            element = %element_name,
                                                            "post-decode hook: bridge link/sync failed, \
                                                             falling back to direct link"
                                                        );
                                                        let _ = pipeline.remove(&bridge);
                                                        vc
                                                    }
                                                } else {
                                                    vc
                                                }
                                            }
                                            Err(_) => {
                                                tracing::warn!(
                                                    element = %element_name,
                                                    "post-decode hook: element not available"
                                                );
                                                vc
                                            }
                                        }
                                    } else {
                                        vc
                                    }
                                }
                                None => vc,
                            }
                        } else {
                            vc
                        };
                        if let Some(sink_pad) = target.static_pad("sink") {
                            if !sink_pad.is_linked() {
                                if let Err(e) = pad.link(&sink_pad) {
                                    tracing::error!(
                                        pad = %pad.name(),
                                        target = %target.name(),
                                        error = %e,
                                        "failed to link decoder pad to downstream element — \
                                         pipeline will not produce frames",
                                    );
                                }
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
                            if let Err(e) = pad.link(&sink_pad) {
                                tracing::error!(
                                    pad = %pad.name(),
                                    error = %e,
                                    "failed to link rtspsrc pad to decodebin — \
                                     pipeline will not produce frames",
                                );
                            }
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
                    let Some(dec) = dec_weak.upgrade() else {
                        return;
                    };
                    if let Some(sink_pad) = dec.static_pad("sink") {
                        if !sink_pad.is_linked() {
                            if let Err(e) = pad.link(&sink_pad) {
                                tracing::error!(
                                    pad = %pad.name(),
                                    error = %e,
                                    "failed to link rtspsrc pad to decoder — \
                                     pipeline will not produce frames",
                                );
                            }
                        }
                    }
                });
            } else {
                source
                    .link(&decode_element)
                    .map_err(|e| MediaError::Unsupported {
                        detail: format!("failed to link source → decoder: {e}"),
                    })?;
            }

            decode_element
                .link(&videoconvert)
                .map_err(|e| MediaError::Unsupported {
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
            selected_decoder,
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
    /// Shared slot that captures the effective video decoder element.
    /// Populated by `element-added` signal on decodebin, or directly
    /// for named decoders.
    pub selected_decoder: SelectedDecoderSlot,
}

/// Stub for non-GStreamer builds (never constructed).
#[cfg(not(feature = "gst-backend"))]
pub(crate) struct BuiltPipeline {
    _private: (),
}

#[cfg(test)]
impl PipelineBuilder {
    pub fn latency_ms(mut self, ms: u32) -> Self {
        self.latency_ms = ms;
        self
    }

    pub fn source_element_name(&self) -> &'static str {
        match &self.spec {
            SourceSpec::Rtsp { .. } => "rtspsrc",
            SourceSpec::File { .. } => "filesrc",
            SourceSpec::V4l2 { .. } => "v4l2src",
            SourceSpec::Custom { .. } => "custom",
        }
    }

    fn rtsp_protocols(&self) -> Option<&'static str> {
        match &self.spec {
            SourceSpec::Rtsp { transport, .. } => Some(match transport {
                RtspTransport::Tcp => "tcp",
                RtspTransport::UdpUnicast => "udp",
            }),
            _ => None,
        }
    }
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

    #[test]
    fn builder_stores_post_decode_hook() {
        let hook: PostDecodeHook = std::sync::Arc::new(|_info| None);
        let b = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"))
            .post_decode_hook(Some(hook));
        assert!(b.post_decode_hook.is_some());
    }

    #[test]
    fn builder_post_decode_hook_defaults_to_none() {
        let b = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"));
        assert!(b.post_decode_hook.is_none());
    }

    #[test]
    fn hook_returns_bridge_for_nvmm() {
        use crate::hook::DecodedStreamInfo;

        let hook: PostDecodeHook = std::sync::Arc::new(|info| {
            if info.memory_type.as_deref() == Some("NVMM") {
                Some("nvvidconv".into())
            } else {
                None
            }
        });

        let nvmm_info = DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: Some("NVMM".into()),
            format: Some("NV12".into()),
        };
        assert_eq!(hook(&nvmm_info), Some("nvvidconv".into()));

        let system_info = DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: None,
            format: Some("I420".into()),
        };
        assert_eq!(hook(&system_info), None);
    }

    #[test]
    fn hook_receives_all_fields() {
        use crate::hook::DecodedStreamInfo;
        use std::sync::{Arc, Mutex};

        let captured = Arc::new(Mutex::new(None));
        let captured_clone = Arc::clone(&captured);

        let hook: PostDecodeHook = std::sync::Arc::new(move |info| {
            *captured_clone.lock().unwrap() = Some(info.clone());
            None
        });

        let info = DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: Some("NVMM".into()),
            format: Some("NV12".into()),
        };
        let _ = hook(&info);

        let got = captured.lock().unwrap().clone().expect("hook should be called");
        assert_eq!(got.media_type, "video/x-raw");
        assert_eq!(got.memory_type.as_deref(), Some("NVMM"));
        assert_eq!(got.format.as_deref(), Some("NV12"));
    }
}
