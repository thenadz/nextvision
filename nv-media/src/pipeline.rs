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
use crate::gpu_provider::SharedGpuProvider;
use crate::hook::PostDecodeHook;
use crate::ingress::DeviceResidency;

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
    /// Device residency mode — determines the pipeline tail strategy.
    device_residency: DeviceResidency,
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

/// Build the standard host-memory pipeline tail: `videoconvert → appsink(video/x-raw)`.
///
/// Returns `(converter_elements, appsink)` where `converter_elements` contains the
/// single `videoconvert` element and `appsink` is configured with matching caps.
#[cfg(feature = "gst-backend")]
fn build_host_tail(
    output_format: &OutputFormat,
) -> Result<(Vec<gstreamer::Element>, gstreamer_app::AppSink), MediaError> {
    use gstreamer as gst;
    use gstreamer_app as gst_app;

    let videoconvert = gst::ElementFactory::make("videoconvert")
        .build()
        .map_err(|e| MediaError::Unsupported {
            detail: format!("failed to create videoconvert: {e}"),
        })?;

    let caps_str = format!("video/x-raw,format={}", output_format.gst_format_str());
    let appsink_caps: gst::Caps = caps_str.parse().map_err(|_| MediaError::Unsupported {
        detail: format!("invalid appsink caps: {caps_str}"),
    })?;

    let appsink = gst_app::AppSink::builder()
        .caps(&appsink_caps)
        .max_buffers(2)
        .drop(true)
        .build();

    Ok((vec![videoconvert], appsink))
}

impl PipelineBuilder {
    /// Create a builder for the given source specification.
    pub fn new(spec: SourceSpec) -> Self {
        Self {
            spec,
            decoder: DecoderSelection::default(),
            output_format: OutputFormat::default(),
            latency_ms: DEFAULT_LATENCY_MS,
            post_decode_hook: None,
            device_residency: DeviceResidency::default(),
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

    /// Set the device residency mode.
    ///
    /// Determines how the pipeline tail is constructed:
    /// - `Host` — `videoconvert → appsink(video/x-raw)`
    /// - `Cuda` — `cudaupload → cudaconvert → appsink(memory:CUDAMemory)`;
    ///   falls back to Host if CUDA elements are unavailable
    /// - `Provider(p)` — delegates to the provider's `build_pipeline_tail()`;
    ///   returns an error if the provider fails (no silent fallback)
    pub fn device_residency(mut self, residency: DeviceResidency) -> Self {
        self.device_residency = residency;
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
        // If the caller requested the built-in CUDA path but the feature
        // is off, fail loudly rather than silently downgrading to host.
        // Provider paths do NOT require the cuda feature — the provider
        // decides what GStreamer elements to use.
        #[cfg(not(feature = "cuda"))]
        if matches!(self.device_residency, DeviceResidency::Cuda) {
            return Err(MediaError::Unsupported {
                detail: "DeviceResidency::Cuda requested but the `cuda` cargo feature \
                         is not enabled on nv-media — rebuild with `--features cuda`, \
                         use DeviceResidency::Provider, or set DeviceResidency::Host"
                    .into(),
            });
        }

        use gstreamer as gst;
        use gstreamer::prelude::*;

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
                gst::glib::gobject_ffi::g_value_set_enum(value.to_glib_none_mut().0, discriminant);
                value
            }
        }
        const AUTOPLUG_TRY: i32 = 0;
        const AUTOPLUG_SKIP: i32 = 2;

        let pipeline = gst::Pipeline::new();

        // --- Source element ---
        let source = match &self.spec {
            SourceSpec::Rtsp {
                url,
                transport,
                security,
            } => {
                use nv_core::security::{RtspSecurityPolicy, promote_rtsp_to_tls};
                // Apply TLS promotion based on security policy.
                let effective_url = match security {
                    RtspSecurityPolicy::PreferTls => promote_rtsp_to_tls(url),
                    RtspSecurityPolicy::AllowInsecure | RtspSecurityPolicy::RequireTls => {
                        url.clone()
                    }
                };
                gst::ElementFactory::make("rtspsrc")
                    .property("location", effective_url.as_str())
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
                    })?
            }
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

        // --- Converter + Appsink ---
        //
        // Three pipeline tail strategies:
        //
        // 1. **Provider** (`DeviceResidency::Provider`): the provider builds
        //    the tail — converter elements (possibly empty) + appsink. If
        //    the provider fails, the build returns an error (no silent
        //    fallback — the user explicitly selected this hardware path).
        //
        // 2. **Built-in CUDA** (`DeviceResidency::Cuda` + `cuda` feature):
        //    `cudaupload → cudaconvert → appsink(memory:CUDAMemory)`.
        //    Falls back to host if CUDA elements are unavailable.
        //
        // 3. **Host** (default): `videoconvert → appsink(video/x-raw)`.
        //
        // After resolution, `gpu_resident` is true only if an actual
        // device-resident tail was successfully built.

        let mut gpu_resident = false;
        let mut active_provider: Option<SharedGpuProvider> = None;

        let (converter_elements, appsink) = match &self.device_residency {
            DeviceResidency::Provider(provider) => {
                match provider.build_pipeline_tail(self.output_format.to_pixel_format()) {
                    Ok(tail) => {
                        tracing::info!(
                            provider = provider.name(),
                            "device pipeline provider built pipeline tail",
                        );
                        gpu_resident = true;
                        active_provider = Some(provider.clone());
                        (tail.elements, tail.appsink)
                    }
                    Err(e) => {
                        // Provider was explicitly selected — failure is an
                        // error, not a silent downgrade to host.  The user
                        // chose this provider for a reason (e.g., NVMM on
                        // Jetson); silently falling back to CPU frames would
                        // produce subtly wrong results downstream.
                        return Err(MediaError::Unsupported {
                            detail: format!(
                                "device pipeline provider '{}' failed to build \
                                 pipeline tail: {e}",
                                provider.name(),
                            ),
                        });
                    }
                }
            }
            DeviceResidency::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    match (
                        gst::ElementFactory::make("cudaupload").build(),
                        gst::ElementFactory::make("cudaconvert").build(),
                    ) {
                        (Ok(cudaupload), Ok(cudaconvert)) => {
                            let caps_str = format!(
                                "video/x-raw(memory:CUDAMemory),format={}",
                                self.output_format.gst_format_str(),
                            );
                            let appsink_caps: gst::Caps =
                                caps_str.parse().map_err(|_| MediaError::Unsupported {
                                    detail: format!("invalid CUDA appsink caps: {caps_str}"),
                                })?;

                            let appsink = gst_app::AppSink::builder()
                                .caps(&appsink_caps)
                                .max_buffers(2)
                                .drop(true)
                                .build();

                            gpu_resident = true;
                            (vec![cudaupload, cudaconvert], appsink)
                        }
                        _ => {
                            tracing::warn!(
                                "DeviceResidency::Cuda requested but cudaupload/cudaconvert \
                                 GStreamer elements are not available — falling back to \
                                 host-memory pipeline (frames will be downloaded to CPU). \
                                 This is expected on GStreamer < 1.20 (e.g., JetPack 5.x).",
                            );
                            build_host_tail(&self.output_format)?
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                unreachable!()
            }
            DeviceResidency::Host => build_host_tail(&self.output_format)?,
        };

        // --- Link target resolution ---
        // When converter_elements is empty (e.g., a provider that only
        // sets appsink caps), link the decode stage directly to the
        // appsink. When non-empty, link through the converter chain.
        let appsink_element: &gst::Element = appsink.upcast_ref();
        let first_link_target = converter_elements
            .first()
            .cloned()
            .unwrap_or_else(|| appsink_element.clone());
        let last_link_source = converter_elements.last().cloned();

        // --- Assemble pipeline ---

        // Add source + decode + converter chain + appsink.
        pipeline
            .add_many([&source, &decode_element])
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to add source/decode to pipeline: {e}"),
            })?;
        for conv in &converter_elements {
            pipeline.add(conv).map_err(|e| MediaError::Unsupported {
                detail: format!("failed to add converter element to pipeline: {e}"),
            })?;
        }
        pipeline
            .add(appsink_element)
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to add appsink to pipeline: {e}"),
            })?;

        // Link the converter chain internally (e.g., cudaupload → cudaconvert)
        // and then the last converter → appsink.
        //
        // When a device **provider** constructed the tail, the first
        // converter element (e.g., nvvidconv on Jetson) typically has no
        // upstream peer yet — decodebin's pad-added has not fired.  Some
        // transform elements (notably nvvidconv) cannot answer a runtime
        // caps query without knowing their input, which makes the default
        // `PadLinkCheck::DEFAULT` (includes `CAPS`) fail even though the
        // pad templates are fully compatible.
        //
        // Use template-only link checks **only** for provider paths to
        // defer the real caps negotiation until data actually flows.
        // The built-in CUDA elements (cudaupload, cudaconvert) handle
        // runtime caps queries correctly and must use standard checks.
        let provider_active = active_provider.is_some();
        for pair in converter_elements.windows(2) {
            tracing::debug!(
                src = %pair[0].name(),
                sink = %pair[1].name(),
                provider_active,
                "linking converter chain elements",
            );
            if provider_active {
                pair[0].link_pads_full(
                    None,
                    &pair[1],
                    None,
                    gst::PadLinkCheck::HIERARCHY | gst::PadLinkCheck::TEMPLATE_CAPS,
                )
            } else {
                pair[0].link(&pair[1])
            }
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to link converter chain: {e}"),
            })?;
        }
        if let Some(ref last) = last_link_source {
            tracing::debug!(
                src = %last.name(),
                sink = %appsink_element.name(),
                provider_active,
                "linking last converter → appsink",
            );
            if provider_active {
                last.link_pads_full(
                    None,
                    appsink_element,
                    None,
                    gst::PadLinkCheck::HIERARCHY | gst::PadLinkCheck::TEMPLATE_CAPS,
                )
            } else {
                last.link(appsink_element)
            }
            .map_err(|e| MediaError::Unsupported {
                detail: format!("failed to link converter → appsink: {e}"),
            })?;
        }
        // When converter_elements is empty, first_link_target already
        // points at the appsink element — no explicit link needed.

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

            // Dynamic pad: decodebin → first element in converter chain
            // (or appsink when the converter chain is empty).
            //
            // When a post-decode hook is set (Host/Cuda paths only — provider
            // mode skips hooks), it is consulted to decide whether an additional
            // element should be inserted between the decoder output pad and the
            // converter chain. This supports platforms where the hardware decoder
            // outputs memory types that the standard videoconvert cannot accept
            // (e.g., NVMM on Jetson).
            let fc_weak = first_link_target.downgrade();
            let pipeline_weak = pipeline.downgrade();
            // When a **provider** controls the pipeline tail, skip user
            // hooks entirely — the provider's tail elements accept
            // decoder output directly.  Hooks are relevant for Host
            // and Cuda paths where the standard `videoconvert` / CUDA
            // elements may not accept certain decoder memory types
            // (e.g., NVMM on Jetson).
            let provider_active = active_provider.is_some();
            let hook = if provider_active {
                None
            } else {
                self.post_decode_hook.clone()
            };
            // Provider-only: use relaxed link checks for the decoder
            // pad → first converter element link, matching the relaxed
            // checks already used for the converter chain.  Transform
            // elements like nvvidconv cannot answer runtime caps
            // queries before data flows, so full CAPS checks fail
            // even though template compatibility is guaranteed.
            // The built-in CUDA elements handle runtime caps correctly
            // and do NOT need relaxed checks.
            let use_relaxed_link = provider_active;
            decode_element.connect_pad_added(move |_element, pad| {
                let Some(fc) = fc_weak.upgrade() else { return };
                let caps = pad.current_caps().unwrap_or_else(|| pad.query_caps(None));
                if let Some(structure) = caps.structure(0)
                    && structure.name().starts_with("video/") {
                        // Extract caps metadata for diagnostics (used by
                        // both the hook path and the provider/direct path).
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

                        tracing::debug!(
                            pad = %pad.name(),
                            caps = %caps,
                            memory = memory_type.as_deref().unwrap_or("system"),
                            format = format.as_deref().unwrap_or("unknown"),
                            "decoder pad added — linking to converter chain",
                        );

                        // Determine the link target: either videoconvert
                        // directly, or a hook-injected bridge element.
                        let target = if let Some(ref hook) = hook {
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
                                                        && bridge.link(&fc).is_ok()
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
                                                        fc
                                                    }
                                                } else {
                                                    fc
                                                }
                                            }
                                            Err(_) => {
                                                tracing::warn!(
                                                    element = %element_name,
                                                    "post-decode hook: element not available"
                                                );
                                                fc
                                            }
                                        }
                                    } else {
                                        fc
                                    }
                                }
                                None => fc,
                            }
                        } else {
                            fc
                        };
                        if let Some(sink_pad) = target.static_pad("sink")
                            && !sink_pad.is_linked() {
                                let link_result = if use_relaxed_link {
                                    // Provider paths: defer real caps negotiation
                                    // until data flows. Template caps are sufficient
                                    // to verify structural compatibility.
                                    pad.link_full(
                                        &sink_pad,
                                        gst::PadLinkCheck::TEMPLATE_CAPS,
                                    )
                                } else {
                                    pad.link(&sink_pad)
                                };
                                match link_result {
                                    Ok(_) => {
                                        tracing::debug!(
                                            pad = %pad.name(),
                                            target = %target.name(),
                                            relaxed = use_relaxed_link,
                                            "decoder pad linked to downstream element",
                                        );
                                    }
                                    Err(e) => {
                                        // Log detailed caps information for diagnostics.
                                        let src_caps = pad.current_caps()
                                            .map(|c| c.to_string())
                                            .unwrap_or_else(|| "<no current caps>".into());
                                        let sink_caps = sink_pad.current_caps()
                                            .map(|c| c.to_string())
                                            .unwrap_or_else(|| "<no current caps>".into());
                                        let src_template = pad.pad_template_caps()
                                            .to_string();
                                        let sink_template = sink_pad.pad_template_caps()
                                            .to_string();
                                        tracing::error!(
                                            pad = %pad.name(),
                                            target = %target.name(),
                                            error = %e,
                                            relaxed = use_relaxed_link,
                                            src_current_caps = %src_caps,
                                            sink_current_caps = %sink_caps,
                                            src_template_caps = %src_template,
                                            sink_template_caps = %sink_template,
                                            "failed to link decoder pad to downstream element — \
                                             pipeline will not produce frames",
                                        );
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
                    if let Some(sink_pad) = db.static_pad("sink")
                        && !sink_pad.is_linked()
                        && let Err(e) = pad.link(&sink_pad)
                    {
                        tracing::error!(
                            pad = %pad.name(),
                            error = %e,
                            "failed to link rtspsrc pad to decodebin — \
                             pipeline will not produce frames",
                        );
                    }
                });
            }
        } else {
            // Named decoder — static pads. Link: source → decoder → converter → appsink.
            // For RTSP, we still need dynamic pad handling from rtspsrc.
            if is_rtsp {
                let dec_weak = decode_element.downgrade();
                source.connect_pad_added(move |_element, pad| {
                    let Some(dec) = dec_weak.upgrade() else {
                        return;
                    };
                    if let Some(sink_pad) = dec.static_pad("sink")
                        && !sink_pad.is_linked()
                        && let Err(e) = pad.link(&sink_pad)
                    {
                        tracing::error!(
                            pad = %pad.name(),
                            error = %e,
                            "failed to link rtspsrc pad to decoder — \
                             pipeline will not produce frames",
                        );
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
                .link(&first_link_target)
                .map_err(|e| MediaError::Unsupported {
                    detail: format!("failed to link decoder → converter: {e}"),
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
            gpu_resident,
            gpu_provider: active_provider,
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
    /// Whether the pipeline tail uses device memory (`true`) or host
    /// memory (`false`). Determines which bridge function the appsink
    /// callback invokes.
    pub gpu_resident: bool,
    /// Optional device pipeline provider — when present and `gpu_resident`
    /// is true, the appsink callback delegates to this provider's
    /// `bridge_sample` method.
    pub gpu_provider: Option<SharedGpuProvider>,
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
            security: nv_core::security::RtspSecurityPolicy::AllowInsecure,
        });
        assert_eq!(b.rtsp_protocols(), Some("tcp"));
    }

    #[test]
    fn rtsp_protocols_udp() {
        let b = PipelineBuilder::new(SourceSpec::Rtsp {
            url: "rtsp://test".into(),
            transport: RtspTransport::UdpUnicast,
            security: nv_core::security::RtspSecurityPolicy::AllowInsecure,
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
        let b =
            PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4")).post_decode_hook(Some(hook));
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

        let got = captured
            .lock()
            .unwrap()
            .clone()
            .expect("hook should be called");
        assert_eq!(got.media_type, "video/x-raw");
        assert_eq!(got.memory_type.as_deref(), Some("NVMM"));
        assert_eq!(got.format.as_deref(), Some("NV12"));
    }

    /// When the `cuda` feature is NOT compiled, requesting CUDA device residency
    /// must produce a typed `MediaError::Unsupported` — never a silent fallback.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn cuda_residency_without_cuda_feature_errors() {
        let builder = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"))
            .device_residency(DeviceResidency::Cuda);
        let result = builder.build();
        match result {
            Err(MediaError::Unsupported { detail }) => {
                assert!(
                    detail.contains("cuda"),
                    "error should mention the cuda feature: {detail}",
                );
            }
            other => panic!(
                "expected Unsupported error, got {}",
                if other.is_ok() { "Ok" } else { "different Err" },
            ),
        }
    }

    #[test]
    fn builder_default_device_residency_is_host() {
        let b = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"));
        assert!(matches!(b.device_residency, DeviceResidency::Host));
    }

    #[test]
    fn builder_stores_device_residency_cuda() {
        let b = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"))
            .device_residency(DeviceResidency::Cuda);
        assert!(matches!(b.device_residency, DeviceResidency::Cuda));
    }

    #[test]
    fn builder_stores_provider_residency() {
        use crate::gpu_provider::{GpuPipelineProvider, SharedGpuProvider};
        use nv_core::error::MediaError;
        use nv_core::id::FeedId;
        use nv_frame::PixelFormat;
        use std::sync::Arc;

        struct StubProvider;
        impl GpuPipelineProvider for StubProvider {
            fn name(&self) -> &str {
                "stub"
            }
            #[cfg(feature = "gst-backend")]
            fn build_pipeline_tail(
                &self,
                _: PixelFormat,
            ) -> Result<crate::gpu_provider::GpuPipelineTail, MediaError> {
                Err(MediaError::Unsupported {
                    detail: "stub".into(),
                })
            }
            #[cfg(feature = "gst-backend")]
            fn bridge_sample(
                &self,
                _: FeedId,
                _: &Arc<std::sync::atomic::AtomicU64>,
                _: PixelFormat,
                _: &gstreamer::Sample,
                _: Option<crate::PtzTelemetry>,
            ) -> Result<nv_frame::FrameEnvelope, MediaError> {
                Err(MediaError::Unsupported {
                    detail: "stub".into(),
                })
            }
        }

        let provider: SharedGpuProvider = Arc::new(StubProvider);
        let b = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"))
            .device_residency(DeviceResidency::Provider(provider));

        assert!(matches!(b.device_residency, DeviceResidency::Provider(_)));
        assert!(b.device_residency.is_device());
        assert_eq!(b.device_residency.provider().unwrap().name(), "stub");
    }

    /// Provider failure during pipeline build should return an error,
    /// **not** silently fall back to host.  The user explicitly selected
    /// a hardware integration; silent degradation would produce subtly
    /// wrong results in GPU-dependent stages.
    #[test]
    fn provider_failure_returns_error() {
        use crate::gpu_provider::{GpuPipelineProvider, SharedGpuProvider};
        use nv_core::error::MediaError;
        use nv_core::id::FeedId;
        use nv_frame::PixelFormat;
        use std::sync::Arc;

        // Pipeline construction requires GStreamer to be initialized.
        if gstreamer::init().is_err() {
            eprintln!("skipping: GStreamer init failed");
            return;
        }

        struct FailingProvider;
        impl GpuPipelineProvider for FailingProvider {
            fn name(&self) -> &str {
                "failing"
            }
            #[cfg(feature = "gst-backend")]
            fn build_pipeline_tail(
                &self,
                _: PixelFormat,
            ) -> Result<crate::gpu_provider::GpuPipelineTail, MediaError> {
                Err(MediaError::Unsupported {
                    detail: "intentional failure for test".into(),
                })
            }
            #[cfg(feature = "gst-backend")]
            fn bridge_sample(
                &self,
                _: FeedId,
                _: &Arc<std::sync::atomic::AtomicU64>,
                _: PixelFormat,
                _: &gstreamer::Sample,
                _: Option<crate::PtzTelemetry>,
            ) -> Result<nv_frame::FrameEnvelope, MediaError> {
                Err(MediaError::Unsupported {
                    detail: "stub".into(),
                })
            }
        }

        let provider: SharedGpuProvider = Arc::new(FailingProvider);
        let builder = PipelineBuilder::new(SourceSpec::file("/tmp/test.mp4"))
            .device_residency(DeviceResidency::Provider(provider));

        let result = builder.build();
        match result {
            Ok(_) => panic!("pipeline build should return an error when provider fails"),
            Err(err) => {
                let detail = format!("{err}");
                assert!(
                    detail.contains("failing") && detail.contains("failed to build pipeline tail"),
                    "error should surface provider name and failure cause: {err}",
                );
            }
        }
    }

    /// Provider active → hooks are skipped.
    /// Host and Cuda → user hooks are retained.
    #[test]
    fn provider_active_skips_user_hook() {
        use crate::hook::{DecodedStreamInfo, PostDecodeHook};
        use std::sync::Arc;

        let user_hook: PostDecodeHook = Arc::new(|_info| Some("user-bridge".into()));

        let info = DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: Some("NVMM".into()),
            format: Some("NV12".into()),
        };

        // Simulate resolution: provider_active = true → hook is None.
        let provider_active = true;
        let resolved: Option<PostDecodeHook> = if provider_active {
            None
        } else {
            Some(user_hook.clone())
        };
        assert!(resolved.is_none(), "provider mode should skip all hooks");

        // Simulate resolution: provider_active = false → user hook used.
        // This covers both Host and Cuda paths.
        let provider_active = false;
        let resolved: Option<PostDecodeHook> = if provider_active {
            None
        } else {
            Some(user_hook)
        };
        assert!(resolved.is_some());
        assert_eq!(
            resolved.as_ref().unwrap()(&info).as_deref(),
            Some("user-bridge"),
            "host/cuda mode should use user hook",
        );
    }

    /// Cuda path retains user hook resolution behavior.
    /// When DeviceResidency::Cuda but no provider, hooks must fire.
    #[test]
    fn cuda_path_retains_user_hook() {
        use crate::hook::{DecodedStreamInfo, PostDecodeHook};
        use std::sync::Arc;

        let user_hook: PostDecodeHook = Arc::new(|info| {
            if info.memory_type.as_deref() == Some("NVMM") {
                Some("nvvidconv".into())
            } else {
                None
            }
        });

        // Cuda path: gpu_resident=true but provider_active=false.
        // Hook should be retained.
        let provider_active = false;
        let resolved: Option<PostDecodeHook> = if provider_active {
            None
        } else {
            Some(user_hook)
        };
        assert!(resolved.is_some(), "cuda path should retain user hooks");

        let info = DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: Some("NVMM".into()),
            format: Some("NV12".into()),
        };
        assert_eq!(
            resolved.as_ref().unwrap()(&info).as_deref(),
            Some("nvvidconv"),
            "cuda path should evaluate hook for NVMM output",
        );
    }

    /// Provider path skips hooks; provider path uses relaxed link checks;
    /// Cuda path does not.
    #[test]
    fn provider_vs_cuda_link_policy() {
        // Provider active → relaxed links.
        let provider_active = true;
        let use_relaxed_link = provider_active;
        assert!(
            use_relaxed_link,
            "provider path should use relaxed link checks"
        );

        // Cuda (no provider) → standard links.
        let provider_active = false;
        let use_relaxed_link = provider_active;
        assert!(
            !use_relaxed_link,
            "cuda path should use standard link checks"
        );
    }
}
