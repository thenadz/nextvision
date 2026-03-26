//! Codec handling and hardware acceleration negotiation.
//!
//! Provides the public [`DecodePreference`] type for user-facing decode
//! configuration and the internal `DecoderSelection` type used by the pipeline
//! builder.
//!
//! # Decode selection model
//!
//! Selection proceeds in two stages:
//!
//! 1. **Preflight** — at session creation time, cached capability data
//!    from the backend registry is checked against the user's
//!    [`DecodePreference`]. If [`RequireHardware`](DecodePreference::RequireHardware)
//!    is set and no hardware decoder is discovered, the session fails
//!    immediately. This is a best-effort availability check — it does
//!    **not** guarantee the hardware decoder can handle the specific
//!    stream codec or profile.
//!
//! 2. **Post-selection verification** — after the backend has negotiated
//!    a decoder and the stream is confirmed flowing, the source layer
//!    inspects which decoder was actually selected. For
//!    `RequireHardware`, a software effective decoder triggers a typed
//!    `MediaError` instead of silent
//!    success. For all modes, a `HealthEvent::DecodeDecision` is
//!    emitted with the effective [`DecodeOutcome`].
//!
//! # Adaptive fallback
//!
//! For [`PreferHardware`](DecodePreference::PreferHardware), repeated
//! hardware decoder failures (before `StreamStarted` confirmation) cause
//! an internal `HwFailureTracker` to temporarily demote the selection
//! to `DecoderSelection::Auto`, preventing reconnect thrash. The
//! demotion has a bounded TTL and resets on success.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Re-export DecodeOutcome from nv-core so downstream can use it via nv-media.
pub use nv_core::health::DecodeOutcome;

// Re-export DecodePreference from nv-core (canonical definition).
pub use nv_core::health::DecodePreference;

// ---------------------------------------------------------------------------
// Internal — GStreamer decoder selection
// ---------------------------------------------------------------------------

/// Decoder selection strategy (internal to `nv-media`).
///
/// The pipeline builder uses this to determine which GStreamer decode element
/// to instantiate.
#[derive(Clone, Debug, Default)]
pub(crate) enum DecoderSelection {
    /// Automatically select the best decoder: prefer hardware, fall back to software.
    #[default]
    Auto,
    /// Force software decoding (useful for environments without GPU access).
    ForceSoftware,
    /// Require hardware decoding — configure `decodebin` to reject software
    /// video decoders via `autoplug-select`. If no hardware decoder can
    /// handle the stream, `decodebin` will error out.
    ForceHardware,
    /// Request a specific GStreamer element by name (e.g., `"nvh264dec"`).
    ///
    /// Falls back to `Auto` if the named element is not available.
    /// Not reachable from the public `DecodePreference` API;
    /// used for internal tests.
    #[allow(dead_code)]
    Named(String),
}

// ---------------------------------------------------------------------------
// Mapping: DecodePreference → DecoderSelection
// ---------------------------------------------------------------------------

/// Extension methods mapping [`DecodePreference`] to media-internal types.
///
/// `DecodePreference` is defined in `nv-core` (backend-neutral). This trait
/// adds the backend-specific mapping to [`DecoderSelection`] that only
/// `nv-media` needs.
pub(crate) trait DecodePreferenceExt {
    /// Map a user-facing preference to the internal decoder selection.
    ///
    /// - `Auto` → `Auto` (decodebin default ranking).
    /// - `CpuOnly` → `ForceSoftware`.
    /// - `PreferHardware` → `ForceHardware` (biased toward hardware; the
    ///   adaptive fallback cache may demote to `Auto` after repeated
    ///   failures).
    /// - `RequireHardware` → `ForceHardware` (rejects software video
    ///   decoders at the `autoplug-select` level).
    fn to_selection(self) -> DecoderSelection;

    /// Returns `true` if this preference demands hardware decode (fail-fast
    /// when unavailable).
    fn requires_hardware(self) -> bool;

    /// Returns `true` if this preference favours hardware but accepts
    /// software as a fallback.
    fn prefers_hardware(self) -> bool;
}

impl DecodePreferenceExt for DecodePreference {
    fn to_selection(self) -> DecoderSelection {
        match self {
            Self::Auto => DecoderSelection::Auto,
            Self::CpuOnly => DecoderSelection::ForceSoftware,
            Self::PreferHardware | Self::RequireHardware => DecoderSelection::ForceHardware,
        }
    }

    fn requires_hardware(self) -> bool {
        matches!(self, Self::RequireHardware)
    }

    fn prefers_hardware(self) -> bool {
        matches!(self, Self::PreferHardware)
    }
}

// ---------------------------------------------------------------------------
// Capability discovery
// ---------------------------------------------------------------------------

/// Lightweight capability information about the decode backend.
///
/// Obtained via [`discover_decode_capabilities()`]. This is a snapshot; the
/// underlying system state may change after construction (e.g., a GPU driver
/// crash).
///
/// # Examples
///
/// ```
/// use nv_media::DecodeCapabilities;
///
/// let caps = nv_media::discover_decode_capabilities();
/// if !caps.backend_available {
///     eprintln!("Media backend not compiled in or failed to initialise");
/// } else if caps.hardware_decode_available {
///     println!("Hardware decoders: {:?}", caps.known_decoder_names);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct DecodeCapabilities {
    /// Whether the media backend is compiled in and initialized
    /// successfully.
    ///
    /// When `false`, the remaining fields are meaningless — the backend
    /// is absent or failed to initialize. Operators can use this to
    /// distinguish a misbuild/misconfiguration from a genuine
    /// no-hardware condition (`backend_available == true` but
    /// `hardware_decode_available == false`).
    pub backend_available: bool,

    /// Whether at least one hardware video decoder was detected in the
    /// backend registry. Only meaningful when `backend_available` is `true`.
    pub hardware_decode_available: bool,

    /// Names of known hardware video decoder elements discovered in the
    /// backend registry. Empty when `hardware_decode_available` is `false`
    /// or the backend is not compiled in.
    pub known_decoder_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// Decode decision report (public)
// ---------------------------------------------------------------------------

/// Diagnostic report for a decode selection decision.
///
/// Created internally when a session starts and the backend identifies
/// which decoder element was selected. Published via
/// [`HealthEvent::DecodeDecision`](nv_core::health::HealthEvent::DecodeDecision)
/// and available for operator inspection via logging.
///
/// # Backend detail
///
/// The `backend_detail` field carries backend-specific information (e.g.,
/// the GStreamer element name). **Do not match on its contents** — it is
/// intended for logging and diagnostics only. Its format is not part of
/// the semver contract.
#[derive(Clone, Debug)]
pub struct DecodeDecisionInfo {
    /// The user-facing preference that was in effect.
    pub preference: DecodePreference,
    /// The effective decode outcome after backend negotiation.
    pub outcome: DecodeOutcome,
    /// Whether the adaptive fallback cache overrode the preference.
    pub fallback_active: bool,
    /// Human-readable reason for fallback (populated when
    /// `fallback_active` is `true`).
    pub fallback_reason: Option<String>,
    /// Backend-specific detail (e.g., element name). Debug-only —
    /// do not match on contents.
    pub backend_detail: String,
}

// ---------------------------------------------------------------------------
// Selected decoder info (internal)
// ---------------------------------------------------------------------------

/// Information about the decoder element selected by the backend.
///
/// Captured at pipeline negotiation time via the `element-added` signal
/// on `decodebin`. For named / non-decodebin decoders, set directly.
#[derive(Clone, Debug)]
pub(crate) struct SelectedDecoderInfo {
    /// Backend element name (e.g., `"nvh264dec"`, `"avdec_h264"`).
    pub element_name: String,
    /// Whether this element was classified as a hardware decoder.
    pub is_hardware: bool,
}

/// Thread-safe slot for communicating the selected decoder from the
/// pipeline's signal callback to the source layer.
pub(crate) type SelectedDecoderSlot = Arc<Mutex<Option<SelectedDecoderInfo>>>;

// ---------------------------------------------------------------------------
// Adaptive fallback cache (internal)
// ---------------------------------------------------------------------------

/// Number of consecutive hardware-decode failures that trigger a temporary
/// software fallback for [`DecodePreference::PreferHardware`] feeds.
const HW_FAILURE_THRESHOLD: u32 = 3;

/// Duration of the temporary software-only fallback window after a
/// threshold-triggered inhibition.
const FALLBACK_COOLDOWN: Duration = Duration::from_secs(60);

/// Bounded per-source hardware decoder failure memory.
///
/// Tracks consecutive hardware decoder failures and temporarily falls back
/// to [`DecoderSelection::Auto`] (from the normal `ForceHardware`) to
/// prevent reconnect thrash. Resets on success or after the cooldown
/// period expires.
///
/// This only applies to [`DecodePreference::PreferHardware`]. For
/// `RequireHardware`, the preflight and post-selection checks handle
/// enforcement; adding silent software fallback would violate the
/// user's explicit guarantee demand.
pub(crate) struct HwFailureTracker {
    /// Number of consecutive hardware decoder failures.
    consecutive_failures: u32,
    /// Timestamp of the most recent failure.
    last_failure: Option<Instant>,
    /// If set, hardware decode is temporarily inhibited until this time.
    fallback_until: Option<Instant>,
}

impl HwFailureTracker {
    pub fn new() -> Self {
        Self {
            consecutive_failures: 0,
            last_failure: None,
            fallback_until: None,
        }
    }

    /// Record a hardware decoder failure.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());
        if self.consecutive_failures >= HW_FAILURE_THRESHOLD && self.fallback_until.is_none() {
            self.fallback_until = Some(Instant::now() + FALLBACK_COOLDOWN);
            tracing::warn!(
                consecutive_failures = self.consecutive_failures,
                cooldown_secs = FALLBACK_COOLDOWN.as_secs(),
                "hardware decoder failure threshold reached, \
                 enabling temporary software fallback",
            );
        }
    }

    /// Record a successful decode session (stream confirmed flowing).
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.last_failure = None;
        self.fallback_until = None;
    }

    /// Whether the tracker recommends falling back to software.
    pub fn should_fallback(&self) -> bool {
        match self.fallback_until {
            Some(deadline) => Instant::now() < deadline,
            None => false,
        }
    }

    /// Adjust the decoder selection based on failure history.
    ///
    /// Returns `Some((adjusted_selection, reason))` if the tracker
    /// recommends overriding. Only applies to `PreferHardware` — other
    /// preferences are not adjusted.
    pub fn adjust_selection(&self, pref: DecodePreference) -> Option<(DecoderSelection, String)> {
        if !self.should_fallback() {
            return None;
        }
        match pref {
            DecodePreference::PreferHardware => Some((
                DecoderSelection::Auto,
                format!(
                    "adaptive fallback: {} consecutive hardware failures, \
                     temporarily allowing software decode",
                    self.consecutive_failures,
                ),
            )),
            // RequireHardware: never silently downgrade (violates user contract).
            // CpuOnly / Auto: no adjustment needed.
            _ => None,
        }
    }

    /// Number of consecutive failures (for testing/diagnostics).
    #[cfg(test)]
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Whether the tracker is currently in the fallback window.
    #[cfg(test)]
    pub fn is_in_fallback(&self) -> bool {
        self.should_fallback()
    }
}

/// Probe the media backend for hardware decode capabilities.
///
/// When the `gst-backend` feature is enabled, this queries the GStreamer
/// element registry for video decoder elements whose metadata hints at
/// hardware acceleration. The classification uses the element's klass
/// string (`"Hardware"` keyword) and a built-in list of known hardware
/// decoder name prefixes — see `is_hardware_video_decoder` for details.
///
/// When the `gst-backend` feature is **disabled**, this returns a
/// capabilities struct with `hardware_decode_available = false` and an
/// empty decoder list.
///
/// This function is intentionally cheap — it reads only the plugin registry
/// (no pipeline construction or device probing).
pub fn discover_decode_capabilities() -> DecodeCapabilities {
    #[cfg(feature = "gst-backend")]
    {
        discover_gst_hw_decoders()
    }
    #[cfg(not(feature = "gst-backend"))]
    {
        DecodeCapabilities {
            backend_available: false,
            hardware_decode_available: false,
            known_decoder_names: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared hardware-decoder classification
// ---------------------------------------------------------------------------

/// Known element-name prefixes for hardware video decoders.
///
/// This list covers the major GStreamer hardware decoder families.
/// If a hardware decoder plugin uses a prefix not in this list **and**
/// the element's `klass` metadata omits the `"Hardware"` keyword, the
/// decoder will not be recognized. File an issue or extend this list if
/// that happens.
pub(crate) const HW_DECODER_PREFIXES: &[&str] =
    &["nv", "va", "msdk", "amf", "qsv", "d3d11", "d3d12"];

/// Classify a GStreamer element as a hardware video decoder.
///
/// Returns `true` when the element is a video decoder that appears to be
/// hardware-accelerated. The check is heuristic — it succeeds when
/// **either** condition holds:
///
/// 1. The element's `klass` metadata contains both `"Decoder"` and
///    `"Video"` **and** `"Hardware"`.
/// 2. The element's name starts with a prefix in [`HW_DECODER_PREFIXES`]
///    **and** the klass contains `"Decoder"` and `"Video"`.
///
/// This function is the **single source of truth** for hardware decoder
/// classification. Both capability discovery and the `autoplug-select`
/// callback delegate to it.
pub(crate) fn is_hardware_video_decoder(klass: &str, element_name: &str) -> bool {
    let is_video_decoder = klass.contains("Decoder") && klass.contains("Video");
    if !is_video_decoder {
        return false;
    }
    klass.contains("Hardware")
        || HW_DECODER_PREFIXES
            .iter()
            .any(|p| element_name.starts_with(p))
}

/// GStreamer-specific hardware decoder discovery.
#[cfg(feature = "gst-backend")]
fn discover_gst_hw_decoders() -> DecodeCapabilities {
    use gst::prelude::*;
    use gstreamer as gst;

    // Ensure GStreamer is initialized.
    if gst::init().is_err() {
        return DecodeCapabilities {
            backend_available: false,
            hardware_decode_available: false,
            known_decoder_names: Vec::new(),
        };
    }

    let registry = gst::Registry::get();
    let mut hw_names: Vec<String> = Vec::new();

    for plugin in registry.plugins() {
        let features = registry.features_by_plugin(plugin.plugin_name().as_str());
        for feature in features {
            let factory: gst::ElementFactory = match feature.downcast() {
                Ok(f) => f,
                Err(_) => continue,
            };
            let klass: String = factory.metadata("klass").unwrap_or_default().into();
            let name = factory.name().to_string();
            if is_hardware_video_decoder(&klass, &name) {
                hw_names.push(name);
            }
        }
    }

    hw_names.sort();
    hw_names.dedup();

    DecodeCapabilities {
        backend_available: true,
        hardware_decode_available: !hw_names.is_empty(),
        known_decoder_names: hw_names,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_auto() {
        assert_eq!(DecodePreference::default(), DecodePreference::Auto);
    }

    #[test]
    fn auto_maps_to_auto_selection() {
        assert!(matches!(
            DecodePreference::Auto.to_selection(),
            DecoderSelection::Auto
        ));
    }

    #[test]
    fn cpu_only_maps_to_force_software() {
        assert!(matches!(
            DecodePreference::CpuOnly.to_selection(),
            DecoderSelection::ForceSoftware
        ));
    }

    #[test]
    fn prefer_hardware_maps_to_force_hardware_selection() {
        assert!(matches!(
            DecodePreference::PreferHardware.to_selection(),
            DecoderSelection::ForceHardware
        ));
    }

    #[test]
    fn require_hardware_maps_to_force_hardware() {
        assert!(matches!(
            DecodePreference::RequireHardware.to_selection(),
            DecoderSelection::ForceHardware
        ));
    }

    #[test]
    fn requires_hardware_only_for_require_hardware() {
        assert!(!DecodePreference::Auto.requires_hardware());
        assert!(!DecodePreference::CpuOnly.requires_hardware());
        assert!(!DecodePreference::PreferHardware.requires_hardware());
        assert!(DecodePreference::RequireHardware.requires_hardware());
    }

    #[test]
    fn decode_preference_clone_and_eq() {
        let a = DecodePreference::PreferHardware;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn prefers_hardware_flag() {
        assert!(!DecodePreference::Auto.prefers_hardware());
        assert!(!DecodePreference::CpuOnly.prefers_hardware());
        assert!(DecodePreference::PreferHardware.prefers_hardware());
        assert!(!DecodePreference::RequireHardware.prefers_hardware());
    }

    #[test]
    fn capabilities_struct_consistent() {
        let caps = discover_decode_capabilities();
        if caps.hardware_decode_available {
            assert!(caps.backend_available, "hardware implies backend available");
            assert!(!caps.known_decoder_names.is_empty());
        }
        if !caps.backend_available {
            assert!(!caps.hardware_decode_available);
            assert!(caps.known_decoder_names.is_empty());
        }
    }

    #[test]
    fn capabilities_backend_available_reflects_feature() {
        let caps = discover_decode_capabilities();
        // When gst-backend is compiled in AND gst::init succeeds,
        // backend_available is true. Without the feature it is false.
        // We can't hard-assert the value because the feature may or
        // may not be enabled, but the struct must be self-consistent.
        if caps.backend_available {
            // backend OK — hardware may or may not be present
        } else {
            assert!(!caps.hardware_decode_available);
        }
    }

    // -----------------------------------------------------------------------
    // is_hardware_video_decoder classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn hw_classifier_rejects_non_video_decoder() {
        // A demuxer is never a hardware video decoder, regardless of name.
        assert!(!is_hardware_video_decoder("Codec/Demuxer", "nvdemux"));
        assert!(!is_hardware_video_decoder("Source/Video", "vasrc"));
    }

    #[test]
    fn hw_classifier_klass_hardware_keyword() {
        // Element with explicit "Hardware" in klass → hardware decoder.
        assert!(is_hardware_video_decoder(
            "Codec/Decoder/Video/Hardware",
            "exotic_decoder"
        ));
    }

    #[test]
    fn hw_classifier_known_prefix_without_hardware_klass() {
        // Elements matching known prefixes but without "Hardware" in klass.
        for prefix in HW_DECODER_PREFIXES {
            let name = format!("{prefix}h264dec");
            assert!(
                is_hardware_video_decoder("Codec/Decoder/Video", &name),
                "expected {name} to be classified as hardware",
            );
        }
    }

    #[test]
    fn hw_classifier_rejects_software_decoder() {
        assert!(!is_hardware_video_decoder(
            "Codec/Decoder/Video",
            "avdec_h264"
        ));
        assert!(!is_hardware_video_decoder(
            "Codec/Decoder/Video",
            "openh264dec"
        ));
        assert!(!is_hardware_video_decoder(
            "Codec/Decoder/Video",
            "libde265dec"
        ));
    }

    #[test]
    fn hw_classifier_unknown_prefix_with_hardware_klass() {
        // A hypothetical new vendor decoder that doesn't match any prefix
        // but correctly sets "Hardware" in its klass.
        assert!(is_hardware_video_decoder(
            "Codec/Decoder/Video/Hardware",
            "newvendordec"
        ));
    }
}
