//! Codec handling and hardware acceleration negotiation.
//!
//! Provides decoder selection logic: prefer hardware decode (VA-API, NVDEC)
//! when available, fall back to software decode.

/// Decoder selection strategy.
///
/// The pipeline builder uses this to determine which GStreamer decode element
/// to instantiate.
#[derive(Clone, Debug)]
pub(crate) enum DecoderSelection {
    /// Automatically select the best decoder: prefer hardware, fall back to software.
    Auto,
    /// Force software decoding (useful for environments without GPU access).
    #[allow(dead_code)] // constructed in tests and gst-backend pipeline build
    ForceSoftware,
    /// Request a specific GStreamer element by name (e.g., `"nvh264dec"`).
    ///
    /// Falls back to `Auto` if the named element is not available.
    #[allow(dead_code)] // constructed in tests and gst-backend pipeline build
    Named(String),
}

impl Default for DecoderSelection {
    fn default() -> Self {
        Self::Auto
    }
}


