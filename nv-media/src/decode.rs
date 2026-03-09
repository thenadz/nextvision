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
    ForceSoftware,
    /// Request a specific GStreamer element by name (e.g., `"nvh264dec"`).
    ///
    /// Falls back to `Auto` if the named element is not available.
    Named(String),
}

impl Default for DecoderSelection {
    fn default() -> Self {
        Self::Auto
    }
}

/// Codec identifier parsed from the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    H264,
    H265,
    Vp8,
    Vp9,
    Av1,
    Mjpeg,
    Other,
}

impl Codec {
    /// Map a codec to the preferred GStreamer software decoder element name.
    pub fn software_decoder_name(self) -> &'static str {
        match self {
            Self::H264 => "avdec_h264",
            Self::H265 => "avdec_h265",
            Self::Vp8 => "vp8dec",
            Self::Vp9 => "vp9dec",
            Self::Av1 => "av1dec",
            Self::Mjpeg => "jpegdec",
            Self::Other => "decodebin",
        }
    }

    /// Map a codec to the preferred hardware decoder element names (in priority order).
    pub fn hardware_decoder_names(self) -> &'static [&'static str] {
        match self {
            Self::H264 => &["nvh264dec", "vaapih264dec", "vah264dec"],
            Self::H265 => &["nvh265dec", "vaapih265dec", "vah265dec"],
            Self::Vp9 => &["nvvp9dec", "vaapivp9dec"],
            Self::Av1 => &["nvav1dec", "vaapiav1dec"],
            _ => &[],
        }
    }
}
