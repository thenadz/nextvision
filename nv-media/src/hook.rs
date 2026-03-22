//! Post-decode hook types for platform-specific pipeline element injection.
//!
//! These types allow callers to inject a GStreamer element between the decoder
//! output and the color-space converter. This is needed on platforms where the
//! hardware decoder outputs memory types that the standard `videoconvert`
//! cannot accept (e.g., `memory:NVMM` on NVIDIA Jetson).
//!
//! The hook is a callback that inspects the decoded stream characteristics
//! and optionally returns a GStreamer element name to insert.

use std::sync::Arc;

/// Information about a decoded stream, provided to [`PostDecodeHook`] callbacks.
///
/// Describes the decoded media stream characteristics so platform-specific
/// hooks can decide whether additional pipeline elements are needed.
#[derive(Debug, Clone)]
pub struct DecodedStreamInfo {
    /// Media type string (e.g., `"video/x-raw"`).
    pub media_type: String,
    /// Optional memory type qualifier (e.g., `Some("NVMM")` for Jetson
    /// GPU-mapped buffers, `None` for system memory).
    pub memory_type: Option<String>,
    /// Optional pixel format string (e.g., `Some("NV12")`).
    pub format: Option<String>,
}

/// Hook invoked once per feed when the decoded stream's caps are known.
///
/// The hook receives a [`DecodedStreamInfo`] describing the decoded stream
/// and returns an optional GStreamer element name to insert between the
/// decoder and the color-space converter. Returning `None` means no
/// additional element is needed.
///
/// # Example
///
/// On Jetson, hardware decoders output `video/x-raw(memory:NVMM)` — GPU-mapped
/// buffers that the standard `videoconvert` cannot accept. A post-decode hook
/// can bridge this:
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use nv_media::PostDecodeHook;
///
/// let hook: PostDecodeHook = Arc::new(|info| {
///     if info.memory_type.as_deref() == Some("NVMM") {
///         Some("nvvidconv".into())
///     } else {
///         None
///     }
/// });
/// ```
pub type PostDecodeHook =
    Arc<dyn Fn(&DecodedStreamInfo) -> Option<String> + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_stream_info_clone() {
        let info = DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: Some("NVMM".into()),
            format: Some("NV12".into()),
        };
        let cloned = info.clone();
        assert_eq!(cloned.media_type, "video/x-raw");
        assert_eq!(cloned.memory_type.as_deref(), Some("NVMM"));
        assert_eq!(cloned.format.as_deref(), Some("NV12"));
    }

    #[test]
    fn decoded_stream_info_none_fields() {
        let info = DecodedStreamInfo {
            media_type: "video/x-raw".into(),
            memory_type: None,
            format: None,
        };
        assert!(info.memory_type.is_none());
        assert!(info.format.is_none());
    }

    #[test]
    fn hook_type_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PostDecodeHook>();
    }
}
