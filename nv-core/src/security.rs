//! Security utilities: URL credential redaction, error sanitization, and RTSP
//! transport security policy.
//!
//! # Credential redaction
//!
//! [`redact_url`] strips `user:password@` from URLs while preserving the
//! host, port, and path for diagnostic purposes. This prevents credentials
//! from leaking into logs, health events, or error messages.
//!
//! # Error sanitization
//!
//! [`sanitize_error_string`] cleans untrusted backend error/debug strings by:
//! - Stripping control characters and bare newlines.
//! - Capping to a configurable maximum length.
//! - Redacting patterns that resemble secrets (e.g., `password=...`,
//!   `token=...`, `key=...`).
//!
//! # RTSP security policy
//!
//! [`RtspSecurityPolicy`] controls whether `rtsps://` (TLS) is preferred,
//! required, or explicitly opted-out for RTSP sources.
//!
//! ## Threat model
//!
//! RTSP streams carry both video data and sometimes credentials in the URL.
//! Without TLS:
//! - Credentials may be visible to network observers (man-in-the-middle).
//! - Video data is transmitted in the clear.
//! - An attacker on the network can spoof or tamper with the stream.
//!
//! `PreferTls` (the default) upgrades bare `rtsp://` URLs to `rtsps://` so
//! that production deployments default to encrypted transport without
//! requiring code changes. Field deployments behind firewalls or with
//! cameras that don't support TLS can opt out with `AllowInsecure`.
//!
//! ## Migration path
//!
//! 1. Existing code that passes explicit `rtsp://` URLs will continue to
//!    work — the URL is promoted to `rtsps://` unless `AllowInsecure` is
//!    set or the URL already uses `rtsps://`.
//! 2. If a camera does not support TLS, set `AllowInsecure` on the
//!    source spec. A health warning will be emitted.
//! 3. For high-security deployments, set `RequireTls` to reject any
//!    unencrypted RTSP source at config validation time.

/// RTSP transport security policy.
///
/// Controls whether `rtsps://` (TLS) is preferred, required, or
/// explicitly opted-out for RTSP sources.
///
/// The default is [`PreferTls`](Self::PreferTls).
///
/// # Examples
///
/// ```
/// use nv_core::security::RtspSecurityPolicy;
///
/// // Default: prefer TLS — bare rtsp:// URLs are promoted to rtsps://
/// let policy = RtspSecurityPolicy::default();
/// assert_eq!(policy, RtspSecurityPolicy::PreferTls);
///
/// // Explicit opt-out for cameras that don't support TLS
/// let policy = RtspSecurityPolicy::AllowInsecure;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RtspSecurityPolicy {
    /// Default: promote bare `rtsp://` to `rtsps://` when scheme is absent
    /// or `rtsp`. Logs a warning if the final URL is still insecure
    /// (e.g., camera doesn't support TLS and caller forces `AllowInsecure`).
    #[default]
    PreferTls,

    /// Allow insecure `rtsp://` without promotion. A health warning is
    /// emitted when an insecure source is used. Use this for cameras that
    /// do not support TLS behind trusted networks.
    AllowInsecure,

    /// Reject any RTSP source that is not `rtsps://`. Returns a config
    /// error at feed creation time if the URL scheme is `rtsp://`.
    RequireTls,
}
impl std::fmt::Display for RtspSecurityPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreferTls => f.write_str("PreferTls"),
            Self::AllowInsecure => f.write_str("AllowInsecure"),
            Self::RequireTls => f.write_str("RequireTls"),
        }
    }
}

/// Whether `SourceSpec::Custom` pipeline fragments are trusted.
///
/// Custom pipeline fragments are raw GStreamer launch-line strings. In
/// production, accepting arbitrary pipeline strings from untrusted config
/// is a security risk. This policy gates custom pipelines behind an
/// explicit opt-in.
///
/// The default is [`Reject`](Self::Reject).
///
/// # Examples
///
/// ```
/// use nv_core::security::CustomPipelinePolicy;
///
/// // Default: reject custom pipelines
/// let policy = CustomPipelinePolicy::default();
/// assert_eq!(policy, CustomPipelinePolicy::Reject);
///
/// // Explicit opt-in for development/trusted config
/// let policy = CustomPipelinePolicy::AllowTrusted;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CustomPipelinePolicy {
    /// Reject `SourceSpec::Custom` at config validation time with a
    /// clear error message explaining how to opt in.
    #[default]
    Reject,

    /// Allow custom pipeline fragments. Use only when the pipeline
    /// string originates from a trusted source (e.g., hard-coded in
    /// application code, not from user input or config files).
    AllowTrusted,
}
impl std::fmt::Display for CustomPipelinePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reject => f.write_str("Reject"),
            Self::AllowTrusted => f.write_str("AllowTrusted"),
        }
    }
}

// ---------------------------------------------------------------------------
// URL redaction
// ---------------------------------------------------------------------------

/// Redact credentials from a URL string.
///
/// Replaces `user:password@` with `***@` while preserving the scheme,
/// host, port, and path for diagnostic purposes. If the URL has no
/// credentials, it is returned unchanged.
///
/// This is a best-effort parser that handles common URL formats without
/// requiring a full URL parser dependency. It works on:
/// - `rtsp://user:pass@host:port/path`
/// - `rtsps://user:pass@host/path`
/// - `http://user:pass@host/path`
/// - URLs without credentials (returned as-is)
///
/// # Examples
///
/// ```
/// use nv_core::security::redact_url;
///
/// assert_eq!(
///     redact_url("rtsp://admin:secret@192.168.1.1:554/stream"),
///     "rtsp://***@192.168.1.1:554/stream"
/// );
/// assert_eq!(
///     redact_url("rtsp://192.168.1.1/stream"),
///     "rtsp://192.168.1.1/stream"
/// );
/// ```
pub fn redact_url(url: &str) -> String {
    // Find "://" to locate the authority section.
    let Some(scheme_end) = url.find("://") else {
        // No scheme — might still have credentials (unlikely but safe).
        return redact_authority(url);
    };
    let authority_start = scheme_end + 3;
    let rest = &url[authority_start..];

    // Find the '@' that separates userinfo from host.
    // Only look before the first '/' (path start) to avoid matching '@'
    // in path/query components.
    let path_start = rest.find('/').unwrap_or(rest.len());
    let authority_section = &rest[..path_start];

    if let Some(at_pos) = authority_section.rfind('@') {
        // Has credentials — redact everything before '@'.
        let after_at = &rest[at_pos..]; // includes '@' and the rest
        format!("{}://***{}", &url[..scheme_end], after_at)
    } else {
        // No credentials — return as-is.
        url.to_string()
    }
}

/// Redact credentials in a string that has no scheme prefix.
fn redact_authority(s: &str) -> String {
    let path_start = s.find('/').unwrap_or(s.len());
    let authority = &s[..path_start];
    if let Some(at_pos) = authority.rfind('@') {
        format!("***{}", &s[at_pos..])
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Error string sanitization
// ---------------------------------------------------------------------------

/// Maximum length for sanitized error strings.
const MAX_ERROR_LEN: usize = 512;

/// Sanitize an untrusted error/debug string from a backend.
///
/// - Strips control characters (except space) and bare newlines.
/// - Caps length at 512 characters.
/// - Redacts patterns resembling secrets (`password=...`, `token=...`,
///   `key=...`, `secret=...`, `auth=...`).
///
/// # Examples
///
/// ```
/// use nv_core::security::sanitize_error_string;
///
/// let dirty = "error: connection failed\n\tat rtspsrc password=hunter2";
/// let clean = sanitize_error_string(dirty);
/// assert!(!clean.contains("hunter2"));
/// assert!(!clean.contains('\n'));
/// ```
pub fn sanitize_error_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len().min(MAX_ERROR_LEN));

    for ch in s.chars() {
        if out.len() >= MAX_ERROR_LEN {
            out.push_str("...[truncated]");
            break;
        }
        // Allow printable characters and space; strip control chars.
        if ch == ' ' || (!ch.is_control() && !ch.is_ascii_control()) {
            out.push(ch);
        } else {
            out.push(' ');
        }
    }

    redact_secret_patterns(&mut out);
    out
}

/// Redact common secret-like patterns: `key=value` where key is a
/// known sensitive token name. Replaces the value with `***`.
fn redact_secret_patterns(s: &mut String) {
    let patterns = [
        "password=",
        "passwd=",
        "token=",
        "secret=",
        "key=",
        "auth=",
        "authorization:",
        "bearer ",
    ];

    for pat in &patterns {
        let mut search_from = 0;
        loop {
            // Recompute lowercase on every iteration so indexes are always
            // consistent with the current contents of `s`.
            let lower = s.to_lowercase();
            if search_from >= lower.len() {
                break;
            }
            let Some(rel_idx) = lower[search_from..].find(pat) else {
                break;
            };
            let abs_idx = search_from + rel_idx;
            let value_start = abs_idx + pat.len();
            // Find end of value: next space, '&', ';', ',', or end of string.
            let value_end = s[value_start..]
                .find([' ', '&', ';', ',', '\'', '"'])
                .map(|p| value_start + p)
                .unwrap_or(s.len());

            if value_end > value_start {
                s.replace_range(value_start..value_end, "***");
                search_from = value_start + 3;
            } else {
                search_from = value_start;
            }
        }
    }
}

/// Apply [`redact_url`] to all URL-like substrings in a string.
///
/// Scans for `scheme://...` patterns and redacts credentials in each.
/// Useful for sanitizing error messages that may embed URLs.
pub fn redact_urls_in_string(s: &str) -> String {
    let mut result = s.to_string();
    // Find URL-like patterns and redact them inline.
    for scheme in &["rtsp://", "rtsps://", "http://", "https://"] {
        let mut search_from = 0;
        while let Some(offset) = result[search_from..].find(scheme) {
            let start = search_from + offset;
            // Find end of URL: next space or end of string.
            let url_end = result[start..]
                .find(|c: char| c.is_whitespace() || c == '\'' || c == '"' || c == '>' || c == ')')
                .map(|p| start + p)
                .unwrap_or(result.len());
            let url = &result[start..url_end];
            let redacted = redact_url(url);
            let redacted_len = redacted.len();
            result.replace_range(start..url_end, &redacted);
            // Advance past the replacement to avoid re-matching the same scheme.
            search_from = start + redacted_len;
        }
    }
    result
}

/// Apply URL scheme promotion for RTSP sources under [`RtspSecurityPolicy::PreferTls`].
///
/// If the URL starts with `rtsp://`, returns a copy with `rtsps://`.
/// If the URL already starts with `rtsps://`, returns it unchanged.
/// If the URL has no recognized scheme, prepends `rtsps://`.
///
/// # Examples
///
/// ```
/// use nv_core::security::promote_rtsp_to_tls;
///
/// assert_eq!(promote_rtsp_to_tls("rtsp://cam/stream"), "rtsps://cam/stream");
/// assert_eq!(promote_rtsp_to_tls("rtsps://cam/stream"), "rtsps://cam/stream");
/// ```
pub fn promote_rtsp_to_tls(url: &str) -> String {
    if url.starts_with("rtsps://") {
        url.to_string()
    } else if let Some(rest) = url.strip_prefix("rtsp://") {
        format!("rtsps://{rest}")
    } else {
        // No recognized scheme — assume RTSP and add TLS scheme.
        format!("rtsps://{url}")
    }
}

/// Check whether an RTSP URL uses insecure (non-TLS) transport.
pub fn is_insecure_rtsp(url: &str) -> bool {
    url.starts_with("rtsp://")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RtspSecurityPolicy --

    #[test]
    fn security_policy_default_is_prefer_tls() {
        assert_eq!(RtspSecurityPolicy::default(), RtspSecurityPolicy::PreferTls);
    }

    #[test]
    fn security_policy_display() {
        assert_eq!(RtspSecurityPolicy::PreferTls.to_string(), "PreferTls");
        assert_eq!(
            RtspSecurityPolicy::AllowInsecure.to_string(),
            "AllowInsecure"
        );
        assert_eq!(RtspSecurityPolicy::RequireTls.to_string(), "RequireTls");
    }

    // -- CustomPipelinePolicy --

    #[test]
    fn custom_pipeline_policy_default_is_reject() {
        assert_eq!(
            CustomPipelinePolicy::default(),
            CustomPipelinePolicy::Reject
        );
    }

    // -- URL redaction --

    #[test]
    fn redact_url_with_credentials() {
        assert_eq!(
            redact_url("rtsp://admin:secret@192.168.1.1:554/stream"),
            "rtsp://***@192.168.1.1:554/stream"
        );
    }

    #[test]
    fn redact_url_without_credentials() {
        assert_eq!(
            redact_url("rtsp://192.168.1.1:554/stream"),
            "rtsp://192.168.1.1:554/stream"
        );
    }

    #[test]
    fn redact_url_rtsps_with_credentials() {
        assert_eq!(
            redact_url("rtsps://user:p%40ss@cam.example.com/live"),
            "rtsps://***@cam.example.com/live"
        );
    }

    #[test]
    fn redact_url_no_scheme() {
        assert_eq!(redact_url("user:pass@host/path"), "***@host/path");
    }

    #[test]
    fn redact_url_empty() {
        assert_eq!(redact_url(""), "");
    }

    #[test]
    fn redact_url_user_only_no_password() {
        // user@ without colon — still redacted (could be a token).
        assert_eq!(
            redact_url("rtsp://tokenuser@host/path"),
            "rtsp://***@host/path"
        );
    }

    #[test]
    fn redact_url_at_in_path_ignored() {
        // '@' in the path (after first '/') should not trigger redaction.
        assert_eq!(
            redact_url("rtsp://host/path@weird"),
            "rtsp://host/path@weird"
        );
    }

    // -- Error sanitization --

    #[test]
    fn sanitize_strips_control_chars() {
        let dirty = "error\x00\x07\ndetail\r\ntab\there";
        let clean = sanitize_error_string(dirty);
        assert!(!clean.contains('\x00'));
        assert!(!clean.contains('\x07'));
        assert!(!clean.contains('\n'));
        assert!(!clean.contains('\r'));
    }

    #[test]
    fn sanitize_truncates_long_strings() {
        let long = "a".repeat(1000);
        let clean = sanitize_error_string(&long);
        assert!(clean.len() < 600); // 512 + "[truncated]"
    }

    #[test]
    fn sanitize_redacts_password_pattern() {
        let s = "connection failed password=hunter2 at host";
        let clean = sanitize_error_string(s);
        assert!(!clean.contains("hunter2"));
        assert!(clean.contains("password=***"));
    }

    #[test]
    fn sanitize_redacts_token_pattern() {
        let s = "error token=abc123secret detail";
        let clean = sanitize_error_string(s);
        assert!(!clean.contains("abc123secret"));
        assert!(clean.contains("token=***"));
    }

    #[test]
    fn sanitize_preserves_useful_context() {
        let s = "connection refused: host 192.168.1.1 port 554";
        let clean = sanitize_error_string(s);
        assert_eq!(clean, s);
    }

    // -- promote_rtsp_to_tls --

    #[test]
    fn promote_rtsp_upgrades_to_rtsps() {
        assert_eq!(
            promote_rtsp_to_tls("rtsp://cam/stream"),
            "rtsps://cam/stream"
        );
    }

    #[test]
    fn promote_rtsp_keeps_rtsps() {
        assert_eq!(
            promote_rtsp_to_tls("rtsps://cam/stream"),
            "rtsps://cam/stream"
        );
    }

    #[test]
    fn promote_rtsp_no_scheme() {
        assert_eq!(promote_rtsp_to_tls("cam/stream"), "rtsps://cam/stream");
    }

    // -- is_insecure_rtsp --

    #[test]
    fn insecure_rtsp_detection() {
        assert!(is_insecure_rtsp("rtsp://host/path"));
        assert!(!is_insecure_rtsp("rtsps://host/path"));
        assert!(!is_insecure_rtsp("http://host/path"));
    }

    // -- redact_urls_in_string --

    #[test]
    fn redact_urls_in_error_string() {
        let s = "failed to connect to rtsp://admin:pass@cam/stream reason timeout";
        let clean = redact_urls_in_string(s);
        assert!(!clean.contains("admin:pass"));
        assert!(clean.contains("rtsp://***@cam/stream"));
    }

    #[test]
    fn redact_urls_no_urls() {
        let s = "plain error message";
        assert_eq!(redact_urls_in_string(s), s);
    }

    // -- redact_secret_patterns: multiple/repeated/mixed --

    #[test]
    fn redact_multiple_secrets_in_one_string() {
        let s = "password=abc token=xyz secret=qqq";
        let clean = sanitize_error_string(s);
        assert!(!clean.contains("abc"));
        assert!(!clean.contains("xyz"));
        assert!(!clean.contains("qqq"));
        assert!(clean.contains("password=***"));
        assert!(clean.contains("token=***"));
        assert!(clean.contains("secret=***"));
    }

    #[test]
    fn redact_repeated_same_key() {
        let s = "token=first&token=second&token=third";
        let clean = sanitize_error_string(s);
        assert!(!clean.contains("first"));
        assert!(!clean.contains("second"));
        assert!(!clean.contains("third"));
        // All three occurrences redacted.
        assert_eq!(clean.matches("token=***").count(), 3);
    }

    #[test]
    fn redact_mixed_delimiters() {
        let s = "password=a1 token=b2&secret=c3;auth=d4,key=e5'passwd=f6\"bearer g7";
        let clean = sanitize_error_string(s);
        for secret in &["a1", "b2", "c3", "d4", "e5", "f6", "g7"] {
            assert!(!clean.contains(secret), "secret {secret} leaked");
        }
    }

    #[test]
    fn redact_no_panic_on_adversarial_strings() {
        // Empty value
        let _ = sanitize_error_string("password= next");
        // Pattern at end of string with no value
        let _ = sanitize_error_string("password=");
        // Overlapping pattern-like text
        let _ = sanitize_error_string("password=password=nested");
        // Only delimiters after key
        let _ = sanitize_error_string("token=&&&");
        // Very long value
        let long_val = format!("secret={}", "x".repeat(2000));
        let clean = sanitize_error_string(&long_val);
        assert!(!clean.contains(&"x".repeat(100)));
        // Unicode content
        let _ = sanitize_error_string("token=日本語テスト done");
        // Repeated pattern with no value between
        let _ = sanitize_error_string("key=key=key=");
    }

    #[test]
    fn redact_case_insensitive() {
        let s = "PASSWORD=upper Token=Mixed SECRET=LOUD";
        let clean = sanitize_error_string(s);
        assert!(!clean.contains("upper"));
        assert!(!clean.contains("Mixed"));
        assert!(!clean.contains("LOUD"));
    }
}
