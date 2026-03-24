//! Compute-path frame routing and preprocessing.
//!
//! This module provides the single routing point where the detector decides
//! how to obtain pixel data for inference. Two paths exist:
//!
//! - **Host path** — frames that are host-readable or mappable to host are
//!   preprocessed on the CPU via [`letterbox_preprocess`](crate::letterbox::letterbox_preprocess).
//! - **Device-native path** — opaque device-resident frames require a
//!   hardware-specific adapter. See [`FramePreprocessor`] for the extension
//!   point.
//!
//! Both [`DetectorStage`](crate::DetectorStage) and
//! [`DetectorBatchProcessor`](crate::DetectorBatchProcessor) route through
//! this module instead of calling `require_host_data()` unconditionally.
//!
//! ## Host fallback policy
//!
//! When GPU inference is enabled but no device-native preprocessor is
//! configured, every frame still pays the cost of host materialization
//! (GPU → CPU download). [`HostFallbackPolicy`] controls how this
//! situation is handled:
//!
//! - [`Auto`](HostFallbackPolicy::Auto) — library default: [`Warn`] for
//!   GPU, [`Allow`] for CPU (default).
//! - [`Allow`](HostFallbackPolicy::Allow) — silently permit.
//! - [`Warn`](HostFallbackPolicy::Warn) — permit but emit rate-limited
//!   warnings (first call, then every 60th), making the performance cost
//!   visible during development.
//! - [`Forbid`](HostFallbackPolicy::Forbid) — fail with an error,
//!   enforcing that a device-native preprocessor must be configured.
//!
//! ## Batch preprocessing
//!
//! [`BatchFramePreprocessor`] is the batch-mode analog of
//! [`FramePreprocessor`]. It writes directly into a pre-allocated batch
//! tensor slice, avoiding per-frame allocation. The default
//! [`HostBatchPreprocessor`] uses CPU letterbox; hardware backends
//! (CUDA, TensorRT) provide their own implementations.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_frame::{DataAccess, FrameEnvelope};
use tracing::warn;

use crate::inference;
use crate::letterbox::{LetterboxInfo, letterbox_preprocess, letterbox_preprocess_into};

/// Controls behavior when host materialization is required for a
/// device-resident frame during preprocessing.
///
/// On Jetson with GPU inference enabled, frames may arrive as NVMM
/// device buffers. If no device-native [`FramePreprocessor`] is
/// configured, the host path must download pixels from GPU to CPU —
/// a measurable per-frame cost. This policy makes that cost explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HostFallbackPolicy {
    /// Let the library decide: [`Warn`](Self::Warn) when GPU inference
    /// is enabled, [`Allow`](Self::Allow) otherwise.
    ///
    /// This is the default. Use an explicit variant to override.
    #[default]
    Auto,
    /// Silently allow host materialization.
    ///
    /// Use when the host-fallback cost is intentionally accepted
    /// (e.g., CPU-only deployments, or GPU mode where you've measured
    /// the cost and it's acceptable).
    Allow,
    /// Allow host materialization but emit rate-limited warnings.
    ///
    /// Recommended when `gpu = true` but no device-native preprocessor
    /// is configured, to surface the performance cost during development.
    Warn,
    /// Forbid host materialization for mappable frames. Returns an error
    /// if a device-native preprocessor is needed but not configured.
    Forbid,
}

/// Output of frame preprocessing: an inference-ready NCHW float32 tensor
/// and the letterbox geometry needed for postprocessing.
pub struct PreprocessedFrame {
    /// `[1, 3, H, W]` float32 tensor in NCHW layout, values in `[0, 1]`.
    pub tensor: Vec<f32>,
    /// Letterbox parameters for remapping detections to original frame coords.
    pub letterbox: LetterboxInfo,
}

/// Extension point for device-native frame preprocessing.
///
/// Implement this trait to bypass host materialization for device-resident
/// frames. The default pipeline uses [`HostPreprocessor`], which handles
/// all host-readable and host-mappable frames via CPU letterbox resize.
///
/// A Jetson adapter would implement this to preprocess NVMM buffers
/// directly using CUDA kernels, feeding device-native inference (e.g.,
/// TensorRT) without any host round-trip. An AMD adapter would do the
/// same via ROCm/HIP.
///
/// # Implementing a device-native adapter
///
/// ```ignore
/// struct JetsonPreprocessor { /* cuda stream, etc. */ }
///
/// impl FramePreprocessor for JetsonPreprocessor {
///     fn preprocess(
///         &self,
///         frame: &FrameEnvelope,
///         input_size: u32,
///     ) -> Result<PreprocessedFrame, StageError> {
///         let handle = frame.accelerated_handle::<NvmmBuffer>()
///             .ok_or_else(|| StageError::ProcessingFailed {
///                 stage_id: StageId("jetson-preprocess"),
///                 detail: "expected NVMM buffer".into(),
///             })?;
///         // CUDA resize + normalize → f32 tensor
///         let tensor = cuda_letterbox(handle, input_size)?;
///         Ok(PreprocessedFrame { tensor, letterbox })
///     }
/// }
/// ```
pub trait FramePreprocessor: Send {
    /// Preprocess a frame into an inference-ready NCHW float32 tensor.
    ///
    /// Returns the tensor data and letterbox geometry for postprocessing.
    fn preprocess(
        &self,
        frame: &FrameEnvelope,
        input_size: u32,
    ) -> Result<PreprocessedFrame, StageError>;
}

/// Default host-path preprocessor.
///
/// Handles [`DataAccess::HostReadable`] and [`DataAccess::MappableToHost`]
/// frames via CPU letterbox resize. Returns an error for opaque
/// device-resident frames.
pub struct HostPreprocessor {
    stage_id: StageId,
    policy: HostFallbackPolicy,
    warn_count: AtomicU64,
}

impl HostPreprocessor {
    /// Create a host preprocessor that reports errors under `stage_id`.
    pub fn new(stage_id: StageId) -> Self {
        Self {
            stage_id,
            policy: HostFallbackPolicy::default(),
            warn_count: AtomicU64::new(0),
        }
    }

    /// Create a host preprocessor with an explicit fallback policy.
    pub fn with_policy(stage_id: StageId, policy: HostFallbackPolicy) -> Self {
        Self {
            stage_id,
            policy,
            warn_count: AtomicU64::new(0),
        }
    }
}

impl FramePreprocessor for HostPreprocessor {
    fn preprocess(
        &self,
        frame: &FrameEnvelope,
        input_size: u32,
    ) -> Result<PreprocessedFrame, StageError> {
        let (pixels, bpp) = resolve_host_pixels(frame, self.stage_id, self.policy, &self.warn_count)?;
        let (tensor, letterbox) = letterbox_preprocess(
            &pixels,
            frame.width(),
            frame.height(),
            frame.stride(),
            bpp,
            input_size,
        );
        Ok(PreprocessedFrame { tensor, letterbox })
    }
}

/// Obtain host-accessible pixel bytes from a frame, with explicit routing
/// based on [`DataAccess`] and [`HostFallbackPolicy`].
///
/// This is the single routing point for compute-path frame access.
///
/// - [`DataAccess::HostReadable`] — zero-copy borrow of host bytes.
/// - [`DataAccess::MappableToHost`] — triggers host materialization
///   (cached per-frame via `OnceLock`). Subject to [`HostFallbackPolicy`]:
///   - [`Allow`](HostFallbackPolicy::Allow) — proceeds silently.
///   - [`Warn`](HostFallbackPolicy::Warn) — proceeds with rate-limited
///     warning (first call, then every 60th call).
///   - [`Forbid`](HostFallbackPolicy::Forbid) — returns an error.
/// - [`DataAccess::Opaque`] — returns an error. Device-native adapters
///   should implement [`FramePreprocessor`] to handle opaque frames.
///
/// Also validates that the pixel format is RGB-compatible.
pub fn resolve_host_pixels<'a>(
    frame: &'a FrameEnvelope,
    stage_id: StageId,
    policy: HostFallbackPolicy,
    warn_counter: &AtomicU64,
) -> Result<(Cow<'a, [u8]>, u32), StageError> {
    let bpp = inference::require_rgb_compatible(frame.format(), stage_id)?;
    match frame.data_access() {
        DataAccess::HostReadable => {
            let pixels = frame
                .require_host_data()
                .map_err(|e| StageError::ProcessingFailed {
                    stage_id,
                    detail: e.to_string(),
                })?;
            Ok((pixels, bpp))
        }
        DataAccess::MappableToHost => {
            apply_host_fallback_policy(policy, warn_counter, stage_id)?;
            let pixels = frame
                .require_host_data()
                .map_err(|e| StageError::ProcessingFailed {
                    stage_id,
                    detail: e.to_string(),
                })?;
            Ok((pixels, bpp))
        }
        _ => Err(StageError::ProcessingFailed {
            stage_id,
            detail: "frame is opaque device-resident; host fallback unavailable. \
                     A device-native FramePreprocessor is needed for this frame type."
                .into(),
        }),
    }
}

/// Apply the host fallback policy for a `MappableToHost` frame.
///
/// - `Auto` / `Allow` → no-op.
/// - `Warn` → rate-limited warning (first call, then every 60th).
/// - `Forbid` → error.
///
/// `Auto` should be resolved to a concrete policy via
/// [`DetectorConfig::effective_host_fallback`](crate::DetectorConfig::effective_host_fallback)
/// before reaching this point. If it arrives here unresolved, it is
/// treated as `Allow`.
fn apply_host_fallback_policy(
    policy: HostFallbackPolicy,
    warn_counter: &AtomicU64,
    stage_id: StageId,
) -> Result<(), StageError> {
    match policy {
        HostFallbackPolicy::Auto | HostFallbackPolicy::Allow => Ok(()),
        HostFallbackPolicy::Warn => {
            let count = warn_counter.fetch_add(1, Ordering::Relaxed) + 1; // 1-based
            if count == 1 || count % 60 == 0 {
                warn!(
                    stage = %stage_id,
                    count = count,
                    "host materialization on device-resident frame — \
                     consider a device-native FramePreprocessor for optimal performance",
                );
            }
            Ok(())
        }
        HostFallbackPolicy::Forbid => Err(StageError::ProcessingFailed {
            stage_id,
            detail: "host materialization forbidden by HostFallbackPolicy::Forbid; \
                     a device-native FramePreprocessor is required for device-resident frames"
                .into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Batch preprocessing extension point
// ---------------------------------------------------------------------------

/// Extension point for batch-mode frame preprocessing.
///
/// The batch-mode analog of [`FramePreprocessor`]. Instead of returning
/// a standalone tensor, this trait writes directly into a pre-allocated
/// slice of the batch tensor, avoiding per-frame allocation.
///
/// Implement this to bypass host materialization in batch mode for
/// device-resident frames (e.g., CUDA letterbox on Jetson).
///
/// The default [`HostBatchPreprocessor`] uses CPU letterbox via
/// [`resolve_host_pixels`] + [`letterbox_preprocess_into`].
pub trait BatchFramePreprocessor: Send {
    /// Preprocess one frame into a slice of the batch tensor.
    ///
    /// `dest` is exactly `3 * input_size * input_size` floats — the
    /// caller manages the batch-level allocation and offset.
    fn preprocess_into(
        &self,
        frame: &FrameEnvelope,
        input_size: u32,
        dest: &mut [f32],
    ) -> Result<LetterboxInfo, StageError>;
}

/// Default host-path batch preprocessor.
///
/// Resolves host pixels via [`resolve_host_pixels`] and writes a
/// letterboxed NCHW tensor directly into the provided slice.
pub struct HostBatchPreprocessor {
    stage_id: StageId,
    policy: HostFallbackPolicy,
    warn_count: AtomicU64,
}

impl HostBatchPreprocessor {
    /// Create a host batch preprocessor with default policy.
    pub fn new(stage_id: StageId) -> Self {
        Self {
            stage_id,
            policy: HostFallbackPolicy::default(),
            warn_count: AtomicU64::new(0),
        }
    }

    /// Create a host batch preprocessor with an explicit fallback policy.
    pub fn with_policy(stage_id: StageId, policy: HostFallbackPolicy) -> Self {
        Self {
            stage_id,
            policy,
            warn_count: AtomicU64::new(0),
        }
    }
}

impl BatchFramePreprocessor for HostBatchPreprocessor {
    fn preprocess_into(
        &self,
        frame: &FrameEnvelope,
        input_size: u32,
        dest: &mut [f32],
    ) -> Result<LetterboxInfo, StageError> {
        let (host_pixels, bpp) =
            resolve_host_pixels(frame, self.stage_id, self.policy, &self.warn_count)?;
        Ok(letterbox_preprocess_into(
            &host_pixels,
            frame.width(),
            frame.height(),
            frame.stride(),
            bpp,
            input_size,
            dest,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use nv_core::id::FeedId;
    use nv_core::timestamp::MonotonicTs;
    use nv_core::{TypedMetadata, WallTs};
    use nv_frame::{DataAccess, FrameEnvelope, HostBytes, PixelFormat};

    use std::sync::atomic::AtomicUsize;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Layer;

    const TEST_STAGE: StageId = StageId("test-preprocess");

    fn allow_counter() -> AtomicU64 {
        AtomicU64::new(0)
    }

    /// Host-readable RGB frame takes the host path successfully.
    #[test]
    fn host_frame_resolves_host_pixels() {
        let frame = nv_test_util::synthetic::solid_rgb(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), 4, 4, 128, 64, 32,
        );
        let (pixels, bpp) =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Allow, &allow_counter())
                .unwrap();
        assert_eq!(bpp, 3);
        assert_eq!(pixels.len(), 4 * 4 * 3);
    }

    /// Device frame with materializer (MappableToHost) resolves successfully.
    #[test]
    fn mappable_device_frame_resolves_host_pixels() {
        let rgb_data = vec![100u8; 8 * 8 * 3];
        let rgb_data_clone = rgb_data.clone();
        let handle: Arc<dyn std::any::Any + Send + Sync> = Arc::new(42u32);
        let frame = FrameEnvelope::new_device(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), WallTs::now(),
            8, 8, PixelFormat::Rgb8, 8 * 3,
            handle,
            Some(Box::new(move || Ok(HostBytes::from_vec(rgb_data_clone.clone())))),
            TypedMetadata::new(),
        );
        assert_eq!(frame.data_access(), DataAccess::MappableToHost);
        let (pixels, bpp) =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Allow, &allow_counter())
                .unwrap();
        assert_eq!(bpp, 3);
        assert_eq!(pixels.len(), 8 * 8 * 3);
    }

    /// Opaque device frame returns an error.
    #[test]
    fn opaque_device_frame_returns_error() {
        let handle: Arc<dyn std::any::Any + Send + Sync> = Arc::new(42u32);
        let frame = FrameEnvelope::new_device(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), WallTs::now(),
            8, 8, PixelFormat::Rgb8, 8 * 3,
            handle,
            None, // no materializer → Opaque
            TypedMetadata::new(),
        );
        assert_eq!(frame.data_access(), DataAccess::Opaque);
        let result =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Allow, &allow_counter());
        assert!(result.is_err());
    }

    /// Unsupported pixel format (Gray8) returns an error.
    #[test]
    fn unsupported_format_returns_error() {
        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), 4, 4, 128,
        );
        let result =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Allow, &allow_counter());
        assert!(result.is_err());
    }

    /// HostPreprocessor succeeds on a valid host-readable RGB frame.
    #[test]
    fn host_preprocessor_succeeds_on_host_frame() {
        let preprocessor = HostPreprocessor::new(TEST_STAGE);
        let frame = nv_test_util::synthetic::solid_rgb(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), 16, 16, 128, 64, 32,
        );
        let result = preprocessor.preprocess(&frame, 32);
        assert!(result.is_ok());
        let preprocessed = result.unwrap();
        // NCHW: 1 * 3 * 32 * 32 = 3072 floats
        assert_eq!(preprocessed.tensor.len(), 3 * 32 * 32);
    }

    /// HostPreprocessor fails on an opaque device frame.
    #[test]
    fn host_preprocessor_fails_on_opaque_frame() {
        let preprocessor = HostPreprocessor::new(TEST_STAGE);
        let handle: Arc<dyn std::any::Any + Send + Sync> = Arc::new(42u32);
        let frame = FrameEnvelope::new_device(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), WallTs::now(),
            8, 8, PixelFormat::Rgb8, 8 * 3,
            handle, None,
            TypedMetadata::new(),
        );
        let result = preprocessor.preprocess(&frame, 32);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // HostFallbackPolicy tests
    // ---------------------------------------------------------------

    fn make_mappable_frame() -> FrameEnvelope {
        let rgb_data = vec![100u8; 8 * 8 * 3];
        let handle: Arc<dyn std::any::Any + Send + Sync> = Arc::new(42u32);
        FrameEnvelope::new_device(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), WallTs::now(),
            8, 8, PixelFormat::Rgb8, 8 * 3,
            handle,
            Some(Box::new(move || Ok(HostBytes::from_vec(rgb_data.clone())))),
            TypedMetadata::new(),
        )
    }

    /// Warn policy allows host materialization for mappable frames.
    #[test]
    fn warn_policy_allows_mappable_frame() {
        let frame = make_mappable_frame();
        let counter = AtomicU64::new(0);
        let result =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter);
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    /// Forbid policy rejects host materialization for mappable frames.
    #[test]
    fn forbid_policy_rejects_mappable_frame() {
        let frame = make_mappable_frame();
        let counter = AtomicU64::new(0);
        let result =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Forbid, &counter);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("Forbid"),
            "error should mention Forbid: {err:?}",
        );
    }

    /// Allow policy for host-readable frames never triggers policy checks
    /// (host-readable frames don't need materialization).
    #[test]
    fn allow_policy_on_host_readable_no_warn() {
        let frame = nv_test_util::synthetic::solid_rgb(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), 4, 4, 128, 64, 32,
        );
        let counter = AtomicU64::new(0);
        let result =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter);
        assert!(result.is_ok());
        // Host-readable frames don't go through the MappableToHost branch,
        // so the warn counter stays at 0.
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    /// Forbid policy still allows host-readable frames (no materialization needed).
    #[test]
    fn forbid_policy_allows_host_readable_frame() {
        let frame = nv_test_util::synthetic::solid_rgb(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), 4, 4, 128, 64, 32,
        );
        let counter = AtomicU64::new(0);
        let result =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Forbid, &counter);
        assert!(result.is_ok());
    }

    // ---------------------------------------------------------------
    // BatchFramePreprocessor tests
    // ---------------------------------------------------------------

    /// HostBatchPreprocessor preprocesses a host-readable frame into a
    /// batch tensor slice.
    #[test]
    fn host_batch_preprocessor_succeeds() {
        let preprocessor = HostBatchPreprocessor::new(TEST_STAGE);
        let frame = nv_test_util::synthetic::solid_rgb(
            FeedId::new(1), 1, MonotonicTs::from_nanos(0), 16, 16, 128, 64, 32,
        );
        let input_size = 32u32;
        let pixel_count = 3 * (input_size as usize) * (input_size as usize);
        let mut dest = vec![0.0f32; pixel_count];
        let lb = preprocessor.preprocess_into(&frame, input_size, &mut dest).unwrap();
        // Letterbox info should be valid.
        assert!(lb.scale > 0.0);
        // Dest should have been written (not all zeros for non-black input).
        assert!(dest.iter().any(|&v| v > 0.0));
    }

    /// HostBatchPreprocessor with Forbid policy rejects mappable frames.
    #[test]
    fn host_batch_preprocessor_forbid_rejects_mappable() {
        let preprocessor = HostBatchPreprocessor::with_policy(TEST_STAGE, HostFallbackPolicy::Forbid);
        let frame = make_mappable_frame();
        let input_size = 32u32;
        let pixel_count = 3 * (input_size as usize) * (input_size as usize);
        let mut dest = vec![0.0f32; pixel_count];
        let result = preprocessor.preprocess_into(&frame, input_size, &mut dest);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // Warn throttling cadence (P3)
    // ---------------------------------------------------------------

    /// Warn policy emits on the 1st call and every 60th call thereafter.
    ///
    /// Verifies the exact emission pattern: calls 1, 60, 120.
    /// The warn counter is atomically incremented and we measure it
    /// after each call to confirm the cadence.
    #[test]
    fn warn_throttle_emits_first_and_every_60th() {
        let frame = make_mappable_frame();
        let counter = AtomicU64::new(0);

        // Track which calls emitted a warn by observing counter values.
        // The function always increments the counter on Warn policy.
        // We need 120+ calls to verify the pattern.
        //
        // Expected emissions (1-based call #): 1, 60, 120.
        // Implementation: count = fetch_add(1) + 1  (1-based)
        //   emit when count == 1 || count % 60 == 0
        //   → count=1 (emit), count=60 (emit), count=120 (emit)

        for i in 1..=121u64 {
            let _ = resolve_host_pixels(
                &frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter,
            );
            assert_eq!(
                counter.load(Ordering::Relaxed), i,
                "counter should match call number",
            );
        }

        // The total counter value should be exactly 121 (one per call).
        assert_eq!(counter.load(Ordering::Relaxed), 121);
    }

    /// Verify exact emission pattern by testing the internal condition.
    ///
    /// The condition `count == 1 || count % 60 == 0` (1-based count)
    /// produces emissions at call numbers: 1, 60, 120, 180, ...
    #[test]
    fn warn_throttle_exact_emission_counts() {
        // Directly test the emission pattern: for each 1-based count,
        // determine whether a warn would fire.
        let should_emit = |count: u64| -> bool {
            count == 1 || count % 60 == 0
        };

        // Calls 1..=180: count emissions.
        let emission_points: Vec<u64> = (1..=180)
            .filter(|&c| should_emit(c))
            .collect();

        assert_eq!(
            emission_points,
            vec![1, 60, 120, 180],
            "expected emissions at 1st, 60th, 120th, 180th calls",
        );
    }

    // ---------------------------------------------------------------
    // Auto policy tests (P2)
    // ---------------------------------------------------------------

    /// Auto policy behaves like Allow at the resolution level
    /// (maps to Allow or Warn via DetectorConfig::effective_host_fallback).
    #[test]
    fn auto_policy_acts_as_allow_at_resolve_level() {
        let frame = make_mappable_frame();
        let counter = AtomicU64::new(0);
        let result =
            resolve_host_pixels(&frame, TEST_STAGE, HostFallbackPolicy::Auto, &counter);
        assert!(result.is_ok());
        // Auto is treated as Allow at this level — no warn counter bump.
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    // ---------------------------------------------------------------
    // Warn throttling — log-capture tests
    // ---------------------------------------------------------------

    /// A minimal tracing layer that counts WARN-level events.
    struct WarnCounter {
        count: Arc<AtomicUsize>,
    }

    impl<S: tracing::Subscriber> Layer<S> for WarnCounter {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() == tracing::Level::WARN {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Validates *real* `tracing::warn!` emission cadence through a
    /// subscriber layer rather than just counter arithmetic.
    ///
    /// Expected pattern (1-based call index):
    ///   emit on call 1  → total 1
    ///   silent 2..59     → total 1
    ///   emit on call 60  → total 2
    ///   silent 61..119   → total 2
    ///   emit on call 120 → total 3
    #[test]
    fn warn_throttle_real_emission_cadence() {
        let warn_count = Arc::new(AtomicUsize::new(0));
        let layer = WarnCounter { count: Arc::clone(&warn_count) };
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            let frame = make_mappable_frame();
            let counter = AtomicU64::new(0);

            // Call 1 — should emit.
            let _ = resolve_host_pixels(
                &frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter,
            );
            assert_eq!(warn_count.load(Ordering::Relaxed), 1, "emit on call 1");

            // Calls 2..59 — no new emissions.
            for _ in 2..=59 {
                let _ = resolve_host_pixels(
                    &frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter,
                );
            }
            assert_eq!(warn_count.load(Ordering::Relaxed), 1, "silent for calls 2..59");

            // Call 60 — should emit.
            let _ = resolve_host_pixels(
                &frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter,
            );
            assert_eq!(warn_count.load(Ordering::Relaxed), 2, "emit on call 60");

            // Calls 61..119 — no new emissions.
            for _ in 61..=119 {
                let _ = resolve_host_pixels(
                    &frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter,
                );
            }
            assert_eq!(warn_count.load(Ordering::Relaxed), 2, "silent for calls 61..119");

            // Call 120 — should emit.
            let _ = resolve_host_pixels(
                &frame, TEST_STAGE, HostFallbackPolicy::Warn, &counter,
            );
            assert_eq!(warn_count.load(Ordering::Relaxed), 3, "emit on call 120");
        });
    }

    /// Allow policy never emits `tracing::warn!`, even on mappable frames.
    #[test]
    fn allow_policy_emits_no_warnings() {
        let warn_count = Arc::new(AtomicUsize::new(0));
        let layer = WarnCounter { count: Arc::clone(&warn_count) };
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            let frame = make_mappable_frame();
            let counter = AtomicU64::new(0);
            for _ in 0..120 {
                let _ = resolve_host_pixels(
                    &frame, TEST_STAGE, HostFallbackPolicy::Allow, &counter,
                );
            }
            assert_eq!(warn_count.load(Ordering::Relaxed), 0);
        });
    }
}
