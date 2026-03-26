//! Sample detection + tracking pipeline.
//!
//! Demonstrates how to wire detection and tracking stages into the
//! NextVision runtime using only public APIs, with a real-time
//! multi-panel UI showing video feeds, detection overlays, and telemetry.
//!
//! Supports two modes:
//! - **Per-feed** (default): each feed runs its own detector `Stage`.
//! - **Batched** (`--batch`): all feeds share a single detector
//!   `BatchProcessor`; detection runs as one batched inference call.
//!
//! Multiple `--video-uri` values spawn multiple feeds. All feeds display
//! in an auto-grid layout with a telemetry dashboard across the top.
//!
//! Run:
//! ```sh
//! # Single feed (per-feed stage):
//! cargo run -p nv-sample-app -- \
//!   --video-uri rtsp://camera/stream
//!
//! # Multi-feed batched:
//! cargo run -p nv-sample-app -- \
//!   --video-uri rtsp://cam1/stream \
//!   --video-uri rtsp://cam2/stream \
//!   --batch --batch-size 4
//!
//! # Headless (no GUI, log-only):
//! cargo run -p nv-sample-app -- \
//!   --video-uri rtsp://cam1/stream --headless
//! ```

mod cli;
mod mode;
mod overlay;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use clap::Parser;
use tracing::{error, info, warn};

use nv_core::{CameraMode, SourceSpec};
use nv_metrics::MetricsExporter;
use nv_runtime::{
    BackpressurePolicy, BatchConfig, DeviceResidency, FeedConfig, FeedPipeline, FrameInclusion,
    HealthEvent, OutputEnvelope, OutputSink, Runtime,
};
use nv_sample_detection::{DetectorBatchProcessor, DetectorConfig, DetectorStage};
use nv_sample_tracking::{TrackerConfig, TrackerStage};

use crate::cli::Cli;
use crate::ui::{UiSink, new_shared_state};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Cli::parse();

    if let Err(e) = run(args) {
        error!("fatal: {e}");
        std::process::exit(1);
    }
}

/// Parse a video URI into a [`SourceSpec`].
fn parse_source(uri: &str, loop_file: bool) -> SourceSpec {
    if uri.starts_with("rtsp://") || uri.starts_with("rtsps://") {
        SourceSpec::rtsp(uri)
    } else if loop_file {
        SourceSpec::file_looping(PathBuf::from(uri))
    } else {
        SourceSpec::file(PathBuf::from(uri))
    }
}

/// Build the shared detector config from CLI args.
fn detector_config(args: &Cli) -> DetectorConfig {
    DetectorConfig {
        model_path: args.model.clone(),
        input_size: args.input_size,
        confidence_threshold: args.conf_threshold,
        gpu: args.gpu,
        ..Default::default()
    }
}

/// Build the shared tracker config from CLI args.
fn tracker_config(args: &Cli) -> TrackerConfig {
    TrackerConfig {
        max_age: args.max_age,
        min_hits: args.min_hits,
        iou_threshold: args.iou_threshold,
        ..Default::default()
    }
}

/// Create a batch processor, using NVMM-native preprocessing on Jetson
/// when the `jetson-nvmm` feature is active and GPU inference is enabled.
///
/// On Jetson, NVMM frames arrive as device-resident unified memory
/// buffers. The default host preprocessor would trigger a full GPU→CPU
/// DMA copy per frame, pinning NVMM surfaces and starving the hardware
/// decoder's output buffer pool. The NVMM preprocessor reads directly
/// from unified memory, eliminating that bottleneck.
fn create_batch_processor(args: &Cli) -> DetectorBatchProcessor {
    let config = detector_config(args);

    #[cfg(feature = "jetson-nvmm")]
    if args.gpu {
        info!("using NVMM-native batch preprocessor (zero-copy unified memory)");
        let policy = config.effective_host_fallback();
        let preprocessor = Box::new(crate::mode::NvmmBatchPreprocessor::new(
            nv_core::id::StageId("sample-detector-batch"),
            policy,
        ));
        return DetectorBatchProcessor::with_preprocessor(config, preprocessor);
    }

    DetectorBatchProcessor::new(config)
}

/// Create a detector stage, using NVMM-native preprocessing on Jetson
/// when the `jetson-nvmm` feature is active and GPU inference is enabled.
fn create_detector_stage(config: &DetectorConfig) -> DetectorStage {
    #[cfg(feature = "jetson-nvmm")]
    if config.gpu {
        let policy = config.effective_host_fallback();
        let preprocessor = Box::new(crate::mode::NvmmPreprocessor::new(
            nv_core::id::StageId("sample-detector"),
            policy,
        ));
        return DetectorStage::with_preprocessor(config.clone(), preprocessor);
    }

    DetectorStage::new(config.clone())
}

/// CUDA JIT compilation on first run can take 30+ seconds.
/// Allow a generous timeout for the initial inference.
const GPU_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// Default preview interval when neither `--preview-fps` nor
/// `--sample-interval` is specified (~5 fps at 30 fps source).
const DEFAULT_PREVIEW_INTERVAL: u32 = 6;

/// Resolve the effective [`FrameInclusion`] from CLI arguments.
///
/// Precedence: `--sample-interval` > `--preview-fps` > default.
///
/// When `--preview-fps` is used, the resulting policy is
/// [`FrameInclusion::TargetFps`] — the executor resolves the actual
/// interval from observed source cadence (frame timestamps) after a
/// brief warmup (~30 frames). The fallback interval assumes ~30 fps
/// during the warmup window so that preview begins immediately.
fn resolve_frame_inclusion(args: &Cli, include_frames: bool) -> FrameInclusion {
    resolve_frame_inclusion_params(args.sample_interval, args.preview_fps, include_frames)
}

/// Pure parameter-based inclusion resolution — testable without Cli.
///
/// Precedence: `sample_interval` > `preview_fps` > default.
/// `include_frames = false` forces [`FrameInclusion::Never`] (headless).
fn resolve_frame_inclusion_params(
    sample_interval: Option<u32>,
    preview_fps: Option<f32>,
    include_frames: bool,
) -> FrameInclusion {
    if !include_frames {
        return FrameInclusion::Never;
    }
    if let Some(interval) = sample_interval {
        return FrameInclusion::sampled(interval);
    }
    if let Some(fps) = preview_fps {
        // Adaptive: resolves from observed source cadence at runtime.
        // Fallback assumes ~30 fps during warmup.
        let fallback = (30.0_f32 / fps).round().max(1.0) as u32;
        return FrameInclusion::target_fps(fps, fallback);
    }
    FrameInclusion::sampled(DEFAULT_PREVIEW_INTERVAL)
}

/// Lightweight output sink that periodically logs detection/track summaries.
///
/// Used in headless mode for feeds that don't need a display.
struct LogSink {
    count: AtomicU64,
}

impl LogSink {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
        }
    }
}

impl OutputSink for LogSink {
    fn emit(&self, output: Arc<OutputEnvelope>) {
        let n = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(60) {
            info!(
                feed = %output.feed_id,
                seq = output.frame_seq,
                detections = output.detections.len(),
                tracks = output.tracks.len(),
                "pipeline running"
            );
        }
    }
}

fn run(args: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder().build()?;
    let runtime_handle = runtime.handle();

    // Metrics: spin up a tokio runtime for the OTel background exporter
    // only when --otlp-endpoint is provided.  No tokio overhead otherwise.
    let metrics_rt = args
        .otlp_endpoint
        .as_ref()
        .map(|_| tokio::runtime::Runtime::new())
        .transpose()?;
    let metrics = args
        .otlp_endpoint
        .as_ref()
        .map(|endpoint| {
            // Enter the tokio context only for the duration of build(),
            // then drop the guard so that block_on() in the shutdown path
            // is never called from within an entered runtime context.
            let _guard = metrics_rt.as_ref().unwrap().enter();
            MetricsExporter::builder()
                .runtime_handle(runtime_handle.clone())
                .otlp_endpoint(endpoint)
                .service_name("nv-sample-app")
                .build()
        })
        .transpose()?;

    // Feed handles — kept alive and passed to the UI.
    let feeds;

    // Optional batch handle (only in batch mode).
    let mut batch_handle: Option<nv_runtime::BatchHandle> = None;

    // Shared UI state (used when not headless).
    let ui_state = if !args.headless {
        Some(new_shared_state())
    } else {
        None
    };

    // Build a sink appropriate for the current run mode.
    let make_sink = |ui_state: &Option<ui::SharedUiState>| -> Box<dyn OutputSink> {
        if let Some(state) = ui_state {
            Box::new(UiSink::new(Arc::clone(state)))
        } else {
            Box::new(LogSink::new())
        }
    };

    /// Complete `FeedConfig` from a partially-built config builder with
    /// common settings (source, camera mode, output sink, backpressure).
    fn build_feed_config(
        builder: nv_runtime::FeedConfigBuilder,
        source: SourceSpec,
        sink: Box<dyn OutputSink>,
        queue_depth: usize,
        frame_inclusion: FrameInclusion,
        decode: nv_runtime::DecodePreference,
        device_residency: DeviceResidency,
    ) -> Result<FeedConfig, Box<dyn std::error::Error>> {
        let mut builder = builder
            .source(source)
            .camera_mode(CameraMode::Fixed)
            .output_sink(sink)
            .decode_preference(decode)
            .device_residency(device_residency)
            .backpressure(BackpressurePolicy::DropOldest { queue_depth })
            .frame_inclusion(frame_inclusion);

        // On Jetson (with or without the jetson-nvmm feature), hardware
        // decoders (nvv4l2decoder) output video/x-raw(memory:NVMM) — GPU-
        // mapped buffers that the standard videoconvert cannot accept.
        // When the provider is NOT active (Host/Cuda residency), insert
        // nvvidconv to bridge NVMM → system memory.
        //
        // When a Provider IS active, the pipeline builder skips this
        // hook entirely (provider controls the full decoder→tail path).
        // So this hook is always safe to set — it only fires for
        // non-provider paths that encounter NVMM decoder output.
        builder = builder.post_decode_hook(crate::mode::nvmm_bridge_hook());

        Ok(builder.build()?)
    }

    /// Register feeds from the URI list, returning handles.  The
    /// `make_builder` closure produces a mode-specific `FeedConfigBuilder`
    /// (batch-pipeline vs. per-feed stages) for each feed.
    fn register_feeds(
        runtime: &Runtime,
        args: &Cli,
        ui_state: &Option<ui::SharedUiState>,
        make_sink: &dyn Fn(&Option<ui::SharedUiState>) -> Box<dyn OutputSink>,
        make_builder: &dyn Fn()
            -> Result<nv_runtime::FeedConfigBuilder, Box<dyn std::error::Error>>,
    ) -> Result<Vec<nv_runtime::FeedHandle>, Box<dyn std::error::Error>> {
        let device_residency = crate::mode::select_device_residency(args.gpu);
        let frame_inclusion = resolve_frame_inclusion(args, ui_state.is_some());
        let mut feeds = Vec::with_capacity(args.video_uri.len());
        for (i, uri) in args.video_uri.iter().enumerate() {
            let source = parse_source(uri, args.loop_file);
            let sink = make_sink(ui_state);
            let builder = make_builder()?;
            let redacted_source = format!("{source:?}");
            let feed_config = build_feed_config(
                builder,
                source,
                sink,
                args.queue_depth,
                frame_inclusion,
                args.decode,
                device_residency.clone(),
            )?;
            let feed = runtime.add_feed(feed_config)?;
            info!(feed_index = i, feed_id = %feed.id(), source = redacted_source, "feed started");
            feeds.push(feed);
        }
        Ok(feeds)
    }

    if args.batch {
        // --- Batch mode -----------------------------------------------
        info!(
            batch_size = args.batch_size,
            batch_latency_ms = args.batch_latency_ms,
            feed_count = args.video_uri.len(),
            "batch mode enabled"
        );

        let gpu_timeout = if args.gpu {
            Some(GPU_STARTUP_TIMEOUT)
        } else {
            None
        };

        let batch_hdl = runtime.create_batch(
            Box::new(create_batch_processor(&args)),
            BatchConfig {
                max_batch_size: args.batch_size,
                max_latency: Duration::from_millis(args.batch_latency_ms),
                queue_capacity: None,
                response_timeout: gpu_timeout,
                max_in_flight_per_feed: 1,
                startup_timeout: gpu_timeout,
            },
        )?;
        batch_handle = Some(batch_hdl.clone());

        let trk_cfg = tracker_config(&args);
        feeds = register_feeds(&runtime, &args, &ui_state, &make_sink, &|| {
            let pipeline = FeedPipeline::builder()
                .batch(batch_hdl.clone())?
                .add_stage(TrackerStage::new(trk_cfg.clone()))
                .build();
            Ok(FeedConfig::builder().feed_pipeline(pipeline))
        })?;
    } else {
        // --- Per-feed mode (default) ----------------------------------
        let det_cfg = detector_config(&args);
        let trk_cfg = tracker_config(&args);
        feeds = register_feeds(&runtime, &args, &ui_state, &make_sink, &|| {
            let detector = create_detector_stage(&det_cfg);
            Ok(FeedConfig::builder().stages(vec![
                Box::new(detector),
                Box::new(TrackerStage::new(trk_cfg.clone())),
            ]))
        })?;
    }

    // Health event logger — logs all health events to tracing so they
    // appear in Docker logs / stdout without needing an OTel collector.
    let mut health_rx = runtime_handle.health_subscribe();
    std::thread::Builder::new()
        .name("health-logger".into())
        .spawn(move || {
            while let Ok(event) = health_rx.blocking_recv() {
                match &event {
                    HealthEvent::FrameLag {
                        feed_id,
                        frame_age_ms,
                        frames_lagged,
                    } => {
                        warn!(
                            feed_id = %feed_id,
                            frame_age_ms,
                            frames_lagged,
                            "frame lag detected"
                        );
                    }
                    HealthEvent::BackpressureDrop {
                        feed_id,
                        frames_dropped,
                    } => {
                        warn!(feed_id = %feed_id, frames_dropped, "backpressure drop");
                    }
                    HealthEvent::SourceDisconnected { feed_id, reason } => {
                        warn!(feed_id = %feed_id, %reason, "source disconnected");
                    }
                    HealthEvent::SourceConnected { feed_id } => {
                        info!(feed_id = %feed_id, "source connected");
                    }
                    HealthEvent::FeedStopped { feed_id, reason } => {
                        info!(feed_id = %feed_id, ?reason, "feed stopped");
                    }
                    HealthEvent::FeedRestarting {
                        feed_id,
                        restart_count,
                    } => {
                        warn!(feed_id = %feed_id, restart_count, "feed restarting");
                    }
                    _ => {
                        info!(?event, "health");
                    }
                }
            }
        })?;

    if let Some(state) = ui_state {
        // UI mode: egui window blocks the main thread.
        let inference_label = if args.gpu { "GPU (CUDA)" } else { "CPU" }.to_string();
        info!("launching UI — close window to stop");

        // The glow/EGL backend can panic when no GPU display is reachable
        // (e.g. SSH X-forwarding without EGL support). Catch that and fall
        // back to headless mode so the pipeline stays alive.
        let ui_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ui::run_ui(state, feeds, batch_handle, runtime_handle, inference_label)
        }));
        match ui_result {
            Ok(Ok(())) => {
                info!("UI closed, shutting down");
            }
            Ok(Err(e)) => {
                warn!("UI failed: {e} — falling back to headless mode");
                wait_for_ctrlc();
            }
            Err(_) => {
                warn!(
                    "UI display initialization panicked (no EGL/GLX display?) \
                     — falling back to headless mode. \
                     Hint: set --headless when running over SSH without GPU display forwarding"
                );
                wait_for_ctrlc();
            }
        }
    } else {
        // Headless mode: wait for Ctrl-C.
        wait_for_ctrlc();
    }

    info!("shutting down");
    // Flush metrics before tearing down the runtime they observe.
    if let Some(m) = metrics
        && let Some(rt) = &metrics_rt
        && let Err(e) = rt.block_on(m.shutdown())
    {
        warn!(error = %e, "metrics shutdown error");
    }
    runtime.shutdown()?;
    Ok(())
}

/// Block the current thread until Ctrl-C is pressed.
fn wait_for_ctrlc() {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        let _ = ctrlc::set_handler(move || {
            stop.store(true, Ordering::Relaxed);
        });
    }
    info!("pipeline running (headless) — press Ctrl-C to stop");
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(200));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Inclusion resolution precedence
    // ---------------------------------------------------------------

    /// `sample_interval` overrides `preview_fps`.
    #[test]
    fn sample_interval_overrides_preview_fps() {
        let result = resolve_frame_inclusion_params(Some(3), Some(5.0), true);
        assert_eq!(result, FrameInclusion::Sampled { interval: 3 });
    }

    /// `preview_fps` used when `sample_interval` is absent.
    #[test]
    fn preview_fps_used_when_interval_absent() {
        let result = resolve_frame_inclusion_params(None, Some(5.0), true);
        assert!(
            matches!(result, FrameInclusion::TargetFps { target, fallback_interval }
                if (target - 5.0).abs() < f32::EPSILON && fallback_interval == 6),
            "expected TargetFps {{ target: 5.0, fallback_interval: 6 }}, got {result:?}",
        );
    }

    /// Default used when both absent.
    #[test]
    fn default_used_when_both_absent() {
        let result = resolve_frame_inclusion_params(None, None, true);
        assert_eq!(
            result,
            FrameInclusion::Sampled {
                interval: DEFAULT_PREVIEW_INTERVAL
            },
        );
    }

    /// Headless mode forces Never regardless of other args.
    #[test]
    fn headless_forces_never() {
        assert_eq!(
            resolve_frame_inclusion_params(Some(3), Some(5.0), false),
            FrameInclusion::Never,
        );
        assert_eq!(
            resolve_frame_inclusion_params(None, Some(5.0), false),
            FrameInclusion::Never,
        );
        assert_eq!(
            resolve_frame_inclusion_params(None, None, false),
            FrameInclusion::Never,
        );
    }

    /// `sample_interval = 0` normalizes to Never.
    #[test]
    fn sample_interval_zero_is_never() {
        let result = resolve_frame_inclusion_params(Some(0), None, true);
        assert_eq!(result, FrameInclusion::Never);
    }

    /// `sample_interval = 1` normalizes to Always.
    #[test]
    fn sample_interval_one_is_always() {
        let result = resolve_frame_inclusion_params(Some(1), None, true);
        assert_eq!(result, FrameInclusion::Always);
    }

    /// `preview_fps` high value produces a fallback interval of 1.
    #[test]
    fn preview_fps_high_value_fallback_of_one() {
        let result = resolve_frame_inclusion_params(None, Some(60.0), true);
        // target_fps(60.0, 1) → fallback = round(30/60) = round(0.5) = 1
        // but since 1 → Always via sampled(1), resolve with 30fps source
        // Note: the TargetFps variant stores the raw values; resolution
        // happens at runtime.
        assert!(
            matches!(result, FrameInclusion::TargetFps { .. }),
            "high preview FPS should still create TargetFps: {result:?}",
        );
    }
}
