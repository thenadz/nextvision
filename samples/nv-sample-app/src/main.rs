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
mod overlay;
mod ui;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use tracing::{error, info};

use nv_metrics::MetricsExporter;
use nv_sample_tracking::{TrackerConfig, TrackerStage};
use nv_sample_detection::{DetectorBatchProcessor, DetectorConfig, DetectorStage};
use nv_core::{CameraMode, SourceSpec};
use nv_runtime::{
    BackpressurePolicy, BatchConfig, FeedConfig, FeedPipeline, FrameInclusion, OutputEnvelope,
    OutputSink, Runtime,
};

use crate::cli::Cli;
use crate::ui::{new_shared_state, UiSink};

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

/// CUDA JIT compilation on first run can take 30+ seconds.
/// Allow a generous timeout for the initial inference.
const GPU_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// Lightweight output sink that periodically logs detection/track summaries.
///
/// Used in headless mode for feeds that don't need a display.
struct LogSink {
    count: AtomicU64,
}

impl LogSink {
    fn new() -> Self {
        Self { count: AtomicU64::new(0) }
    }
}

impl OutputSink for LogSink {
    fn emit(&self, output: Arc<OutputEnvelope>) {
        let n = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 60 == 0 {
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

    // Metrics: spin up a tokio runtime for the OTel background exporter.
    // Only active when --otlp-endpoint is provided.
    let _tokio_rt = tokio::runtime::Runtime::new()?;
    let _tokio_guard = _tokio_rt.enter();
    let _metrics = args.otlp_endpoint.as_ref().map(|endpoint| {
        MetricsExporter::builder()
            .runtime_handle(runtime_handle.clone())
            .otlp_endpoint(endpoint)
            .service_name("nv-sample-app")
            .build()
    }).transpose()?;

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
    ///
    /// When `include_frames` is true, `FrameInclusion::Always` attaches
    /// full frame pixels to every output — needed for UI overlay rendering
    /// but doubles per-frame Arc memory. Pass `false` for headless or
    /// production pipelines where frame pixel data is not needed downstream.
    fn build_feed_config(
        builder: nv_runtime::FeedConfigBuilder,
        source: SourceSpec,
        sink: Box<dyn OutputSink>,
        queue_depth: usize,
        include_frames: bool,
        decode: nv_runtime::DecodePreference,
    ) -> Result<FeedConfig, Box<dyn std::error::Error>> {
        let mut builder = builder
            .source(source)
            .camera_mode(CameraMode::Fixed)
            .output_sink(sink)
            .decode_preference(decode)
            .backpressure(BackpressurePolicy::DropOldest { queue_depth });

        // On Jetson, hardware decoders (nvv4l2decoder) output
        // video/x-raw(memory:NVMM) — GPU-mapped buffers that the
        // standard videoconvert cannot accept. Insert nvvidconv to
        // copy NVMM → system memory when detected.
        builder = builder.post_decode_hook(std::sync::Arc::new(|info| {
            if info.memory_type.as_deref() == Some("NVMM") {
                Some("nvvidconv".into())
            } else {
                None
            }
        }));

        if include_frames {
            // Carry full pixel data in every OutputEnvelope so the UI
            // can render detection/track overlays on the video frame.
            builder = builder.frame_inclusion(FrameInclusion::Always);
        }
        // When false, the default FrameInclusion::Never avoids the
        // per-frame pixel payload — appropriate for headless runs.

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
        make_builder: &dyn Fn() -> Result<nv_runtime::FeedConfigBuilder, Box<dyn std::error::Error>>,
    ) -> Result<Vec<nv_runtime::FeedHandle>, Box<dyn std::error::Error>> {
        let mut feeds = Vec::with_capacity(args.video_uri.len());
        for (i, uri) in args.video_uri.iter().enumerate() {
            let source = parse_source(uri, args.loop_file);
            let sink = make_sink(ui_state);
            let builder = make_builder()?;
            let feed_config = build_feed_config(
                builder, source, sink, args.queue_depth,
                ui_state.is_some(), args.decode,
            )?;
            let feed = runtime.add_feed(feed_config)?;
            info!(feed_index = i, feed_id = %feed.id(), uri, "feed started");
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

        let gpu_timeout = if args.gpu { Some(GPU_STARTUP_TIMEOUT) } else { None };

        let batch_hdl = runtime.create_batch(
            Box::new(DetectorBatchProcessor::new(detector_config(&args))),
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
            Ok(FeedConfig::builder().stages(vec![
                Box::new(DetectorStage::new(det_cfg.clone())),
                Box::new(TrackerStage::new(trk_cfg.clone())),
            ]))
        })?;
    }

    if let Some(state) = ui_state {
        // UI mode: egui window blocks the main thread.
        let inference_label = if args.gpu { "GPU (CUDA)" } else { "CPU" }.to_string();
        info!("launching UI — close window to stop");
        ui::run_ui(state, feeds, batch_handle, runtime_handle, inference_label)
            .map_err(|e| format!("UI error: {e}"))?;
        info!("UI closed, shutting down");
    } else {
        // Headless mode: wait for Ctrl-C.
        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = Arc::clone(&stop);
            ctrlc::set_handler(move || {
                stop.store(true, Ordering::Relaxed);
            })?;
        }

        info!("pipeline running (headless) — press Ctrl-C to stop");
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    info!("shutting down");
    runtime.shutdown()?;
    Ok(())
}
