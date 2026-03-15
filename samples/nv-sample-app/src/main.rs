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
    } else {
        let mut spec = SourceSpec::file(PathBuf::from(uri));
        if loop_file {
            if let SourceSpec::File { loop_, .. } = &mut spec {
                *loop_ = true;
            }
        }
        spec
    }
}

/// Build the shared detector config from CLI args.
fn detector_config(args: &Cli) -> DetectorConfig {
    DetectorConfig {
        model_path: args.model.clone(),
        input_size: args.input_size,
        confidence_threshold: args.conf_threshold,
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

    // Hold feed handles so feeds aren't dropped.
    let mut _feeds = Vec::new();

    // Optional batch handle (only in batch mode).
    let mut batch_handle: Option<nv_runtime::BatchHandle> = None;

    // Shared UI state (used when not headless).
    let ui_state = if !args.headless {
        Some(new_shared_state())
    } else {
        None
    };

    if args.batch {
        // --- Batch mode -----------------------------------------------
        info!(
            batch_size = args.batch_size,
            batch_latency_ms = args.batch_latency_ms,
            feed_count = args.video_uri.len(),
            "batch mode enabled"
        );

        let batch_hdl = runtime.create_batch(
            Box::new(DetectorBatchProcessor::new(detector_config(&args))),
            BatchConfig {
                max_batch_size: args.batch_size,
                max_latency: Duration::from_millis(args.batch_latency_ms),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
        )?;
        batch_handle = Some(batch_hdl.clone());

        for (i, uri) in args.video_uri.iter().enumerate() {
            let source = parse_source(uri, args.loop_file);

            let sink: Box<dyn OutputSink> = if let Some(ref state) = ui_state {
                Box::new(UiSink::new(Arc::clone(state)))
            } else {
                Box::new(LogSink::new())
            };

            let pipeline = FeedPipeline::builder()
                .batch(batch_hdl.clone())
                .expect("single batch point")
                .add_stage(TrackerStage::new(tracker_config(&args)))
                .build();

            let feed_config = FeedConfig::builder()
                .source(source)
                .camera_mode(CameraMode::Fixed)
                .feed_pipeline(pipeline)
                .output_sink(sink)
                .frame_inclusion(FrameInclusion::Always)
                .backpressure(BackpressurePolicy::DropOldest {
                    queue_depth: args.queue_depth,
                })
                .build()?;

            let feed = runtime.add_feed(feed_config)?;
            let feed_id = feed.id();
            info!(feed_index = i, %feed_id, uri, "feed started");
            _feeds.push(feed);
        }
    } else {
        // --- Per-feed mode (default) ----------------------------------
        for (i, uri) in args.video_uri.iter().enumerate() {
            let source = parse_source(uri, args.loop_file);

            let sink: Box<dyn OutputSink> = if let Some(ref state) = ui_state {
                Box::new(UiSink::new(Arc::clone(state)))
            } else {
                Box::new(LogSink::new())
            };

            let feed_config = FeedConfig::builder()
                .source(source)
                .camera_mode(CameraMode::Fixed)
                .stages(vec![
                    Box::new(DetectorStage::new(detector_config(&args))),
                    Box::new(TrackerStage::new(tracker_config(&args))),
                ])
                .output_sink(sink)
                .frame_inclusion(FrameInclusion::Always)
                .backpressure(BackpressurePolicy::DropOldest {
                    queue_depth: args.queue_depth,
                })
                .build()?;

            let feed = runtime.add_feed(feed_config)?;
            let feed_id = feed.id();
            info!(feed_index = i, %feed_id, uri, "feed started");
            _feeds.push(feed);
        }
    }

    if let Some(state) = ui_state {
        // UI mode: egui window blocks the main thread.
        info!("launching UI — close window to stop");
        ui::run_ui(state, _feeds, batch_handle, runtime_handle)
            .map_err(|e| format!("UI error: {e}"))?;
        info!("UI closed, shutting down");
    } else {
        // Headless mode: wait for Ctrl-C.
        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = Arc::clone(&stop);
            ctrlc::set_handler(move || {
                stop.store(true, Ordering::Relaxed);
            })
            .expect("failed to install Ctrl-C handler");
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
