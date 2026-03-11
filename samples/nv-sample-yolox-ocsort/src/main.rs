//! YOLOX + OC-SORT sample pipeline.
//!
//! Demonstrates how to wire YOLOX-s detection and OC-SORT tracking into
//! the NextVision runtime using only public APIs, with a real-time overlay
//! window that renders bounding boxes, track IDs, and FPS.
//!
//! Run:
//! ```sh
//! cargo run -p nv-sample-yolox-ocsort -- \
//!   --video-uri rtsp://camera/stream \
//!   --model yolox_s.onnx
//! ```

mod cli;
mod overlay;
mod sink;

use std::path::PathBuf;

use clap::Parser;
use tracing::error;

use nv_byo_ocsort::{OcSortConfig, OcSortStage};
use nv_byo_yolox::{YoloxConfig, YoloxStage};
use nv_core::{CameraMode, SourceSpec};
use nv_runtime::{BackpressurePolicy, FeedConfig, FrameInclusion, Runtime};

use crate::cli::Cli;
use crate::sink::OverlaySink;

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

fn run(args: Cli) -> Result<(), Box<dyn std::error::Error>> {
    // --- Source ---
    let source = if args.video_uri.starts_with("rtsp://")
        || args.video_uri.starts_with("rtsps://")
    {
        SourceSpec::rtsp(&args.video_uri)
    } else {
        let mut spec = SourceSpec::file(PathBuf::from(&args.video_uri));
        if args.loop_file {
            if let SourceSpec::File { loop_, .. } = &mut spec {
                *loop_ = true;
            }
        }
        spec
    };

    // --- Stages ---
    let detector = YoloxStage::new(YoloxConfig {
        model_path: args.model,
        input_size: args.input_size,
        confidence_threshold: args.conf_threshold,
        nms_threshold: args.nms_threshold,
        ..Default::default()
    });

    let tracker = OcSortStage::new(OcSortConfig {
        max_age: args.max_age,
        min_hits: args.min_hits,
        iou_threshold: args.iou_threshold,
        ..Default::default()
    });

    // --- Sink (spawns the UI thread) ---
    let sink = OverlaySink::spawn(640, 480);

    // --- Runtime ---
    let runtime = Runtime::builder().build()?;

    let feed_config = FeedConfig::builder()
        .source(source)
        .camera_mode(CameraMode::Fixed)
        .stages(vec![
            Box::new(detector),
            Box::new(tracker),
        ])
        .output_sink(Box::new(sink))
        .frame_inclusion(FrameInclusion::Always)
        .backpressure(BackpressurePolicy::DropOldest {
            queue_depth: args.queue_depth,
        })
        .build()?;

    let _feed = runtime.add_feed(feed_config)?;

    // Block main thread until the process is killed (Ctrl-C).
    // The OS default SIGINT handler terminates the process, which drops
    // the runtime and performs a graceful sink shutdown.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
