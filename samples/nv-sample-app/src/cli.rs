use clap::Parser;
use std::path::PathBuf;

/// CLI arguments for the sample detection + tracking pipeline.
#[derive(Parser, Debug)]
#[command(name = "nv-sample-app")]
#[command(about = "Run detection + tracking on a video source")]
pub struct Cli {
    /// Video source URI(s). Pass multiple times for multi-feed batching:
    ///   --video-uri rtsp://cam1/stream --video-uri rtsp://cam2/stream
    #[arg(long, short = 'u', required = true)]
    pub video_uri: Vec<String>,

    /// Path to the ONNX model file.
    #[arg(long, short = 'm', default_value = "samples/models/yolo26s.onnx")]
    pub model: PathBuf,

    /// Model input size (width = height).
    #[arg(long, default_value_t = 640)]
    pub input_size: u32,

    /// Detector confidence threshold.
    #[arg(long, default_value_t = 0.25)]
    pub conf_threshold: f32,

    /// Tracker max coast age (frames).
    #[arg(long, default_value_t = 30)]
    pub max_age: u32,

    /// Tracker min-hits before confirmed.
    #[arg(long, default_value_t = 3)]
    pub min_hits: u32,

    /// Tracker IoU association threshold.
    #[arg(long, default_value_t = 0.3)]
    pub iou_threshold: f32,

    /// Frame queue depth (backpressure).
    #[arg(long, default_value_t = 4)]
    pub queue_depth: usize,

    /// Whether the video source should loop (file sources only).
    #[arg(long, default_value_t = false)]
    pub loop_file: bool,

    /// Enable cross-feed batch inference (uses BatchProcessor instead of
    /// per-feed Stage for detection).
    #[arg(long, default_value_t = false)]
    pub batch: bool,

    /// Maximum frames per batch (only used with --batch).
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Maximum wait time (ms) before dispatching a partial batch.
    #[arg(long, default_value_t = 50)]
    pub batch_latency_ms: u64,

    /// Run in headless mode (no GUI window). Uses log-only output sinks
    /// instead of the multi-panel UI.
    #[arg(long, default_value_t = false)]
    pub headless: bool,

    /// Run detection on GPU via the CUDA execution provider.
    /// Requires the `gpu` feature on nv-sample-detection.
    #[arg(long, default_value_t = false)]
    pub gpu: bool,

    /// OTLP gRPC endpoint for OpenTelemetry metrics export.
    /// When set, runtime diagnostics are exported as standard OTel metrics.
    /// Example: --otlp-endpoint http://localhost:4317
    #[arg(long)]
    pub otlp_endpoint: Option<String>,

    /// Video decoder preference: auto, software, prefer-hw, require-hw.
    ///
    /// Use "software" to force CPU decoding — bypasses hardware decoder
    /// and NVMM memory, which avoids EGL/nvbufsurftransform issues in
    /// containers without full GPU display access.
    #[arg(long, default_value = "auto", value_parser = parse_decode_preference)]
    pub decode: nv_runtime::DecodePreference,
}

fn parse_decode_preference(s: &str) -> Result<nv_runtime::DecodePreference, String> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Ok(nv_runtime::DecodePreference::Auto),
        "software" | "sw" | "cpu" => Ok(nv_runtime::DecodePreference::CpuOnly),
        "prefer-hw" | "prefer_hw" => Ok(nv_runtime::DecodePreference::PreferHardware),
        "require-hw" | "require_hw" => Ok(nv_runtime::DecodePreference::RequireHardware),
        _ => Err(format!(
            "unknown decode preference '{s}'; expected: auto, software, prefer-hw, require-hw"
        )),
    }
}
