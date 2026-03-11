use clap::Parser;
use std::path::PathBuf;

/// CLI arguments for the YOLOX + OC-SORT sample pipeline.
#[derive(Parser, Debug)]
#[command(name = "yolox-ocsort-sample")]
#[command(about = "Run YOLOX-s detection + OC-SORT tracking on a video source")]
pub struct Cli {
    /// Video source URI (RTSP URL or file path).
    #[arg(long, short = 'u')]
    pub video_uri: String,

    /// Path to the YOLOX ONNX model file.
    #[arg(long, short = 'm', default_value = "yolox_s.onnx")]
    pub model: PathBuf,

    /// Model input size (width = height).
    #[arg(long, default_value_t = 640)]
    pub input_size: u32,

    /// Detector confidence threshold.
    #[arg(long, default_value_t = 0.25)]
    pub conf_threshold: f32,

    /// Detector NMS IoU threshold.
    #[arg(long, default_value_t = 0.45)]
    pub nms_threshold: f32,

    /// Tracker max coast age (frames).
    #[arg(long, default_value_t = 30)]
    pub max_age: u32,

    /// Tracker min-hits before confirmed.
    #[arg(long, default_value_t = 3)]
    pub min_hits: u32,

    /// Tracker IoU association threshold.
    #[arg(long, default_value_t = 0.3)]
    pub iou_threshold: f32,

    /// Maximum output frames per second (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    pub max_fps: u32,

    /// Frame queue depth (backpressure).
    #[arg(long, default_value_t = 4)]
    pub queue_depth: usize,

    /// Whether the video source should loop (file sources only).
    #[arg(long, default_value_t = false)]
    pub loop_file: bool,
}
