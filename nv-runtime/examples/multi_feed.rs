//! Multi-feed example: run several camera feeds through the same runtime.
//!
//! This example demonstrates:
//! - Building a runtime with a custom feed limit.
//! - Adding multiple feeds with independent configurations.
//! - Subscribing to aggregate output and health events.
//! - Graceful shutdown.
//!
//! **Note:** This example will not actually connect to cameras. It uses a
//! mock ingress factory for illustration. Replace with real RTSP URLs and
//! remove the mock factory for production use.

use nv_core::config::{CameraMode, SourceSpec};
use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::StageId;
use nv_perception::{Stage, StageContext, StageOutput};
use nv_runtime::{FeedConfig, OutputEnvelope, OutputSink, Runtime};

/// A minimal detection stage that passes frames through unchanged.
struct PassthroughStage;

impl Stage for PassthroughStage {
    fn id(&self) -> StageId {
        StageId("passthrough")
    }

    fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        Ok(StageOutput::empty())
    }
}

/// A simple output sink that prints a summary of each output.
struct PrintSink {
    label: &'static str,
}

impl OutputSink for PrintSink {
    fn emit(&self, output: OutputEnvelope) {
        println!(
            "[{}] feed={:?} seq={} detections={}",
            self.label,
            output.feed_id,
            output.frame_seq,
            output.detections.len(),
        );
    }
}

fn main() -> Result<(), nv_core::error::NvError> {
    // Build the runtime with a 4-feed limit.
    let runtime = Runtime::builder().max_feeds(4).build()?;

    // Subscribe to health and output events.
    let mut health_rx = runtime.health_subscribe();
    let mut output_rx = runtime.output_subscribe();

    // Add feeds — in production, replace with real camera RTSP URLs.
    let feeds = [
        ("cam-lobby", "rtsp://192.168.1.10/stream"),
        ("cam-entrance", "rtsp://192.168.1.11/stream"),
        ("cam-parking", "rtsp://192.168.1.12/stream"),
    ];

    let mut handles = Vec::new();
    for (label, url) in feeds {
        let config = FeedConfig::builder()
            .source(SourceSpec::rtsp(url))
            .camera_mode(CameraMode::Fixed)
            .stages(vec![Box::new(PassthroughStage)])
            .output_sink(Box::new(PrintSink { label }))
            .build()?;

        let handle = runtime.add_feed(config)?;
        println!("Started feed {:?} ({label})", handle.id());
        handles.push(handle);
    }

    println!("Running {} feeds...", runtime.feed_count()?);

    // In a real application you would loop on health_rx and output_rx.
    // Here we just demonstrate the API surface.
    std::thread::spawn(move || {
        while let Ok(event) = health_rx.blocking_recv() {
            match event {
                HealthEvent::SourceConnected { feed_id } => {
                    println!("Health: feed {feed_id:?} connected");
                }
                HealthEvent::FeedStopped { feed_id, reason } => {
                    println!("Health: feed {feed_id:?} stopped: {reason:?}");
                }
                HealthEvent::OutputLagged { messages_lost } => {
                    println!("Health: output lagged, {messages_lost} messages lost");
                }
                _ => {}
            }
        }
    });

    std::thread::spawn(move || {
        while let Ok(output) = output_rx.blocking_recv() {
            println!(
                "Output: feed={:?} seq={} latency={:?}",
                output.feed_id, output.frame_seq, output.provenance.total_latency,
            );
        }
    });

    // Wait for a signal or timeout (in production, use real signal handling).
    std::thread::sleep(std::time::Duration::from_secs(5));

    // Check metrics before shutdown.
    for handle in &handles {
        let m = handle.metrics();
        println!(
            "Feed {:?}: processed={}, dropped={}, restarts={}",
            handle.id(),
            m.frames_processed,
            m.frames_dropped,
            m.restarts,
        );
    }

    runtime.shutdown()?;
    println!("Runtime shut down cleanly.");
    Ok(())
}
