# NextVision

[![CI](https://github.com/thenadz/nextvision/actions/workflows/ci.yml/badge.svg)](https://github.com/thenadz/nextvision/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Rust: 1.92+](https://img.shields.io/badge/Rust-1.92%2B-orange.svg)](https://www.rust-lang.org)

A domain-agnostic, production-oriented, high-performance Rust video perception runtime built on GStreamer.

## Overview

NextVision ingests live or recorded video, normalizes frames into a library-owned abstraction, pushes them through a linear sequence of user-supplied perception stages, maintains temporal state (tracks, trajectories, motion features), models PTZ/view-state, and emits structured, provenanced outputs.

## Workspace crates

| Crate | Purpose |
|---|---|
| `nv-core` | Shared types: IDs, timestamps, geometry, errors, metadata, health events, metrics |
| `nv-frame` | Frame abstraction — zero-copy, ref-counted `FrameEnvelope` |
| `nv-media`     | GStreamer backend — the only crate that depends on `gstreamer-rs`. Optional `cuda` feature for GPU-resident frames |
| `nv-perception` | Stage trait, detection/track types, perception artifacts, derived signals |
| `nv-temporal` | Temporal state: trajectories, motion features, continuity, retention |
| `nv-view` | PTZ/view-state, camera motion modeling, epoch policy, context validity |
| `nv-runtime` | Pipeline orchestration, feed lifecycle, output, concurrency, provenance |
| `nv-metrics` | Optional OpenTelemetry metrics bridge — exports diagnostics and health events via OTLP |
| `nv-test-util` | Test harnesses, synthetic frames, mock stages, controllable clock |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full specification.

## Quick start

```rust
use nv_runtime::{Runtime, FeedConfig};
use nv_core::{SourceSpec, CameraMode};
use nv_runtime::{OutputSink, SharedOutput};

struct MyOutputSink;
impl OutputSink for MyOutputSink {
    fn emit(&self, _output: SharedOutput) {
        // forward to Kafka, gRPC, file, etc.
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder().build()?;

    let _feed = runtime.add_feed(
        FeedConfig::builder()
            .source(SourceSpec::rtsp("rtsp://cam1/stream"))
            .camera_mode(CameraMode::Fixed)
            .stages(vec![/* your stages here */])
            .output_sink(Box::new(MyOutputSink))
            .build()?
    )?;

    // feed is now running
    // runtime.shutdown();
    Ok(())
}
```

### Batch inference across feeds

```rust
use nv_runtime::{
    Runtime, FeedConfig, FeedPipeline, BatchConfig, BackpressurePolicy, OutputSink, SharedOutput,
};
use nv_core::{SourceSpec, CameraMode};
use nv_perception::batch::{BatchProcessor, BatchEntry};
use nv_perception::{StageId, StageOutput, DetectionSet};
use nv_core::error::StageError;
use std::time::Duration;

struct MyDetector { /* model handle */ }
impl BatchProcessor for MyDetector {
    fn id(&self) -> StageId { StageId("detector") }
    fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
        for item in items.iter_mut() {
            // run inference on item.frame …
            item.output = Some(StageOutput::with_detections(DetectionSet::empty()));
        }
        Ok(())
    }
}

struct Sink;
impl OutputSink for Sink { fn emit(&self, _: SharedOutput) {} }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder().build()?;

    // 1. Create a shared batch coordinator.
    let batch = runtime.create_batch(
        Box::new(MyDetector { /* … */ }),
        BatchConfig {
            max_batch_size: 8,
            max_latency: Duration::from_millis(50),
            queue_capacity: None, // defaults to max_batch_size * 4
            response_timeout: None, // defaults to 5s safety margin
            max_in_flight_per_feed: 1, // prevent timeout-induced stacking
            startup_timeout: None, // defaults to 30s; increase for GPU warm-up
        },
    )?;

    // 2. Build per-feed pipelines that share the batch point.
    for url in ["rtsp://cam1/stream", "rtsp://cam2/stream"] {
        let pipeline = FeedPipeline::builder()
            // .stage(Box::new(pre_batch_stage))
            .batch(batch.clone())?
            // .stage(Box::new(post_batch_stage))
            .build();

        runtime.add_feed(
            FeedConfig::builder()
                .source(SourceSpec::rtsp(url))
                .camera_mode(CameraMode::Fixed)
                .feed_pipeline(pipeline)
                .output_sink(Box::new(Sink))
                .build()?
        )?;
    }

    // runtime.shutdown();
    Ok(())
}
```

## Design principles

- **Domain-agnostic**: No transit, retail, security, or other vertical concepts in core.
- **GStreamer is an implementation detail**: All GStreamer types confined to `nv-media`.
- **Performance first**: Zero-copy frame handoff, bounded queues, allocation-conscious hot paths.
- **Operational clarity**: Every queue is bounded, every restart path is explicit, every failure is typed.
- **Composition over framework**: Linear stage pipeline, not a DAG engine or meta-framework.

## Requirements

- **Rust 1.92+** (edition 2024)
- **GStreamer 1.16+** runtime libraries (for video ingestion)
- Linux recommended for production; macOS supported for development; Windows supported with use of VS Code devcontainer

### Installation

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
nv-runtime = "0.1"          # Full runtime (includes nv-core, nv-frame, etc.)
nv-perception = "0.1"       # Stage trait, detection/track types
nv-metrics = "0.1"          # Optional: OpenTelemetry metrics bridge
```

For most applications, `nv-runtime` is the only direct dependency needed — it
re-exports the essential types from `nv-core`, `nv-frame`, `nv-perception`,
`nv-temporal`, and `nv-view`.

### Building from source

```bash
# Install GStreamer development libraries (Debian/Ubuntu)
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Build the workspace
cargo build --workspace

# Run the test suite
cargo test --workspace
```

## Telemetry

The runtime exposes lightweight telemetry through public APIs. Per-feed and
batch counters are atomic and allocation-free. Aggregate diagnostics snapshots
(`Runtime::diagnostics()`) allocate a `Vec` for the feed/batch lists and may
clone short strings (e.g. decode-status detail).

```rust
use nv_runtime::{Runtime, FeedHandle, QueueTelemetry, BatchMetrics};

fn inspect(runtime: &Runtime, feed: &FeedHandle) {
    // Runtime uptime (monotonic, since creation).
    let uptime = runtime.uptime();

    // Feed uptime (session-scoped — resets on restart).
    let feed_uptime = feed.uptime();

    // Per-feed frame metrics.
    let metrics = feed.metrics();
    println!("frames processed: {}", metrics.frames_processed);

    // Queue depths (source + sink).
    let queues = feed.queue_telemetry();
    println!(
        "source queue: {}/{}, sink queue: {}/{}",
        queues.source_depth, queues.source_capacity,
        queues.sink_depth, queues.sink_capacity,
    );
}
```

Batch coordinator telemetry:

```rust
use nv_runtime::BatchHandle;

fn inspect_batch(batch: &BatchHandle) {
    let m = batch.metrics();
    println!("batches: {}, pending: {}", m.batches_dispatched, m.pending_items());
    if let Some(fill) = m.avg_fill_ratio() {
        println!("avg fill: {:.0}%", fill * 100.0);
    }
}
```

All counters are atomic — no per-frame allocations, no background polling
threads, no additional mutex contention on hot paths.

### OpenTelemetry export (`nv-metrics`)

The optional `nv-metrics` crate bridges runtime diagnostics and health events
into OpenTelemetry instruments, exported via OTLP/gRPC. It runs in a background
tokio task and does not touch the frame-processing hot path.

Two families of instruments are exported:

- **Periodic gauges** (30 instruments) — sampled from `Runtime::diagnostics()`
  on a configurable poll interval. Covers per-feed frame counts, queue depths,
  stage latencies, view-state, batch metrics, and runtime uptime.
- **Event-driven counters** (18 instruments) — incremented in real time as
  `HealthEvent` variants arrive via `health_subscribe()`. Covers source
  disconnects, reconnects, stage errors, panics, feed restarts, backpressure
  drops, view epoch changes, and view degradations.

```rust
use nv_metrics::MetricsExporter;

// Inside a tokio runtime:
let _exporter = MetricsExporter::builder()
    .runtime_handle(handle.clone())
    .otlp_endpoint("http://localhost:4317")
    .service_name("my-app")
    .build()?;
// Gauges poll automatically; health counters fire on events.
// Call exporter.shutdown().await for clean teardown;
// drop alone cancels the poll loop but does not flush.
```

See the [nv-metrics crate docs](nv-metrics/src/lib.rs) for the full instrument
table and builder options.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
