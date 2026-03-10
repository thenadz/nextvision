# NextVision

A domain-agnostic, production-oriented, high-performance Rust video perception runtime built on GStreamer.

## Overview

NextVision ingests live or recorded video, normalizes frames into a library-owned abstraction, pushes them through a linear sequence of user-supplied perception stages, maintains temporal state (tracks, trajectories, motion features), models PTZ/view-state, and emits structured, provenanced outputs.

## Workspace crates

| Crate | Purpose |
|---|---|
| `nv-core` | Shared types: IDs, timestamps, geometry, errors, metadata, health events, metrics |
| `nv-frame` | Frame abstraction — zero-copy, ref-counted `FrameEnvelope` |
| `nv-media` | GStreamer backend — the only crate that depends on `gstreamer-rs` |
| `nv-perception` | Stage trait, detection/track types, perception artifacts, derived signals |
| `nv-temporal` | Temporal state: trajectories, motion features, continuity, retention |
| `nv-view` | PTZ/view-state, camera motion modeling, epoch policy, context validity |
| `nv-runtime` | Pipeline orchestration, feed lifecycle, output, concurrency, provenance |
| `nv-test-util` | Test harnesses, synthetic frames, mock stages, controllable clock |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full specification.

## Quick start

```rust
use nv_runtime::{Runtime, FeedConfig};
use nv_core::{SourceSpec, CameraMode};
use nv_runtime::{BackpressurePolicy, OutputSink, SharedOutput};

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

## Design principles

- **Domain-agnostic**: No transit, retail, security, or other vertical concepts in core.
- **GStreamer is an implementation detail**: All GStreamer types confined to `nv-media`.
- **Performance first**: Zero-copy frame handoff, bounded queues, allocation-conscious hot paths.
- **Operational clarity**: Every queue is bounded, every restart path is explicit, every failure is typed.
- **Composition over framework**: Linear stage pipeline, not a DAG engine or meta-framework.

## License

Licensed under either of Apache License, Version 2.0 or MIT license, at your option.
