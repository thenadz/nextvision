# NextVision Samples

Example crates demonstrating how to build perception pipelines on top of
the NextVision runtime using only public APIs.

> **Reference implementations only.** The detection and tracking crates
> bundled here are intentionally simple — they exist to show how the
> library's `Stage` and `BatchProcessor` traits are wired together.
> Library users will typically substitute their own models and algorithms.

## Crate overview

| Crate | Kind | Description |
|---|---|---|
| `nv-sample-detection` | Library | ONNX object-detector — `Stage` and `BatchProcessor` adapters |
| `nv-sample-tracking` | Library | Multi-object tracker — `Stage` adapter |
| `nv-sample-app` | Binary | Runnable sample wiring detection + tracking with a real-time multi-panel UI |

## Running the sample

```sh
# Per-feed mode (default): each feed gets its own detector session
cargo run -p nv-sample-app -- \
  --video-uri rtsp://camera/stream

# From a local file (with optional looping)
cargo run -p nv-sample-app -- \
  --video-uri /path/to/video.mp4 \
  --loop-file

# Batch mode: feeds share a single detector session
cargo run -p nv-sample-app -- \
  --video-uri rtsp://cam1/stream \
  --batch --batch-size 4 --batch-latency-ms 50

# Headless (no GUI, log-only output):
cargo run -p nv-sample-app -- \
  --video-uri rtsp://cam1/stream --headless
```

Press **Ctrl-C** to shut down gracefully.

## Architecture

```
Video source
    │
    ▼
┌──────────────────────┐
│  Detector             │  ← Stage (per-feed) or BatchProcessor (shared)
│  letterbox → ONNX →  │
│  end-to-end decode    │
└──────────┬───────────┘
           │ DetectionSet
           ▼
┌──────────────────────┐
│  Tracker             │  ← Stage (always per-feed)
│  predict → associate │
│  → update → prune    │
└──────────┬───────────┘
           │ Vec<Track>
           ▼
┌──────────────────────┐
│  OutputSink          │  → multi-panel UI with bounding boxes,
│                      │    track IDs, and live telemetry
└──────────────────────┘
```

### Per-feed vs batch mode

- **Per-feed** (`--video-uri`): Each feed creates its own `DetectorStage`,
  loading a separate ONNX session. Simple, no cross-feed coordination.

- **Batch** (`--batch`): All feeds submit frames to a shared
  `DetectorBatchProcessor`. The runtime accumulates up to `--batch-size`
  frames (or waits `--batch-latency-ms`), runs a single batched ONNX
  call, then distributes results back to each feed's pipeline. Tracking
  remains per-feed.

## CLI reference

Run `cargo run -p nv-sample-app -- --help` for all options.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--video-uri` | (required) | RTSP URL or local file path |
| `--model` | `samples/models/yolo26s.onnx` | Path to the ONNX model |
| `--input-size` | `640` | Model input size (width = height) |
| `--conf-threshold` | `0.25` | Detector confidence threshold |
| `--max-age` | `30` | Tracker max coast age (frames) |
| `--min-hits` | `3` | Tracker min hits before confirmed |
| `--queue-depth` | `4` | Backpressure queue depth |
| `--batch` | `false` | Enable cross-feed batch inference |
| `--batch-size` | `8` | Max frames per batch |
| `--batch-latency-ms` | `50` | Max wait before dispatching partial batch |
| `--loop-file` | `false` | Loop file sources |
| `--headless` | `false` | Disable GUI, use log-only sinks |

## Obtaining the model

The sample ships with `samples/models/yolo26s.onnx`, an end-to-end
detection model. End-to-end models include a learned query head that
outputs final detections directly — no NMS post-processing is required.

The output tensor shape is `[batch, max_dets, 6]` where each row is
`[x1, y1, x2, y2, confidence, class_id]`.

The model expects `[batch, 3, 640, 640]` float32 input (RGB, normalised
to 0–1).

If you need to export your own model, ensure it uses a dynamic batch
dimension and end-to-end output format (6 columns per detection).
The startup validation will reject models with incompatible output
shapes.
