# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-21

Initial public release of the NextVision video perception runtime.

### Added

#### `nv-core` — Shared types and vocabulary

- `FeedId`, `TrackId`, `DetectionId`, `StageId` — strongly-typed identifiers for all runtime entities.
- `MonotonicTs`, `WallTs`, `Duration` — monotonic and wall-clock timestamps with duration arithmetic.
- `BBox`, `Point2`, `Polygon`, `AffineTransform2D` — normalized-coordinate geometry primitives.
- `SourceSpec` — video source specification (RTSP, file, V4L2, custom pipeline) with integrated `RtspSecurityPolicy`.
- `RtspSecurityPolicy` — TLS enforcement modes (`PreferTls`, `AllowInsecure`, `RequireTls`).
- `CameraMode` — fixed vs. observed camera classification.
- `ReconnectPolicy` — configurable reconnection backoff strategy.
- `NvError`, `MediaError`, `StageError`, `TemporalError`, `ViewError`, `RuntimeError`, `ConfigError` — typed error hierarchy with contextual detail.
- `HealthEvent` — enumeration of runtime lifecycle, error, and status events (source connect/disconnect, stage errors, panics, backpressure, view degradation, restarts).
- `TypedMetadata` — type-map for extensible per-frame and per-observation metadata.
- `FeedMetrics`, `StageMetrics` — atomic, allocation-free throughput and latency counters.
- `redact_url()`, `sanitize_error_string()` — security-aware URL redaction and error string sanitization.

#### `nv-frame` — Frame abstraction

- `FrameEnvelope` — immutable, `Arc`-backed frame container with monotonic and wall-clock timestamps, codec info, and data access.
- `PixelFormat` — enumeration of supported pixel formats (RGB8, BGR8, RGBA8, NV12, I420, Gray8).
- `Residency` — host (CPU) vs. device (accelerator) data location.
- `DataAccess` — fine-grained host-accessibility classification (`HostReadable`, `MappableToHost`, `Opaque`).

#### `nv-media` — GStreamer backend

- `MediaIngress` trait and `MediaIngressFactory` — media source lifecycle abstraction.
- GStreamer-backed implementation with automatic codec negotiation, hardware/software decode selection, and PTS discontinuity detection.
- Zero-copy frame bridging from GStreamer `GstSample` to `FrameEnvelope`.
- Automatic reconnection with configurable backoff on network failures.
- `discover_decode_capabilities()` — runtime decoder capability probing.
- `PostDecodeHook` — optional post-decode frame transformation (GPU transfers, preprocessing).
- `cuda` feature flag for GPU-resident (CUDA device memory) pipeline paths.

#### `nv-perception` — Stage model and perception types

- `Stage` trait — primary user-implementable interface for per-frame perception, with `on_start`, `on_stop`, and `on_view_epoch_change` lifecycle hooks.
- `StageContext` — input bundle providing the current frame, accumulated artifacts, temporal snapshot, view snapshot, and stage metrics.
- `StageOutput` — builder-friendly output with optional detections, tracks, signals, scene features, and typed artifacts.
- `StageCategory` — self-describing stage classification (`FrameAnalysis`, `Association`, `TemporalAnalysis`, `Sink`, `Custom`).
- `StageCapabilities` — declared input/output capabilities for pipeline validation.
- `StagePipeline` — ordered stage container with construction-time dependency validation.
- `Detection` — per-frame object detection with bounding box, confidence, class, embedding, and extensible metadata.
- `DetectionSet` — per-frame detection collection.
- `Track`, `TrackObservation`, `TrackState` — tracked-object lifecycle with per-observation metadata.
- `DerivedSignal`, `SignalValue` — generic named scalar signals.
- `SceneFeature`, `SceneFeatureValue` — scene-level feature observations.
- `PerceptionArtifacts` — frame-local accumulator with typed inter-stage artifact passing.
- `BatchProcessor` trait — GPU-accelerated cross-feed batch inference interface.
- `BatchEntry` — per-frame batch item with feed context and output slot.

#### `nv-temporal` — Temporal state management

- `TemporalStore` — per-feed track history management with bounded retention.
- `TemporalStoreSnapshot` — read-only, `Arc`-wrapped snapshot for safe sharing with stages.
- `Trajectory`, `TrajectorySegment`, `TrajectoryPoint` — spatial path representation with epoch-aware segmentation.
- `MotionFeatures` — per-segment motion analysis (displacement, speed, direction, curvature).
- `SegmentBoundary` — epoch-change boundary with causal explanation.
- `RetentionPolicy` — bounded eviction by track count, age, or mixed strategy.
- Event-driven retention — eviction fires on frame arrival and track lifecycle transitions, not timers.

#### `nv-view` — Camera view-state and PTZ modeling

- `ViewState` — current best estimate of camera pose/zoom/orientation.
- `ViewSnapshot` — read-only, `Arc`-wrapped snapshot for stage consumption.
- `ViewEpoch`, `ViewVersion` — logical epoch and monotonic version tracking for view discontinuities.
- `CameraMotionState` — `Stable`/`Moving`/`Unknown` state machine.
- `MotionSource` — explicit (PTZ telemetry) vs. inferred motion classification.
- `TransitionPhase` — position within a camera-motion transition (`Beginning`/`Middle`/`End`/`Stabilizing`/`Settled`).
- `ViewStateProvider` trait — user-extensible motion information source.
- `EpochPolicy` trait — user-defined response to camera motion (`Continue`/`Degrade`/`Compensate`/`Segment`).
- `ContextValidity` — `Valid`/`Degraded`/`Invalid` classification with causal `DegradationReason`.
- `ViewBoundContext` — bind user data to a specific view for automatic staleness checks.

#### `nv-runtime` — Pipeline orchestration

- `Runtime` and `RuntimeBuilder` — top-level lifecycle manager with configurable feed limits and broadcast capacity.
- `RuntimeHandle` — cloneable control surface for adding/removing feeds, subscribing to outputs and health events, and querying diagnostics.
- `FeedConfig` and `FeedConfigBuilder` — per-feed configuration with source, camera mode, stages, output sink, restart policy, queue depths, and backpressure.
- `FeedHandle` — read-only monitoring handle with metrics, queue telemetry, uptime, and pause/resume controls.
- Dedicated OS thread per feed with independent lifecycle, restart, and temporal state.
- `BackpressurePolicy` — `DropOldest` (default), `DropNewest`, or `Block` queue overflow strategies.
- `OutputEnvelope` — complete perception result with detections, tracks, signals, scene features, view state, full provenance, optional source frame, and admission summary.
- `OutputSink` trait — user-implementable non-fallible output receiver, decoupled on a dedicated sink thread.
- `SinkFactory` — optional factory for constructing fresh sinks after timeout or panic recovery.
- `FrameInclusion` — `Always`/`Sampled`/`Never`/`TargetFps` frame-attachment strategies for controlling pixel delivery rate independently of perception rate.
- `Provenance` — full per-frame audit trail: per-stage timings, view-system decisions, pipeline latency, frame age, queue hold time.
- `BatchHandle`, `BatchConfig`, `BatchCoordinator` — cross-feed GPU batch inference with backpressure, timeout, and per-feed in-flight caps.
- `BatchMetrics` — batch throughput metrics (dispatched, latency, errors, fill ratio).
- `RestartPolicy` — automatic feed restart with configurable trigger, delay, max restarts, and reset window.
- `RuntimeDiagnostics`, `FeedDiagnostics` — one-call composite diagnostic snapshots for all feeds and batch processors.
- Broadcast output channel with `LagDetector` — deterministic ring-buffer saturation detection with throttled `HealthEvent::OutputLagged` events.
- Graceful shutdown with configurable timeouts; sink thread detachment on timeout with `HealthEvent::SinkTimeout`.
- `AdmissionSummary` — per-frame track admission/rejection statistics.

#### `nv-metrics` — Optional OpenTelemetry bridge

- `MetricsExporter` — standalone OTLP/gRPC metrics bridge; polls `RuntimeHandle::diagnostics()` on a configurable interval.
- 30 periodic gauge instruments covering per-feed frame counts, queue depths, stage latencies, view state, batch metrics, and runtime uptime.
- 18 event-driven counter instruments for source disconnects, reconnects, stage errors, panics, feed restarts, backpressure drops, view epoch changes, and output lag.
- Builder pattern with endpoint, service name, and custom attribute configuration.
- Graceful shutdown with flush.

#### `nv-test-util` — Testing helpers

- `StageHarness` — isolated single-stage testing without a full runtime.
- Synthetic frame factories (`solid_frame`, `gradient_frame`) for deterministic testing.
- `MockStage` — configurable `Stage` implementation for pipeline testing.
- `NullTemporalAccess` — stub temporal store for isolated stage tests.
- `TestClock` — controllable monotonic clock for reproducible temporal tests.

#### Architecture

- Domain-agnostic core — no vertical-specific concepts (traffic, retail, security) in any library crate.
- GStreamer isolation — all GStreamer types confined to `nv-media`; public API uses library-defined types.
- Event-driven temporal state — retention and degradation fire on observation events, not timers.
- Zero-copy frame bridging — `Arc`-backed frames shared across stages and output without data copies.
- Bounded memory — every queue, retention policy, and broadcast channel has explicit capacity limits.
- Per-feed isolation — independent lifecycle, state, and failure domains per feed.
- First-class PTZ/view-state — automatic epoch segmentation, continuity degradation, and trajectory partitioning under camera motion.

[Unreleased]: https://github.com/thenadz/nextvision/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thenadz/nextvision/releases/tag/v0.1.0
