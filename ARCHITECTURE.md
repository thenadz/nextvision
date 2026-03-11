# NextVision Architecture Specification

## 1. Architecture overview

NextVision is a Rust workspace that provides a concurrent video perception runtime. It ingests live or recorded video through GStreamer, normalizes frames into a library-owned abstraction, pushes them through a linear sequence of user-supplied perception stages, maintains temporal state (tracks, trajectories, motion features), models PTZ/view-state, and emits structured, provenanced outputs.

### Guiding constraints

| Constraint | Consequence |
|---|---|
| Domain-agnostic | No transit, retail, security, or other vertical concepts in core. |
| GStreamer is an implementation detail | All GStreamer types are confined to `nv-media`. Public API exposes library-defined types only. |
| Performance first | Zero-copy frame handoff where possible; bounded queues everywhere; allocation-conscious hot path. |
| Operational clarity over elegance | Every queue is bounded. Every restart path is explicit. Every failure mode is typed. |
| Composition over framework | The stage model is a concrete ordered pipeline, not a DAG engine or meta-framework. |

### High-level data flow

```
SourceSpec ─► [nv-media] ─► RawFrame
                               │
                               ▼
                          FrameEnvelope  (library-owned, Arc<>-wrapped pixel data)
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
           Stage 0         Stage 1        Stage N    (user-supplied, ordered)
           (detect)        (track)        (custom)
                │              │              │
                └──────┬───────┘              │
                       ▼                      │
                 PerceptionArtifacts ──────────┘
                       │
                       ▼
                 TemporalStore   ◄──── ViewState / CameraMotionState
                       │
                       ▼
                 OutputEnvelope  (structured, provenanced)
                       │
                       ▼
                 User callback / channel
```

Every feed (source) runs as an isolated pipeline. Feeds share nothing except the global runtime handle and metrics registry.

---

## 2. Crate/workspace layout

```
nextvision/
├── Cargo.toml                    # workspace root
├── AGENTS.md
├── ARCHITECTURE.md
│
├── nv-core/                      # shared types, ids, timestamps, geometry, errors, observability
│   └── src/
│       ├── lib.rs
│       ├── id.rs                 # FeedId, TrackId, DetectionId, StageId — all newtypes
│       ├── timestamp.rs          # MonotonicTs, WallTs, Duration
│       ├── geom.rs               # BBox, Point2, Polygon, AffineTransform2D
│       ├── error.rs              # NvError, StageError, MediaError, etc.
│       ├── metadata.rs           # TypedMetadata bag (type-map pattern)
│       ├── health.rs             # HealthEvent enum, StopReason
│       └── metrics.rs            # MetricsRegistry, per-feed counter/histogram helpers
│
├── nv-frame/                     # frame abstraction — zero-copy, ref-counted
│   └── src/
│       ├── lib.rs
│       ├── frame.rs              # FrameEnvelope, PixelData, PixelFormat
│       ├── pool.rs               # optional buffer pool / slab
│       └── convert.rs            # pixel format conversion utilities
│
├── nv-media/                     # GStreamer backend — only crate that depends on gstreamer-rs
│   └── src/
│       ├── lib.rs
│       ├── source.rs             # SourceHandle, reconnect logic
│       ├── pipeline.rs           # GStreamer pipeline construction, appsink wiring
│       ├── decode.rs             # codec handling, HW accel negotiation
│       └── bridge.rs             # GstSample → FrameEnvelope zero-copy bridge
│
├── nv-perception/                # stage model, detection/track types, artifacts
│   └── src/
│       ├── lib.rs
│       ├── stage.rs              # Stage trait, StageContext, StageOutput
│       ├── detection.rs          # Detection, DetectionSet
│       ├── track.rs              # Track, TrackObservation, TrackState
│       ├── artifact.rs           # PerceptionArtifacts — typed map of stage outputs
│       └── signal.rs             # DerivedSignal — generic named scalar/vector signals
│
├── nv-temporal/                  # temporal state: trajectories, motion, continuity
│   └── src/
│       ├── lib.rs
│       ├── store.rs              # TemporalStore — per-feed, owns tracks + trajectories
│       ├── trajectory.rs         # Trajectory, TrajectorySegment, MotionFeatures
│       ├── continuity.rs         # ContinuityState, DegradationReason
│       └── retention.rs          # eviction policy, sliding-window, max-age
│
├── nv-view/                      # PTZ/view-state, camera motion, context validity
│   └── src/
│       ├── lib.rs
│       ├── view_state.rs         # ViewState, ViewSnapshot, ViewVersion, ViewStateChange
│       ├── camera_mode.rs        # CameraMode enum
│       ├── camera_motion.rs      # CameraMotionState, CameraMotionEstimate, MotionSource
│       ├── ptz.rs                # PtzTelemetry, PtzCommand (optional input)
│       ├── provider.rs           # ViewStateProvider trait, MotionReport, MotionPollContext
│       ├── epoch.rs              # EpochPolicy trait, EpochDecision, DefaultEpochPolicy
│       ├── transition.rs         # TransitionPhase state machine
│       ├── transform.rs          # GlobalTransformEstimate, homography bridge
│       ├── validity.rs           # ContextValidity, ViewEpoch, ViewVersion
│       └── bound.rs              # ViewBoundContext<T>, BoundContextValidity
│
├── nv-runtime/                   # pipeline orchestration, feed lifecycle, output, concurrency
│   └── src/
│       ├── lib.rs
│       ├── runtime.rs            # Runtime, RuntimeConfig, RuntimeHandle
│       ├── feed.rs               # FeedPipeline, FeedHandle — per-feed actor
│       ├── scheduler.rs          # stage executor, thread/task scheduling
│       ├── backpressure.rs       # bounded channel wrappers, drop policy
│       ├── shutdown.rs           # graceful shutdown, drain, abort
│       ├── output.rs             # OutputEnvelope, OutputSink trait
│       ├── provenance.rs         # Provenance, StageProvenance, ViewProvenance
│       └── tracing.rs            # span helpers, structured log context
│
└── nv-test-util/                 # test harnesses, synthetic frames, mock stages
    └── src/
        ├── lib.rs
        ├── synthetic.rs          # synthetic FrameEnvelopes with known content
        ├── mock_stage.rs         # configurable mock Stage impls
        └── clock.rs              # controllable test clock
```

### Why 8 crates, not 10

The original design had `nv-output` (3 files: envelope, provenance, sink) and `nv-observe` (3 files: metrics, health, tracing) as separate crates. Both were too thin to justify the crate boundary overhead:

- **`nv-output`** depended on 4 other crates (`nv-core`, `nv-perception`, `nv-temporal`, `nv-view`) and contributed only an aggregation struct (`OutputEnvelope`), a provenance struct, and a 1-method trait (`OutputSink`). These are direct artifacts of the pipeline orchestrator and belong in `nv-runtime`.
- **`nv-observe`** needed to reference types from `nv-core`, `nv-view`, and `nv-perception` (for `HealthEvent` variants that carry `StageError`, `MediaError`, `ViewEpoch`, etc.). This created dependency pressure that either forced `nv-observe` to depend on half the workspace or forced the `HealthEvent` enum to live elsewhere. `HealthEvent` and `MetricsRegistry` are foundational types — `HealthEvent` goes in `nv-core` (where the error/id types it references already live), metrics helpers go in `nv-core`, and tracing span helpers go in `nv-runtime` (where the feed/stage execution context lives).

### Dependency graph (simplified)

```
nv-core          ← depended on by everything
nv-frame         ← nv-core
nv-media         ← nv-core, nv-frame, gstreamer-rs (private)
nv-perception    ← nv-core, nv-frame
nv-temporal      ← nv-core, nv-perception, nv-view
nv-view          ← nv-core
nv-runtime       ← nv-core, nv-frame, nv-media, nv-perception, nv-temporal, nv-view
nv-test-util     ← nv-core, nv-frame, nv-perception (dev-dependency only)
```

Key rule: `gstreamer-rs` appears **only** in `nv-media`'s `Cargo.toml`. No other crate transitively depends on it.

---

## 3. Public API shape

The top-level user-facing API lives in `nv-runtime` and re-exports essential types from the other crates.

```rust
// Minimal user program (pseudocode)
let runtime = Runtime::builder()
    .metrics_registry(registry)
    .build()?;

let feed = runtime.add_feed(
    FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://cam1/stream"))
        .camera_mode(CameraMode::Observed)  // required — no silent default
        .stages(vec![
            Box::new(MyDetector::new(model_path)),
            Box::new(MyTracker::new(tracker_config)),
            Box::new(MySignalStage::new()),
        ])
        .view_state_provider(Box::new(MyEgomotionEstimator::new()))
        .epoch_policy(Box::new(DefaultEpochPolicy::default()))
        .output_sink(Box::new(MyOutputSink::new()))
        .backpressure(BackpressurePolicy::DropOldest { queue_depth: 4 })
        .temporal(TemporalConfig::default())
        .build()?,   // fails if Observed but no provider, or Fixed with provider
)?;

// feed is now running; returns FeedHandle
feed.health_recv();    // → async broadcast::Receiver<HealthEvent>
feed.metrics();        // → FeedMetrics snapshot

// later
feed.pause()?;
feed.resume()?;
runtime.remove_feed(feed.id())?;
runtime.shutdown().await?;
```

### Design decisions

- **Builder pattern for config**, plain methods for runtime operations.
- **No dynamic registration of stages after start.** Stages are fixed at feed creation. This eliminates a class of race conditions and simplifies reasoning.
- **`FeedHandle` is the primary user touchpoint** for per-feed operations once running.
- **`Runtime` manages cross-feed concerns**: thread pools, global metrics, shutdown coordination.

---

## 4. Key structs/enums/traits

### `nv-core`

```rust
/// Opaque, Copy, Eq, Hash, Display. Backed by u64.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeedId(u64);
pub struct TrackId(u64);
pub struct DetectionId(u64);
pub struct StageId(&'static str);  // compile-time name, not runtime-generated

/// Monotonic timestamp — nanoseconds since feed start. Never wall-clock.
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct MonotonicTs(u64);

/// Wall-clock timestamp — only for output/provenance, never for ordering logic.
#[derive(Clone, Copy)]
pub struct WallTs(i64); // microseconds since unix epoch

/// Axis-aligned bounding box, normalized [0,1] coordinates.
#[derive(Clone, Copy)]
pub struct BBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// 2D point, normalized [0,1].
#[derive(Clone, Copy)]
pub struct Point2 { pub x: f32, pub y: f32 }

/// 2D affine transform (3x3 matrix stored as [f64; 6]).
#[derive(Clone, Copy)]
pub struct AffineTransform2D { pub m: [f64; 6] }

/// Typed metadata bag — type-map pattern using TypeId keys.
/// Stores arbitrary `Send + Sync + 'static` values keyed by their concrete type.
/// At most one value per concrete type. Used on FrameEnvelope, Detection, Track,
/// OutputEnvelope, PerceptionArtifacts, and StageOutput.
pub struct TypedMetadata { /* AnyMap internally */ }

impl TypedMetadata {
    /// Create an empty metadata bag.
    pub fn new() -> Self;

    /// Insert a value. If a value of this type already exists, it is replaced
    /// and the old value is returned.
    pub fn insert<T: Send + Sync + 'static>(&mut self, val: T) -> Option<T>;

    /// Get a reference to the stored value of type T, if present.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T>;

    /// Get a mutable reference to the stored value of type T, if present.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T>;

    /// Remove and return the stored value of type T, if present.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T>;

    /// Returns true if a value of type T is stored.
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool;

    /// Number of entries.
    pub fn len(&self) -> usize;

    /// Merge another TypedMetadata into this one.
    /// Keys present in `other` overwrite keys in `self` (last-writer-wins).
    pub fn merge(&mut self, other: TypedMetadata);
}

impl Clone for TypedMetadata { /* deep clone of all stored values */ }
impl Default for TypedMetadata { /* empty */ }
```

**Cloning cost:** `TypedMetadata::clone()` deep-clones every stored value. This is acceptable because metadata bags are typically small (2-5 entries) and contain lightweight types (ids, scores, small vecs). If a stage stores large data in metadata (e.g., a full feature map), it should wrap it in `Arc<T>` so that cloning the metadata bag clones only the Arc, not the data.

**Collision on merge:** `merge()` uses last-writer-wins by `TypeId`. Two stages that both insert `MyFeatureVec` into `stage_artifacts` will collide — the later stage's value wins. Stages that need to coexist should use distinct wrapper types (e.g., `struct DetectorFeatures(Vec<f32>)` vs. `struct TrackerFeatures(Vec<f32>)`). This is the standard type-map pattern and is documented as a requirement for stage authors.

/// Top-level error enum.
pub enum NvError {
    Media(MediaError),
    Stage(StageError),
    Temporal(TemporalError),
    View(ViewError),
    Runtime(RuntimeError),
    Config(ConfigError),
}
```

### `nv-frame`

```rust
pub enum PixelFormat {
    Rgb8,
    Bgr8,
    Rgba8,
    Nv12,
    I420,
    Gray8,
}

/// Immutable, ref-counted frame. The hot type on the pipeline.
/// Clone is cheap (Arc bump). Pixel data is never copied between stages.
pub struct FrameEnvelope {
    pub(crate) inner: Arc<FrameInner>,
}

struct FrameInner {
    feed_id: FeedId,
    seq: u64,                   // monotonic frame counter per feed
    ts: MonotonicTs,
    wall_ts: WallTs,
    width: u32,
    height: u32,
    format: PixelFormat,
    stride: u32,
    data: PixelData,            // enum: Borrowed(*const u8 + len), Owned(Vec<u8>)
    metadata: TypedMetadata,    // stage-injected per-frame metadata
}
```

`PixelData` uses an inner enum:

```rust
enum PixelData {
    /// Zero-copy: pointer into GStreamer-mapped buffer.
    /// The GstBuffer/GstMapInfo are held alive by an opaque PinGuard stored alongside.
    Mapped { ptr: *const u8, len: usize, _guard: PinGuard },
    /// Owned copy — fallback, or for synthetic test frames.
    Owned(Vec<u8>),
}
// SAFETY: Mapped variant is immutable; the guard prevents deallocation.
unsafe impl Send for PixelData {}
unsafe impl Sync for PixelData {}
```

The `PinGuard` type is `pub(crate)` in `nv-media` and holds the `gst::Buffer` + `gst::MappedBuffer` alive. It is erased to an opaque `Box<dyn Any + Send + Sync>` before crossing into `nv-frame`, so `nv-frame` has no source-level dependency on GStreamer.

### `nv-perception`

```rust
/// A single detection within one frame.
#[derive(Clone)]
pub struct Detection {
    pub id: DetectionId,
    pub class_id: u32,
    pub confidence: f32,
    pub bbox: BBox,
    pub embedding: Option<Vec<f32>>,   // re-id / feature vector
    pub metadata: TypedMetadata,
}

/// All detections for one frame.
pub struct DetectionSet {
    pub detections: Vec<Detection>,
}

/// Track lifecycle state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Coasted,    // no observation this frame, predicted
    Lost,       // coasted too long, pending deletion
}

/// One observation of a track in a single frame.
#[derive(Clone)]
pub struct TrackObservation {
    pub ts: MonotonicTs,
    pub bbox: BBox,
    pub confidence: f32,
    pub state: TrackState,
    pub detection_id: Option<DetectionId>,
}

/// A live track.
#[derive(Clone)]
pub struct Track {
    pub id: TrackId,
    pub class_id: u32,
    pub state: TrackState,
    pub current: TrackObservation,
    pub metadata: TypedMetadata,
}

// -- Stage trait --

/// Context provided to every stage invocation.
pub struct StageContext<'a> {
    pub feed_id: FeedId,
    pub frame: &'a FrameEnvelope,
    pub artifacts: &'a PerceptionArtifacts,
    pub temporal: &'a TemporalStoreSnapshot,
    pub view: &'a ViewSnapshot,
    pub metrics: &'a StageMetrics,
}

/// What a stage returns.
pub struct StageOutput {
    /// New or updated detections.
    pub detections: Option<DetectionSet>,
    /// New or updated tracks.
    pub tracks: Option<Vec<Track>>,
    /// Derived signals (generic named scalars/vectors).
    pub signals: Vec<DerivedSignal>,
    /// Arbitrary typed artifacts for downstream stages.
    pub artifacts: TypedMetadata,
}

/// The core user-implementable trait.
///
/// All methods take `&mut self`. The executor holds exclusive ownership of each
/// stage on the feed's dedicated OS thread — stages are never shared across
/// threads or called concurrently within a feed. `&mut self` on `process`
/// allows stages to maintain internal mutable state (model caches, assignment
/// buffers, running statistics) directly, without interior-mutability wrappers.
///
/// Requires `Send + 'static` but NOT `Sync` — stages do not need to be
/// shareable across threads. The executor moves each `Box<dyn Stage>` onto
/// the feed's stage thread at startup and never aliases it.
pub trait Stage: Send + 'static {
    /// Unique name for this stage (used in provenance and metrics).
    fn id(&self) -> StageId;

    /// Process one frame. Must not block on I/O.
    /// Expensive inference should use interior async or pre-warmed models.
    /// `&mut self` is provided because most real stages need to mutate internal
    /// state (tracker assignment matrices, detection caches, counters, etc.).
    fn process(&mut self, ctx: &StageContext) -> Result<StageOutput, StageError>;

    /// Called once when the feed starts. Allocate GPU resources here.
    fn on_start(&mut self) -> Result<(), StageError> { Ok(()) }

    /// Called once on feed shutdown. Release resources here.
    fn on_stop(&mut self) -> Result<(), StageError> { Ok(()) }

    /// Called when view state changes significantly (new ViewEpoch).
    /// Stages that maintain internal state dependent on camera view
    /// should reset or adapt here.
    fn on_view_epoch_change(&mut self, _new_epoch: ViewEpoch) -> Result<(), StageError> { Ok(()) }
}

/// Accumulated outputs of all stages that have run so far for this frame.
pub struct PerceptionArtifacts {
    pub detections: DetectionSet,
    pub tracks: Vec<Track>,
    pub signals: Vec<DerivedSignal>,
    pub stage_artifacts: TypedMetadata,
}
```

### Artifact accumulation semantics

When each stage returns a `StageOutput`, the pipeline executor merges it into the `PerceptionArtifacts` accumulator using these rules:

| Field | Merge behavior | Rationale |
|---|---|---|
| `detections` | **Replace.** If `StageOutput.detections` is `Some(set)`, the accumulator's `DetectionSet` is replaced entirely. If `None`, the previous value is kept. | A pipeline typically has **one** detection stage. If a later stage filters or refines detections, it returns the full, corrected set. Appending would create duplicates. If a downstream stage needs to add detections to an upstream set, it reads the accumulator via `ctx.artifacts.detections`, appends its own, and returns the combined set. |
| `tracks` | **Replace.** If `StageOutput.tracks` is `Some(vec)`, the accumulator's `tracks` is replaced entirely. If `None`, the previous value is kept. | Same rationale as detections. A tracker stage returns the authoritative track set for this frame. A re-id or refinement stage downstream reads the current tracks, modifies metadata or merges identities, and returns the corrected full set. |
| `signals` | **Append.** `StageOutput.signals` is always appended to the accumulator's `signals` vec. | Signals are named and independent. Multiple stages may emit different signals (e.g., "scene_complexity" from one stage, "crowd_density" from another). Appending is natural. Duplicate signal names from different stages are allowed — consumers disambiguate by stage provenance. |
| `stage_artifacts` | **Merge (insert-or-overwrite by TypeId).** Each typed entry from `StageOutput.artifacts` is inserted into the accumulator's `stage_artifacts`. If a key (TypeId) already exists, it is overwritten. | Stage artifacts are typed by `TypeId`. Two stages inserting the same concrete type means the later stage's value wins. This is intentional — if two stages both produce `MyIntermediateResult`, the later one is the refinement. Stages that need to preserve both should use distinct wrapper types. |

The executor applies these rules immediately after each stage's `process()` returns, before calling the next stage. This means stage N+1 always sees the accumulated result of stages 0..N in `ctx.artifacts`.

**Key constraint:** A stage that returns `Some(DetectionSet)` or `Some(Vec<Track>)` is asserting ownership of the full detection/track set. It is not asking the executor to merge — it is declaring "these are the detections/tracks for this frame as of my stage." This is a conscious design choice that avoids the ambiguity of partial-merge semantics.

/// A generic named signal — domain users define the names and semantics.
#[derive(Clone)]
pub struct DerivedSignal {
    pub name: &'static str,
    pub value: SignalValue,
    pub ts: MonotonicTs,
}

pub enum SignalValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Boolean(bool),
    Categorical(&'static str),
}
```

---

## 5. GStreamer integration boundary

### Principle

`nv-media` is the **only** crate that links against `gstreamer-rs`. It exposes:

- `MediaSource` — internal type that owns the GStreamer pipeline.
- `bridge::to_frame_envelope(sample: &gst::Sample) -> Result<FrameEnvelope, MediaError>` — the single crossing point.

`SourceSpec` (the user-facing config enum) lives in `nv-core`, not `nv-media`, to prevent downstream crates from transitively depending on `gstreamer-rs` via the config path. `nv-media` consumes `SourceSpec` to construct its internal pipeline.

### `SourceSpec` (defined in `nv-core`)

```rust
pub enum SourceSpec {
    Rtsp { url: String, transport: RtspTransport },
    File { path: PathBuf, loop_: bool },
    V4l2 { device: String },
    Custom { pipeline_fragment: String },
}

pub enum RtspTransport {
    Tcp,
    UdpUnicast,
}
```

`Custom` is an escape hatch. The library constructs the GStreamer pipeline internally for all other variants.

### Zero-copy bridge

1. `appsink` callback receives `gst::Sample`.
2. `bridge` maps the underlying `gst::Buffer` as read-only.
3. A `PinGuard` struct holds the `gst::MappedBuffer<Readable>`, preventing GStreamer from reclaiming the memory.
4. `PinGuard` is type-erased to `Box<dyn Any + Send + Sync>` and stored inside `PixelData::Mapped`.
5. `FrameEnvelope` is constructed and sent on the bounded channel to the stage executor.
6. When the last `Arc<FrameInner>` drops, the `PinGuard` drops, releasing the GStreamer buffer back to its pool.

This achieves **zero pixel-data copies** from decode → stage processing, while keeping `gstreamer-rs` types fully invisible to downstream crates.

### Reconnection

`nv-media` owns reconnection internally. The strategy is configurable per-source:

```rust
pub struct ReconnectPolicy {
    pub max_attempts: u32,       // 0 = infinite
    pub base_delay: Duration,    // e.g. 1s
    pub max_delay: Duration,     // e.g. 30s
    pub backoff: BackoffKind,    // Exponential or Linear
}
```

During reconnection, the feed pipeline is paused (no frames emitted), and `HealthEvent::SourceReconnecting` is published.

---

## 6. Frame abstraction design

### Goals

- Immutable after construction.
- Cheap to clone (Arc).
- Zero-copy from GStreamer where possible.
- Safe to send across threads.
- Carries enough metadata to be self-describing.

### `FrameEnvelope` public API

```rust
impl FrameEnvelope {
    pub fn feed_id(&self) -> FeedId;
    pub fn seq(&self) -> u64;
    pub fn ts(&self) -> MonotonicTs;
    pub fn wall_ts(&self) -> WallTs;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn format(&self) -> PixelFormat;
    pub fn stride(&self) -> u32;
    pub fn data(&self) -> &[u8];       // borrows pixel data
    pub fn metadata(&self) -> &TypedMetadata;
}
```

No mutable access. No pixel format conversion on the hot path — conversion utilities exist in `nv-frame::convert` but are opt-in and allocate a new `FrameEnvelope`.

### Why normalized coordinates?

`BBox` and `Point2` use `[0, 1]` normalized coordinates relative to frame dimensions. This eliminates resolution-dependency throughout the perception pipeline. Stages that need absolute pixel coordinates can compute them trivially: `px_x = normalized_x * width`.

---

## 7. Stage model design

### Stage execution is linear and synchronous per frame

For each incoming frame on a given feed:

1. A snapshot of the current `TemporalStore` and `ViewState` is taken.
2. A `PerceptionArtifacts` accumulator is initialized (empty).
3. Each stage is called in order with `StageContext` containing the frame, accumulated artifacts, temporal snapshot, and view snapshot.
4. Each stage's `StageOutput` is merged into the accumulator.
5. After all stages complete, the accumulator is written into `TemporalStore` and forwarded to output.

### Why linear, not a DAG?

- A linear pipeline covers the vast majority of real perception workloads (detect → track → classify → signal).
- DAGs introduce scheduling complexity, partial-ordering ambiguity, and error-propagation nightmares that are not justified for this library's scope.
- Users who need branching can compose multiple feeds or fan-out inside a custom stage.

This follows AGENTS.md rule 22: *Do not build a framework inside the framework.*

### Stage contract

- `process()` **must not block on I/O**. If a stage wraps an inference server, it must manage its own connection pool or async bridge internally.
- `process()` takes `&mut self`. The executor owns each stage exclusively on the feed's stage thread. No `Sync` bound is required — stages are never shared across threads.
- `process()` is called on the feed's dedicated stage-execution thread. Stages within a single feed are never called concurrently (no intra-feed parallelism). Different feeds run in parallel.
- A stage that returns `Err(StageError)` causes the frame to be dropped and a `HealthEvent::StageError` to be emitted. The feed continues processing subsequent frames.
- A stage that panics is caught via `catch_unwind`. The feed is restarted. Note: `catch_unwind` requires the stage to be `UnwindSafe`. The executor wraps calls in `AssertUnwindSafe` — stage authors need not add `UnwindSafe` bounds, but panics may leave internal state inconsistent, which is why the feed restarts rather than continuing.

### `on_view_epoch_change`

When the view system detects a significant camera motion discontinuity (PTZ jump, large rotation), all stages are notified. Stages that maintain internal state relative to frame content (e.g., background models, local maps) should reset here.

---

## 8. Temporal state design

### `TemporalStore`

Each feed owns a `TemporalStore`. It is **not** shared between feeds.

```rust
pub struct TemporalStore {
    tracks: HashMap<TrackId, TrackHistory>,
    retention: RetentionPolicy,
    view_epoch: ViewEpoch,
}

pub struct TrackHistory {
    pub track: Track,
    pub observations: VecDeque<TrackObservation>,
    pub trajectory: Trajectory,
    pub first_seen: MonotonicTs,
    pub last_seen: MonotonicTs,
    pub view_epoch_at_creation: ViewEpoch,
}

pub struct RetentionPolicy {
    pub max_track_age: Duration,         // evict tracks older than this
    pub max_observations_per_track: usize, // ring buffer depth
    pub max_concurrent_tracks: usize,    // hard cap, oldest-evicted
}
```

### `Trajectory` and `MotionFeatures`

```rust
pub struct Trajectory {
    pub segments: Vec<TrajectorySegment>,
}

/// A segment is a contiguous run of observations within a single ViewEpoch.
/// When the EpochPolicy produces a Segment decision (ViewEpoch change),
/// the current segment is closed and a new one begins.
/// When the policy produces Compensate, the segment continues but
/// existing positions are transformed.
/// When the policy produces Degrade or Continue, the segment continues
/// with no coordinate changes.
///
/// Every segment records why it was opened and (once complete) why
/// it was closed, via the SegmentBoundary enum. This makes trajectory
/// continuity fully auditable without guesswork.
pub struct TrajectorySegment {
    pub view_epoch: ViewEpoch,
    pub points: Vec<TrajectoryPoint>,
    pub motion: MotionFeatures,
    /// Why this segment was opened. Records the causal event so
    /// downstream consumers can audit trajectory continuity decisions.
    pub opened_by: SegmentBoundary,
    /// How this segment was closed. None if the segment is still active.
    /// The next segment's `opened_by` will be the complement of this
    /// (e.g., closed by Segment → next opened by EpochChange).
    pub closed_by: Option<SegmentBoundary>,
    /// If this segment has been compensated (coordinates transformed
    /// mid-segment due to detected camera motion), the cumulative
    /// compensation transform is stored here.
    pub compensation: Option<AffineTransform2D>,
    /// Number of times compensation was applied within this segment.
    /// Nonzero only if compensation occurred. Useful for auditing
    /// whether coordinates in this segment have been transformed.
    pub compensation_count: u32,
}

/// Why a trajectory segment boundary (open or close) occurred.
/// Provides temporal auditability — every segment boundary is explained.
pub enum SegmentBoundary {
    /// First segment for this track (track was newly created).
    TrackCreated,
    /// EpochPolicy returned Segment, causing a ViewEpoch change.
    EpochChange { from_epoch: ViewEpoch, to_epoch: ViewEpoch },
    /// Feed was restarted; temporal state was cleared and rebuilt.
    FeedRestart,
    /// Track was lost (coasted too long, evicted by retention policy).
    TrackLost,
    /// Track ended normally (e.g., object left the scene).
    TrackEnded,
}

pub struct TrajectoryPoint {
    pub ts: MonotonicTs,
    pub position: Point2,  // centroid of bbox, normalized
    pub bbox: BBox,
}

pub struct MotionFeatures {
    pub displacement: f32,       // total path length in normalized coords
    pub net_displacement: f32,   // straight-line start→end
    pub mean_speed: f32,         // displacement / elapsed time
    pub max_speed: f32,
    pub direction: Option<f32>,  // radians, None if stationary
    pub is_stationary: bool,     // below configurable speed threshold
}
```

### Snapshot

Stages receive a `TemporalStoreSnapshot` — a read-only, cheaply-cloned view of the current temporal state. The store itself is mutated only by the pipeline executor between frames (after all stages have run), never during stage execution.

```rust
pub struct TemporalStoreSnapshot {
    inner: Arc<TemporalStoreInner>, // snapshot of track map at frame boundary
}
```

### Snapshotting strategy

The snapshot is taken once per frame, before stage execution begins. The strategy is **clone-and-arc**: the executor clones the current `TemporalStoreInner` (the full `HashMap<TrackId, TrackHistory>`) into a new `Arc`.

**Cost analysis:** With N active tracks, each holding an observations `VecDeque` (bounded by `max_observations_per_track`, typically 30-100) and a `Trajectory` (bounded by segment count and eviction), the clone cost is proportional to N * observations_per_track. For typical deployments (50-200 active tracks, 50 observations each), this is ~10K-20K small struct copies per frame — well under 100μs on modern hardware.

**Why not a persistent data structure:** Persistent/immutable data structures (e.g., `im::HashMap`) eliminate the clone but add per-access overhead (pointer chasing, deeper trees) that affects every stage's temporal state lookup on every frame. The snapshot is taken once per frame; stage lookups happen many times per frame. Optimizing the uncommon operation (snapshot) at the expense of the common one (lookup) is the wrong tradeoff.

**Why not double-buffering:** Double-buffering (two `TemporalStoreInner` instances, swap after write) would eliminate cloning but requires the executor to know that no stage holds a reference to the old buffer before overwriting. This is achievable but adds lifecycle complexity. It is a valid v1 optimization if profiling shows snapshot cloning is a bottleneck.

**v0 decision:** Start with clone-and-arc. Profile under realistic track counts. If snapshot cloning appears in flamegraphs, switch to double-buffering. The `TemporalStoreSnapshot` type is opaque, so the strategy can change without public API impact.

### Eviction

The `TemporalStore` runs eviction **before** processing each frame:
1. Tracks in `Lost` state with `last_seen` older than `max_track_age` are purged.
2. If `max_concurrent_tracks` is exceeded, the oldest `Lost` tracks are purged first, then oldest `Coasted`.
3. Per-track observation ring buffers are truncated to `max_observations_per_track`.

---

## 9. PTZ/view-state design

The view system must handle three fundamentally different camera scenarios correctly:

1. **Fixed cameras** — no motion, no telemetry. Single epoch forever. Trivial.
2. **PTZ cameras with telemetry** — motion is known via ONVIF or serial. Epoch and continuity decisions use telemetry.
3. **Moving/PTZ cameras without telemetry** — motion is inferred from video (optical flow, feature matching, homography estimation). Epoch and continuity decisions use confidence-scored estimates.

The design must not privilege case 2 over case 3. Both are first-class.

### Core types

```rust
/// Opaque monotonic epoch counter. Incremented when the view system
/// (via its configured EpochPolicy) determines a discontinuity occurred
/// that warrants segmenting temporal state.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ViewEpoch(u64);

/// Monotonic version counter for the view-state itself.
/// Incremented on every ViewState update (even within the same epoch).
/// Allows consumers to test "is my cached transform still current?"
/// without deep-comparing the full ViewState.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ViewVersion(u64);

/// The current best estimate of the camera's view.
pub struct ViewState {
    pub epoch: ViewEpoch,
    pub version: ViewVersion,
    pub motion: CameraMotionState,
    pub motion_source: MotionSource,
    pub transition: TransitionPhase,
    pub ptz: Option<PtzTelemetry>,
    pub global_transform: Option<GlobalTransformEstimate>,
    pub validity: ContextValidity,
    pub stability_score: f32,       // [0.0, 1.0] — 1.0 = fully stable
}

/// Read-only, cheaply-cloneable snapshot of ViewState.
/// Created once per frame by the view system and shared with all stages
/// via `StageContext.view`. Clone is an Arc bump.
///
/// Stages and output consumers receive ViewSnapshot, not ViewState.
/// This prevents stages from accidentally mutating the view system's
/// internal state and makes the snapshot semantics explicit.
pub struct ViewSnapshot {
    inner: Arc<ViewState>,
}

impl ViewSnapshot {
    pub fn epoch(&self) -> ViewEpoch;
    pub fn version(&self) -> ViewVersion;
    pub fn motion(&self) -> &CameraMotionState;
    pub fn motion_source(&self) -> &MotionSource;
    pub fn transition(&self) -> TransitionPhase;
    pub fn ptz(&self) -> Option<&PtzTelemetry>;
    pub fn global_transform(&self) -> Option<&GlobalTransformEstimate>;
    pub fn validity(&self) -> &ContextValidity;
    pub fn stability_score(&self) -> f32;
}

pub enum CameraMotionState {
    /// Camera is not moving. Coordinates are stable.
    Stable,
    /// Camera is actively moving. Coordinates are shifting.
    Moving {
        angular_velocity: Option<f32>,
        /// Estimated frame-to-frame displacement magnitude (normalized coords).
        displacement: Option<f32>,
    },
    /// Camera motion state is unknown (no telemetry, no estimator, or
    /// estimator confidence is below threshold).
    Unknown,
}

/// How the current motion state was determined.
/// Critical for downstream trust decisions — telemetry-sourced motion
/// is more trustworthy than inferred motion from noisy optical flow.
pub enum MotionSource {
    /// From PTZ telemetry (ONVIF, serial, etc.)
    Telemetry,
    /// From video-based egomotion estimation (optical flow, homography, etc.)
    Inferred { confidence: f32 },
    /// From a user-supplied external source.
    External,
    /// No motion information available for this frame.
    /// Only possible when CameraMode::Observed is set but the provider
    /// returned an empty report. The view system treats this as Unknown
    /// motion, not as stable — see CameraMode documentation.
    None,
}

/// Declared camera installation mode. Set on FeedConfig.
/// This is the primary safety mechanism against silently treating
/// a moving camera as fixed.
pub enum CameraMode {
    /// Camera is physically fixed (bolted mount, no PTZ, no gimbal).
    /// The view system is bypassed entirely: CameraMotionState::Stable,
    /// MotionSource::None, ViewEpoch held constant within a feed session.
    /// No ViewStateProvider needed. This is the only mode where the
    /// library assumes stability.
    ///
    /// **Epoch on restart:** A feed restart (see §13) increments ViewEpoch
    /// even for Fixed feeds. This is deliberate — a restart clears the
    /// TemporalStore, invalidating all prior track/trajectory state.
    /// The epoch increment signals this discontinuity to any consumer
    /// that cached data across frames. Within a single uninterrupted
    /// session, ViewEpoch never changes for Fixed feeds.
    Fixed,
    /// Camera may move (PTZ, gimbal, handheld, vehicle-mounted, drone, etc.)
    /// A ViewStateProvider is REQUIRED. If the provider returns no data
    /// for a frame, the view system defaults to CameraMotionState::Unknown
    /// (not Stable), and ContextValidity::Degraded. This prevents silent
    /// false-stability.
    Observed,
}

/// Where in a camera motion transition the current frame sits.
/// Enables downstream logic to distinguish "just started moving"
/// from "mid-move" from "just settled."
pub enum TransitionPhase {
    /// No transition in progress. Camera is stable or has been stable.
    Settled,
    /// Camera has begun moving (first frame of detected motion).
    MoveStart,
    /// Camera is mid-move.
    Moving,
    /// Camera has stopped moving (first frame of detected stability after motion).
    MoveEnd,
}
```

### TransitionPhase state machine

`TransitionPhase` is a strict state machine. The view system updates it every frame based on the previous phase and the current frame's `CameraMotionState`. Invalid transitions are a library bug.

**State transition table:**

| Previous phase | Current motion state | Next phase | Notes |
|---|---|---|---|
| `Settled` | `Stable` | `Settled` | No change. Camera remains still. |
| `Settled` | `Moving` / `Unknown` | `MoveStart` | Motion detected. First frame of a new motion event. |
| `MoveStart` | `Moving` / `Unknown` | `Moving` | Camera continues to move. |
| `MoveStart` | `Stable` | `MoveEnd` | Camera moved for exactly one frame then stopped (rare but valid — e.g., a single-frame jitter). |
| `Moving` | `Moving` / `Unknown` | `Moving` | Camera still moving. |
| `Moving` | `Stable` | `MoveEnd` | Motion ended. First frame of stability after motion. |
| `MoveEnd` | `Stable` | `Settled` | Camera confirmed stable. Transition complete. |
| `MoveEnd` | `Moving` / `Unknown` | `MoveStart` | Camera resumed motion immediately after settling for one frame. |

**Key properties:**

- `MoveStart` and `MoveEnd` each last **exactly one frame**. They are edge-triggered signals. Downstream consumers that need to react to the start or end of camera motion should check for these phases.
- `Settled` and `Moving` are level-triggered — they persist across multiple frames.
- `Unknown` motion state (e.g., when the provider returns empty data) is treated as `Moving` for transition purposes. This is the conservative choice: if we don't know whether the camera moved, we assume it did and let the `EpochPolicy` decide how to react. Stages see the `MotionSource::None` on `ViewState` and can apply their own logic.
- On feed start, the initial phase is `Settled` (camera is assumed stable until proven otherwise).
- On feed restart (after `FeedRestart` boundary), the phase resets to `Settled`.

pub struct PtzTelemetry {
    pub pan: f32,                // degrees
    pub tilt: f32,               // degrees
    pub zoom: f32,               // normalized [0, 1] where 1 = max zoom
    pub ts: MonotonicTs,
}

pub struct GlobalTransformEstimate {
    pub transform: AffineTransform2D,
    pub confidence: f32,
    pub method: TransformEstimationMethod,
    /// ViewVersion at which this transform was computed.
    /// Consumers can compare against current ViewState.version
    /// to know if this estimate is stale.
    pub computed_at: ViewVersion,
}

pub enum TransformEstimationMethod {
    PtzModel,        // computed from PTZ telemetry + camera model
    FeatureMatching, // computed from inter-frame feature matching
    External,        // provided by user
}

pub enum ContextValidity {
    /// View is stable; all temporal state within this epoch is valid.
    Valid,
    /// View is changing; temporal state is degraded.
    /// Tracks/trajectories from earlier in this epoch may be unreliable
    /// for direct spatial comparison with current positions.
    Degraded { reason: DegradationReason },
    /// View has changed so much that prior context is invalid.
    /// A new ViewEpoch has been (or should be) opened, depending
    /// on the configured EpochPolicy.
    Invalid,
}

pub enum DegradationReason {
    PtzMoving,
    LargeJump,
    ZoomChange,
    OcclusionOrBlur,
    InferredMotionLowConfidence,
    Unknown,
}
```

### Motion input: ViewStateProvider trait

The `ViewStateProvider` trait is the single integration point for all camera motion information — telemetry, inferred, or externally computed. It returns a `MotionReport`, not raw PTZ telemetry, so it works for all three camera scenarios.

```rust
/// What the view system receives from the provider each frame.
pub struct MotionReport {
    /// PTZ telemetry, if available.
    pub ptz: Option<PtzTelemetry>,
    /// Frame-to-frame transform estimate (e.g., from optical flow
    /// or homography). This is the primary egomotion signal when
    /// PTZ telemetry is absent.
    pub frame_transform: Option<GlobalTransformEstimate>,
    /// Whether the provider believes the camera is moving.
    /// If None, the view system infers motion from ptz deltas
    /// or frame_transform magnitude.
    pub motion_hint: Option<CameraMotionState>,
}
```

**MotionSource derivation:** `MotionReport` deliberately does **not** carry a `MotionSource` field. The view system derives `MotionSource` from the report contents using these rules:

| Report contents | Derived `MotionSource` |
|---|---|
| `ptz` is `Some` | `Telemetry` |
| `ptz` is `None`, `frame_transform` is `Some` with `method: FeatureMatching` | `Inferred { confidence }` (confidence from `GlobalTransformEstimate.confidence`) |
| `ptz` is `None`, `frame_transform` is `Some` with `method: External` | `External` |
| `ptz` is `None`, `frame_transform` is `Some` with `method: PtzModel` | Invalid — `PtzModel` without `ptz` telemetry is a provider bug. The view system logs a warning and falls through to `External` (conservative: higher trust than `Inferred`, lower than `Telemetry`). |
| Both `ptz` and `frame_transform` are `Some` | `Telemetry` (telemetry is authoritative; transform is supplementary) |
| Both are `None`, `motion_hint` is `Some` | `External` |
| All fields are `None` | `None` |

The derivation inspects `GlobalTransformEstimate.method` to distinguish video-inferred transforms from externally-supplied ones. This prevents an external transform provider from being misclassified as `Inferred`, which would cause downstream trust gating (e.g., "only act on Telemetry or External sources") to apply the wrong confidence model.

This eliminates the redundancy of asking the provider to state something the fields already encode, and removes the possibility of inconsistency (e.g., provider says `Telemetry` but `ptz` is `None`). The derived `MotionSource` is stored on `ViewState` for downstream consumption.

/// User-implementable. Provides camera motion information each frame.
///
/// Implementations fall into three categories:
/// - Telemetry providers: poll ONVIF/serial, return ptz field populated.
/// - Egomotion providers: run optical flow or feature matching on the frame,
///   return frame_transform field populated.
/// - External providers: receive transforms or motion hints from an outside system.
///
/// The view system derives `MotionSource` from which fields are populated
/// (see MotionSource derivation table above). Providers do not set it directly.
///
/// Required when `CameraMode::Observed` is configured on the feed.
/// Not used when `CameraMode::Fixed` — fixed feeds bypass the
/// view system entirely and keep CameraMotionState::Stable forever.
pub trait ViewStateProvider: Send + Sync + 'static {
    /// Called once per frame. Return the current motion report.
    ///
    /// For egomotion providers that need pixel data, the frame
    /// is available in the report context.
    fn poll(&self, ctx: &MotionPollContext) -> MotionReport;
}

/// Context given to ViewStateProvider::poll.
pub struct MotionPollContext<'a> {
    pub ts: MonotonicTs,
    pub frame: &'a FrameEnvelope,
    pub previous_view: &'a ViewState,
}
```

### Hot-path implications of `ViewStateProvider::poll`

`poll()` is called on the stage thread, synchronously, **before** any stage executes for each frame. Its latency is added directly to every frame's pipeline latency. This is an intentional design choice — the view state must be known before stages run, because stages receive `ViewSnapshot` in their context.

**Implications for provider implementations:**

- **Telemetry providers** (ONVIF, serial): These should pre-fetch telemetry asynchronously (e.g., on a background thread or tokio task) and have the latest value ready for `poll()` to return immediately. `poll()` should be a non-blocking read of the latest cached telemetry, not a network round-trip.
- **Egomotion providers** (optical flow, feature matching): These perform computation on the frame (typically 1-5ms). This cost is unavoidable and is visible in the `nv.feed.pipeline_latency_ns` histogram. If egomotion computation is too expensive, the provider can run it asynchronously and return the previous frame's result with a one-frame lag — this is a valid tradeoff that the provider author makes, not the library.
- **External providers**: Should return pre-computed data. `poll()` should be a channel read or atomic load.

The library does not provide an async variant of `poll()` because the stage thread is synchronous (OS thread running blocking FFI). An async poll would require bridging to a tokio runtime just to call one function, adding complexity for no benefit. Providers that need async internally should manage their own background tasks and have `poll()` return the latest available result.

### Discontinuity policy: EpochPolicy trait

The original design hard-coded "increment epoch + segment all trajectories" on every discontinuity. This is wrong for cameras that move frequently but return to known positions (PTZ presets, auto-tracking). The `EpochPolicy` trait lets the view system ask a configurable strategy what to do when motion is detected.

```rust
/// Returned by EpochPolicy to tell the view system what to do
/// when a discontinuity is detected.
pub enum EpochDecision {
    /// No change. View is continuous despite motion.
    /// Example: small PTZ adjustment within tolerance.
    Continue,
    /// Degrade context validity but keep the current epoch.
    /// Tracks and trajectories remain in the same epoch
    /// but ContextValidity is set to Degraded.
    /// Example: camera is moving slowly, tracking may still work.
    Degrade { reason: DegradationReason },
    /// Degrade and compensate: keep the same epoch, degrade validity,
    /// but apply the given transform to adapt existing track positions
    /// to the new view. The temporal store will transform active track
    /// bboxes/positions in the current segment rather than segmenting.
    /// Example: high-confidence homography available, small PTZ move.
    Compensate {
        reason: DegradationReason,
        transform: AffineTransform2D,
    },
    /// Full segmentation. Increment epoch, close trajectory segments,
    /// notify stages.
    /// Example: large PTZ jump, zoom change, scene cut.
    Segment,
}

/// User-implementable. Decides what happens when the view system
/// detects a camera motion event.
///
/// The library provides `DefaultEpochPolicy` which uses configurable
/// thresholds. Users building on PTZ-heavy deployments can supply
/// more nuanced policies.
pub trait EpochPolicy: Send + Sync + 'static {
    fn decide(&self, ctx: &EpochPolicyContext) -> EpochDecision;
}

/// Context passed to EpochPolicy::decide.
pub struct EpochPolicyContext<'a> {
    pub previous_view: &'a ViewState,
    pub current_report: &'a MotionReport,
    pub motion_state: CameraMotionState,
    /// How long the camera has been in the current motion state.
    pub state_duration: Duration,
}
```

### DefaultEpochPolicy

The library ships a `DefaultEpochPolicy` that covers common cases:

```rust
pub struct DefaultEpochPolicy {
    /// Pan/tilt delta (degrees) above which a PTZ move triggers Segment.
    /// Below this, triggers Degrade. Default: 15.0.
    pub segment_angle_threshold: f32,
    /// Zoom ratio change above which a zoom move triggers Segment. Default: 0.3.
    pub segment_zoom_threshold: f32,
    /// Inferred-motion displacement (normalized coords) above which
    /// triggers Segment. Default: 0.25.
    pub segment_displacement_threshold: f32,
    /// Minimum confidence for a Compensate decision instead of Segment
    /// when a transform is available. Default: 0.8.
    pub compensate_min_confidence: f32,
    /// If true, small motions below segment thresholds produce Degrade
    /// instead of Continue. Default: true.
    pub degrade_on_small_motion: bool,
}

impl Default for DefaultEpochPolicy { /* reasonable defaults */ }
```

Users who need different behavior (e.g., "always compensate if transform confidence > 0.9", or "never segment, always degrade") implement `EpochPolicy` directly.

### View-state integration into the pipeline

1. Before stage execution for each frame, the view system runs.
2. If `CameraMode::Fixed`, steps 3-6 are skipped. The ViewState is always `Stable` / `Valid` / epoch 0.
3. If `CameraMode::Observed`, the `ViewStateProvider` is polled with `MotionPollContext` (which includes the frame for egomotion providers).
4. The view system computes `CameraMotionState` from the `MotionReport`:
   - If `motion_hint` is `Some`, use it directly.
   - If `ptz` is `Some`, compute from pan/tilt/zoom deltas vs. previous.
   - If `frame_transform` is `Some`, compute from transform displacement magnitude and confidence.
   - If the report is empty (no ptz, no frame_transform, no motion_hint): `CameraMotionState::Unknown` and `ContextValidity::Degraded { reason: Unknown }`. **The library never infers stability from absence of data on an Observed feed.**
5. The view system determines `TransitionPhase` from the previous phase and the current `CameraMotionState` using the state transition table defined above.
6. If motion is detected (state is `Moving` or transitioned to `MoveEnd` with a large accumulated delta), the configured `EpochPolicy` is consulted.
7. Based on the `EpochDecision`:
   - **`Continue`**: No temporal state changes. `ContextValidity::Valid`.
   - **`Degrade`**: `ContextValidity::Degraded`. No epoch change. No trajectory segmentation.
   - **`Compensate`**: `ContextValidity::Degraded`. No epoch change. The `TemporalStore` applies the transform to all active track positions in the current trajectory segment (see *Compensation transform semantics* below).
   - **`Segment`**: `ViewEpoch` is incremented. `on_view_epoch_change` is called on all stages. `TemporalStore` closes trajectory segments for all active tracks and opens new ones.
8. `ViewVersion` is incremented on any ViewState change (even within the same epoch).
9. `stability_score` is updated: 1.0 when `Settled`, decays toward 0.0 during `Moving` proportional to displacement magnitude, recovers toward 1.0 after `MoveEnd` over a configurable settling window.
10. The `ViewSnapshot` (read-only clone of `ViewState`) is included in every `StageContext`.

### Compensation transform semantics

When an `EpochDecision::Compensate { transform }` is applied, the `TemporalStore` must transform existing `BBox` and `Point2` coordinates in the current trajectory segment. This raises a geometric problem: applying an affine transform to an axis-aligned bounding box (AABB) generally produces a rotated quadrilateral, not an AABB.

**Transform application rules:**

1. **`Point2` (centroids, trajectory points):** Transform directly. `Point2` after affine transform remains a `Point2`. No precision loss.

2. **`BBox` (axis-aligned bounding boxes):** Transform all four corners of the AABB through the affine transform, then compute the **axis-aligned bounding box of the transformed corners** (the AABB envelope). This is the standard "transform-and-re-bound" approach.

   ```rust
   // Pseudocode for BBox compensation:
   fn compensate_bbox(bbox: BBox, transform: &AffineTransform2D) -> BBox {
       let corners = [
           transform.apply(Point2 { x: bbox.x_min, y: bbox.y_min }),
           transform.apply(Point2 { x: bbox.x_max, y: bbox.y_min }),
           transform.apply(Point2 { x: bbox.x_max, y: bbox.y_max }),
           transform.apply(Point2 { x: bbox.x_min, y: bbox.y_max }),
       ];
       BBox {
           x_min: corners.iter().map(|c| c.x).min(),
           y_min: corners.iter().map(|c| c.y).min(),
           x_max: corners.iter().map(|c| c.x).max(),
           y_max: corners.iter().map(|c| c.y).max(),
       }
   }
   ```

3. **Precision degradation:** Re-bounding a rotated rectangle as an AABB grows the box. After N compensations, this growth compounds. The `TrajectorySegment.compensation_count` field tracks how many times compensation has been applied. Consumers that depend on tight bounding boxes should check this counter and treat high-compensation-count segments with reduced confidence.

4. **Cumulative transform:** The `TrajectorySegment.compensation` field stores the **cumulative** composed transform (all compensations within the segment composed together), not just the latest. This means a consumer can apply the inverse of `compensation` to recover the original (pre-compensation) coordinates if needed, subject to floating-point precision.

5. **Compensation ceiling:** The `DefaultEpochPolicy` should refuse to produce `Compensate` when the transform involves large rotation (> ~5 degrees) or significant scale change (> ~10%), because AABB re-bounding under large rotations produces boxes that are much larger than the original object. In such cases, `Segment` is the correct decision. This is enforced by the policy, not the temporal store — the store applies whatever transform it receives.

6. **What is NOT compensated:** Detection confidences, embeddings, and class IDs are not modified by compensation. Only spatial fields (`BBox` and `Point2`) are transformed. `MotionFeatures` are recomputed from the transformed trajectory points after compensation.

### Why ViewEpoch still matters

`Compensate` and `Degrade` reduce the frequency of epoch changes but do not eliminate the concept. When compensation is impossible (no transform, or confidence too low) and the view has changed substantially, `Segment` is the correct and only safe response. Epoch segmentation remains the firewall against silently mixing incompatible coordinates.

### Why ViewVersion?

`ViewVersion` is strictly finer-grained than `ViewEpoch`. It increments on every frame where *anything* in the ViewState changed — including small PTZ adjustments that don't trigger an epoch change. Consumers that cache data relative to a specific view (e.g., a stage that builds a local occupancy grid) can cheaply test `my_cached_version == current_view.version` to invalidate without deep-comparing the full `ViewState`.

### Why MotionSource on ViewState?

A downstream analytics layer may reasonably apply different trust thresholds to telemetry-sourced vs. inferred-sourced motion. For example: "gate counting is reliable when motion_source is Telemetry with Known PTZ preset, but should be paused when motion_source is Inferred with confidence < 0.5." Without `MotionSource`, consumers would have to reverse-engineer this from the presence/absence of `ptz` and `global_transform` fields. That is fragile. Making it explicit is cheap and directly supports auditability.

### View-bound context: `ViewBoundContext`

Downstream users often associate calibration data, spatial regions, transforms, or reference frames with a specific camera view. When the camera moves, these associations become stale or invalid. Without an explicit model for this, every consumer must independently track "was this calibration computed under the current view?" — leading to subtle staleness bugs.

`ViewBoundContext` is a lightweight binding model that pairs user-supplied context data with the `ViewVersion` and `ViewEpoch` at which it was valid.

```rust
/// A piece of user-supplied context data that is bound to a specific
/// camera view. The library does not interpret the contents — it only
/// tracks whether the binding is still valid under the current view.
pub struct ViewBoundContext<T> {
    /// The user's data (calibration, region set, reference transform, etc.)
    pub data: T,
    /// The ViewVersion at which this data was created or last confirmed valid.
    pub bound_at: ViewVersion,
    /// The ViewEpoch at which this data was created.
    pub bound_epoch: ViewEpoch,
}

impl<T> ViewBoundContext<T> {
    /// Create a new binding at the current view state.
    pub fn bind(data: T, view: &ViewState) -> Self;

    /// Check whether this binding is still valid under the given view state.
    pub fn validity(&self, current: &ViewState) -> BoundContextValidity;

    /// Rebind this context to the current view (caller asserts data is still valid).
    pub fn rebind(&mut self, view: &ViewState);
}

pub enum BoundContextValidity {
    /// Same ViewVersion — context is exactly current.
    Current,
    /// Same ViewEpoch but different ViewVersion — view has changed within
    /// the epoch (small motions, compensations). Context may be approximately
    /// valid but should be checked. Includes how many versions have elapsed.
    StaleWithinEpoch { versions_behind: u64 },
    /// Different ViewEpoch — view has changed fundamentally.
    /// Context should be considered invalid unless the user can re-derive it.
    InvalidAcrossEpoch { epochs_behind: u64 },
}
```

`ViewBoundContext` is a utility type, not part of the pipeline's data flow. Stages and output consumers use it to manage their own view-dependent data. The library provides it because the `ViewVersion`/`ViewEpoch` comparison logic is easy to get wrong, and centralizing it eliminates a class of bugs.

**Example usage in a stage:**

```rust
struct MyCalibrationStage {
    calibration: Option<ViewBoundContext<CalibrationData>>,
}

impl Stage for MyCalibrationStage {
    fn process(&mut self, ctx: &StageContext) -> Result<StageOutput, StageError> {
        if let Some(ref cal) = self.calibration {
            match cal.validity(&ctx.view) {
                BoundContextValidity::Current => { /* use as-is */ }
                BoundContextValidity::StaleWithinEpoch { .. } => { /* use with caution */ }
                BoundContextValidity::InvalidAcrossEpoch { .. } => {
                    // Recompute calibration and rebind.
                    // With &mut self, we can update self.calibration directly.
                    self.calibration = Some(ViewBoundContext::bind(
                        self.recompute_calibration(ctx),
                        ctx.view,
                    ));
                }
            }
        }
        // ...
    }
}
```

This keeps the core ViewState lean while giving downstream code an explicit, auditable mechanism for managing view-dependent data lifecycle.

---

## 10. Output/provenance model

### `OutputEnvelope`

```rust
pub struct OutputEnvelope {
    pub feed_id: FeedId,
    pub frame_seq: u64,
    pub ts: MonotonicTs,
    pub wall_ts: WallTs,
    pub detections: DetectionSet,
    pub tracks: Vec<Track>,
    pub signals: Vec<DerivedSignal>,
    pub view: ViewState,
    pub provenance: Provenance,
    pub metadata: TypedMetadata,
    /// Present only when `FrameInclusion::Always` is set on the feed.
    pub frame: Option<FrameEnvelope>,
}
```

### `Provenance`

```rust
pub struct Provenance {
    pub stages: Vec<StageProvenance>,
    pub view_provenance: ViewProvenance,
    pub frame_receive_ts: MonotonicTs,
    pub pipeline_complete_ts: MonotonicTs,
    pub total_latency: Duration,
}

pub struct StageProvenance {
    pub stage_id: StageId,
    pub start_ts: MonotonicTs,
    pub end_ts: MonotonicTs,
    pub latency: Duration,
    pub result: StageResult,
}

pub enum StageResult {
    /// Stage completed successfully.
    Ok,
    /// Stage failed on this frame. The frame was dropped.
    Error(StageOutcomeCategory),
    /// Stage opted out for this frame (e.g., frame too similar to previous).
    Skipped,
}

/// Typed failure category for stage provenance.
/// Avoids stringly-typed errors in structured output while keeping the
/// enum small enough that consumers can match exhaustively.
/// The full diagnostic detail remains in tracing logs — this is
/// a summary for programmatic filtering, alerting, and dashboarding.
pub enum StageOutcomeCategory {
    /// Inference or computation failed (model error, NaN output, etc.)
    ProcessingFailed,
    /// Stage ran out of a resource (GPU OOM, buffer limit, etc.)
    ResourceExhausted,
    /// Model or external dependency could not be loaded or contacted.
    DependencyUnavailable,
    /// Stage panicked (caught by executor). Feed will restart.
    Panic,
    /// A category not covered above. Carries a static tag for filtering.
    /// This is NOT a free-form string — it is a short, stable identifier
    /// chosen by the stage author (e.g., "calibration_stale", "roi_invalid").
    Other { tag: &'static str },
}

/// Per-frame provenance of the view system's decisions.
/// Consumers can audit exactly why the view system made
/// the choices it did for any given frame.
pub struct ViewProvenance {
    /// The motion source used for this frame's view decision.
    pub motion_source: MotionSource,
    /// The epoch decision made by the EpochPolicy this frame.
    /// None if the view system was not consulted (no motion detected).
    pub epoch_decision: Option<EpochDecision>,
    /// The transition phase at this frame.
    pub transition: TransitionPhase,
    /// The stability score at this frame.
    pub stability_score: f32,
    /// ViewEpoch at this frame.
    pub epoch: ViewEpoch,
    /// ViewVersion at this frame.
    pub version: ViewVersion,
}
```

Every output carries full provenance of which stages ran, how long each took, whether any errored, and exactly what the view system decided and why. This makes debugging production perception and PTZ-related issues tractable without reverse-engineering trust from partial fields.

### `OutputSink` trait

```rust
/// User-implementable. Receives structured outputs from the pipeline.
///
/// `emit` receives an `Arc<OutputEnvelope>` — the same Arc that is
/// broadcast to subscribers, so zero cloning is needed when multiple
/// consumers share the same output.
pub trait OutputSink: Send + Sync + 'static {
    fn emit(&self, output: Arc<OutputEnvelope>);
}
```

`emit` is deliberately not `async` and not fallible. The per-feed sink worker calls `emit` on a dedicated thread, so a slow sink cannot block the perception pipeline. If the user's sink needs async I/O, it should internally buffer/channel. If it fails, it should log and drop.

---

## 11. Concurrency model

### Thread architecture

```
┌──────────────────────────────────────────────────────┐
│  Runtime                                             │
│                                                      │
│   ┌─────────────┐   ┌─────────────┐                 │
│   │  Feed A      │   │  Feed B      │    ...         │
│   │              │   │              │                 │
│   │  ┌────────┐  │   │  ┌────────┐  │                │
│   │  │GstThrd │  │   │  │GstThrd │  │                │
│   │  │(decode)│  │   │  │(decode)│  │                │
│   │  └───┬────┘  │   │  └───┬────┘  │                │
│   │      │ch(4)  │   │      │ch(4)  │                │
│   │  ┌───▼────┐  │   │  ┌───▼────┐  │                │
│   │  │StgThrd │  │   │  │StgThrd │  │                │
│   │  │(stages)│  │   │  │(stages)│  │                │
│   │  └───┬────┘  │   │  └───┬────┘  │                │
│   │      │ch(2)  │   │      │ch(2)  │                │
│   │  ┌───▼────┐  │   │  ┌───▼────┐  │                │
│   │  │OutThrd │  │   │  │OutThrd │  │                │
│   │  └────────┘  │   │  └────────┘  │                │
│   └─────────────┘   └─────────────┘                 │
│                                                      │
│   ┌─────────────┐                                    │
│   │ Supervisor   │  (monitors feed health,           │
│   │ task         │   triggers restarts)              │
│   └─────────────┘                                    │
└──────────────────────────────────────────────────────┘
```

- **GstThread**: Owned by `nv-media`. Runs the GStreamer main loop and appsink callback. Pushes `FrameEnvelope` into a bounded channel.
- **StgThread**: Per-feed dedicated OS thread (not a tokio task — stages may call blocking FFI like TensorRT or ONNX Runtime). Pulls frames, runs stages sequentially, writes temporal store, pushes `OutputEnvelope`.
- **OutThread**: Per-feed lightweight thread or task. Calls `OutputSink::emit`. Isolates output latency from the perception hot path.
- **Supervisor**: Single async task. Receives `HealthEvent`s. Triggers restarts per policy.

### Why OS threads, not async tasks for stages?

Perception stages commonly call into native inference libraries (TensorRT, ONNX Runtime, OpenCV DNN, libtorch) that are blocking and not async-aware. Running these on an async executor stalls other tasks. Dedicated OS threads avoid this entirely.

The async runtime (tokio) is used for supervision, health monitoring, metrics export, and source management — not for perception hot paths.

---

## 12. Backpressure strategy

### Bounded channels everywhere

| Channel | Capacity | Drop policy |
|---|---|---|
| GstThread → StgThread | Configurable (default: 4) | Configurable: `DropOldest`, `DropNewest`, `Block` |
| StgThread → OutThread | Configurable (default: 2) | `DropOldest` (output must never block perception) |

### `BackpressurePolicy`

```rust
pub enum BackpressurePolicy {
    /// Drop the oldest frame in the queue to make room. Default.
    DropOldest { queue_depth: usize },
    /// Drop the incoming frame if the queue is full.
    DropNewest { queue_depth: usize },
    /// Block the producer until space is available.
    /// Use with caution — can cause GStreamer buffer buildup.
    Block { queue_depth: usize },
}
```

### Why `DropOldest` is the default

In real-time perception, processing a stale frame is worse than skipping it. `DropOldest` ensures stages always see the most recent available frame. The number of dropped frames is tracked in `FeedMetrics`.

### Memory bounding

Frame pixel data is held by `Arc`. Dropped frames release their Arc reference immediately. With `DropOldest(4)`, at most 5 frames of pixel data are alive per feed (4 in queue + 1 being processed). This provides a hard memory ceiling.

---

## 13. Restart/shutdown semantics

### Feed restart

```rust
pub struct RestartPolicy {
    pub max_restarts: u32,            // 0 = never restart
    pub restart_window: Duration,     // reset counter after this much uptime
    pub restart_delay: Duration,      // wait before restarting
    pub restart_on: RestartTrigger,
}

pub enum RestartTrigger {
    /// Restart on source failure only.
    SourceFailure,
    /// Restart on source failure or stage panic.
    SourceOrStagePanic,
    /// Never restart automatically.
    Never,
}
```

Restart sequence:
1. All stages receive `on_stop()`.
2. GStreamer pipeline is torn down.
3. `TemporalStore` is cleared (fresh start, no stale state carried across).
4. `ViewEpoch` is incremented (even for `CameraMode::Fixed` feeds — the restart boundary is a state discontinuity regardless of camera type).
5. After `restart_delay`, GStreamer pipeline is reconstructed.
6. Stages receive `on_start()`.
7. Frames begin flowing.

### Runtime shutdown

`runtime.shutdown()` is async and returns a future:

1. All feeds are signaled to stop.
2. Each feed drains its stage pipeline (in-flight frame completes).
3. Each feed calls `on_stop()` on all stages.
4. GStreamer pipelines are torn down.
5. Output sinks receive any remaining buffered outputs.
6. The future resolves when all feeds have terminated.

Timeout: `runtime.shutdown_timeout(Duration)` — if feeds haven't stopped within the timeout, they are forcibly aborted.

### Feed removal

`runtime.remove_feed(id)` removes a single feed. Same sequence as shutdown but for one feed. Other feeds are unaffected.

---

## 14. Observability model

### Metrics (via `metrics` crate facade)

Per-feed counters and histograms:

```
nv.feed.frames_received{feed_id}          counter
nv.feed.frames_dropped{feed_id}           counter
nv.feed.frames_processed{feed_id}         counter
nv.feed.pipeline_latency_ns{feed_id}      histogram
nv.feed.stage_latency_ns{feed_id,stage}   histogram
nv.feed.queue_depth{feed_id,position}     gauge
nv.feed.tracks_active{feed_id}            gauge
nv.feed.view_epoch{feed_id}               gauge
nv.feed.view_version{feed_id}             gauge
nv.feed.view_stability{feed_id}           gauge
nv.feed.view_compensations{feed_id}       counter
nv.feed.view_degradations{feed_id}        counter
nv.feed.restarts{feed_id}                 counter
nv.feed.source_reconnects{feed_id}        counter
nv.feed.stage_errors{feed_id,stage}       counter
```

The library uses the `metrics` crate (facade pattern). Users plug in their own recorder (Prometheus, StatsD, etc.).

### Health events

```rust
pub enum HealthEvent {
    SourceConnected { feed_id: FeedId },
    SourceDisconnected { feed_id: FeedId, reason: MediaError },
    SourceReconnecting { feed_id: FeedId, attempt: u32 },
    StageError { feed_id: FeedId, stage_id: StageId, error: StageError },
    StagePanic { feed_id: FeedId, stage_id: StageId },
    FeedRestarting { feed_id: FeedId, restart_count: u32 },
    FeedStopped { feed_id: FeedId, reason: StopReason },
    BackpressureDrop { feed_id: FeedId, frames_dropped: u64 },
    ViewEpochChanged { feed_id: FeedId, epoch: ViewEpoch, decision: EpochDecision },
    ViewDegraded { feed_id: FeedId, reason: DegradationReason, stability_score: f32 },
    ViewCompensationApplied { feed_id: FeedId, epoch: ViewEpoch },
}
```

Health events are broadcast via `tokio::sync::broadcast`. Users subscribe through `FeedHandle::health_recv()` or `Runtime::health_recv()` (aggregate).

### Tracing

The library integrates with the `tracing` crate. Key spans:

- `feed{feed_id}` — per-feed root span
- `frame{feed_id, seq}` — per-frame span
- `stage{feed_id, stage_id}` — per-stage-invocation span

Users configure the tracing subscriber externally. The library only emits spans and events.

---

## 15. Error model

### Error hierarchy

```rust
// nv-core
pub enum NvError {
    Media(MediaError),
    Stage(StageError),
    Temporal(TemporalError),
    View(ViewError),
    Runtime(RuntimeError),
    Config(ConfigError),
}

pub enum MediaError {              // Clone
    ConnectionFailed { url: String, detail: String },
    DecodeFailed { detail: String },
    Eos,       // end of stream (file source)
    Timeout,
    Unsupported { detail: String },
}

pub enum StageError {
    ProcessingFailed { stage_id: StageId, detail: String },
    ResourceExhausted { stage_id: StageId },
    ModelLoadFailed { stage_id: StageId, detail: String },  // Clone
}

pub enum RuntimeError {
    FeedNotFound { feed_id: FeedId },
    AlreadyRunning,
    ShutdownInProgress,
    FeedLimitExceeded { max: usize },
}

pub enum ConfigError {
    InvalidSource { detail: String },
    InvalidPolicy { detail: String },
    MissingRequired { field: &'static str },
    /// CameraMode::Observed set but no ViewStateProvider supplied,
    /// or CameraMode::Fixed set but a ViewStateProvider was supplied.
    CameraModeConflict { detail: String },
}
```

### Error philosophy

- **Media errors** are expected in production (network issues, camera restarts). They trigger reconnection or feed restart, not panics.
- **Stage errors** for a single frame are non-fatal. The frame is dropped, metrics are incremented, and the feed continues.
- **Stage panics** are caught. The feed restarts per policy.
- **Config errors** are returned at feed creation time. They prevent broken feeds from starting.
- **Runtime errors** (feed not found, already running) are programmer errors — returned as `Result` to the caller.

---

## 16. Extension points

| Extension point | Mechanism | Purpose |
|---|---|---|
| `Stage` trait | User implements | Plug in detection, tracking, classification, signal derivation, or any per-frame computation. `Send + 'static`, not `Sync` — stages are exclusively owned by their feed thread. |
| `ViewStateProvider` trait | User implements | Supply camera motion information — PTZ telemetry, video-inferred egomotion, or external transforms. |
| `EpochPolicy` trait | User implements (optional) | Control how the view system responds to camera motion: segment, degrade, compensate, or continue. Default: `DefaultEpochPolicy`. |
| `ViewBoundContext<T>` | Generic utility struct | Bind user-managed calibration/regions/transforms to a specific ViewVersion/ViewEpoch with explicit staleness checks. |
| `OutputSink` trait | User implements | Forward outputs to Kafka, gRPC, files, websockets, or any destination. |
| `TypedMetadata` | Type-map on frames, detections, tracks, outputs | Attach arbitrary domain-specific data without modifying core types. |
| `SourceSpec::Custom` | GStreamer launch string | Escape hatch for exotic sources not covered by built-in variants. |
| `RetentionPolicy` | Config struct | Tune temporal state memory usage per deployment. |
| `BackpressurePolicy` | Config enum | Tune per-feed queue behavior. |
| Frame pixel format converters | `nv-frame::convert` | Add converters for domain-specific pixel formats. |

### What is deliberately NOT an extension point

- **Stage scheduling**: Linear, fixed. Not pluggable.
- **Temporal store implementation**: Single implementation, not trait-abstracted. Consistency matters more than swappability.
- **Reconnection behavior**: Configurable, but not user-supplied strategy. Reconnection is infrastructure, not domain logic.

---

## 17. Likely failure modes

| Failure mode | Mitigation |
|---|---|
| Camera goes offline | `ReconnectPolicy` handles automatic reconnection with exponential backoff. `HealthEvent::SourceDisconnected` is emitted. |
| Stage inference takes too long | Frames queue up to `queue_depth`, then oldest are dropped. Latency histograms make this visible. No timeout kill on stages — stages own their own timeout behavior. |
| Stage panics | `catch_unwind` at the stage call boundary. Feed restarts per `RestartPolicy`. Panic is logged and emitted as `HealthEvent::StagePanic`. |
| Memory growth from track accumulation | `RetentionPolicy` enforces hard caps on track count and observation history depth. Eviction runs every frame. |
| PTZ camera jumps mid-frame | View system detects discontinuity, consults `EpochPolicy`. Policy decides: `Continue` (within tolerance), `Degrade` (mark degraded), `Compensate` (transform existing positions), or `Segment` (increment epoch, close trajectory segments, notify stages). No hard-coded behavior. |
| GStreamer thread hangs | The bounded channel between GstThread and StgThread will fill. Supervisor detects no frames received for configurable timeout and triggers feed restart. |
| Output sink is slow | Bounded channel between StgThread and OutThread drops oldest outputs. Perception pipeline never blocks on output delivery. |
| Too many feeds for available CPU | Each feed uses 2-3 OS threads. With 100+ feeds, thread count becomes a concern. Future optimization: shared thread pool for stage execution with per-feed work-stealing. Initial design uses per-feed threads for isolation simplicity. |
| Corrupt or unusual video stream | GStreamer handles codec errors internally. `nv-media` catches decoder errors, emits `HealthEvent`, and allows the stream to continue (GStreamer skips corrupt frames). |

---

## 18. Tradeoffs and rejected alternatives

### Linear pipeline vs. DAG

**Chosen**: Linear stage pipeline.
**Rejected**: Directed acyclic graph of stages.
**Why**: DAGs require complex scheduling, partial-failure semantics, and fan-in synchronization. Linear pipelines cover >95% of real perception workloads. Users who need DAG-like behavior can compose multiple feeds or implement branching within a stage. Follows rule 22: don't build a framework inside the framework.

### OS threads vs. async for stages

**Chosen**: Dedicated OS thread per feed for stage execution.
**Rejected**: Running stages on a tokio runtime.
**Why**: Perception stages universally call blocking FFI (TensorRT, ONNX, OpenCV). Blocking an async executor violates its cooperative scheduling model. `spawn_blocking` adds overhead and complexity. Dedicated threads are simpler, faster, and correct.

### Shared thread pool vs. per-feed threads

**Chosen**: Per-feed threads (initially).
**Rejected**: Global thread pool with work-stealing.
**Why**: Per-feed threads provide strict isolation — one slow feed cannot steal CPU from another. This is the operationally correct default. A shared pool is a valid future optimization for high-feed-count deployments, but adds scheduling complexity and reduces isolation. The architecture allows this change without public API breakage.

### Trait-based media backend vs. GStreamer-specific

**Chosen**: GStreamer-specific `nv-media`, no `MediaBackend` trait.
**Rejected**: Abstract `MediaBackend` trait with GStreamer as one implementation.
**Why**: There is no second media backend planned. An abstract trait would be speculative generalization that adds complexity without value today. If a second backend emerges, the `nv-media` API surface is small enough to extract a trait at that time. YAGNI.

### Normalized coordinates vs. pixel coordinates

**Chosen**: Normalized `[0, 1]` coordinates throughout.
**Rejected**: Pixel coordinates.
**Why**: Normalized coordinates are resolution-independent. Streams may be decoded at varying resolutions. Tracking across resolution changes is broken if coordinates are pixel-based. The cost (one multiply to convert to pixels when needed) is negligible.

### Arc-based frame sharing vs. buffer pool with checkout

**Chosen**: `Arc<FrameInner>` with automatic deallocation on drop.
**Rejected**: Explicit buffer pool with checkout/return.
**Why**: Arc-based sharing is simpler and sufficient. A buffer pool adds complexity around lifetime management and doesn't improve performance when frames are GStreamer-mapped (the pool is on GStreamer's side). For synthetic/owned frames, allocation is rare enough that pooling isn't necessary.

### Clearing temporal state on restart vs. preserving it

**Chosen**: Full temporal state clear on feed restart.
**Rejected**: Preserving tracks/trajectories across restarts.
**Why**: After a restart, the GStreamer pipeline, decoder state, and potentially the camera itself have reset. Track IDs from the old session have no guaranteed correspondence with objects in the new session. Preserving stale state creates subtle bugs. Clean slate is the operationally honest choice. Domain layers can implement their own persistence if needed.

### Single `OutputEnvelope` per frame vs. event stream

**Chosen**: One `OutputEnvelope` per processed frame.
**Rejected**: Fine-grained event stream (detection events, track events, signal events separately).
**Why**: A single envelope per frame is simpler to consume, simpler to serialize, and provides a complete, consistent snapshot. Users don't need to reassemble events into a coherent frame view. If users want event-stream semantics, they can decompose the envelope in their `OutputSink`.

### `StageId` as `&'static str` vs. runtime `String`

**Chosen**: `&'static str`.
**Rejected**: `String`.
**Why**: Stage names are fixed at compile time (they're hardcoded in stage implementations). Using `&'static str` avoids allocation, enables `Copy`, and makes it clear these are not user-generated identifiers.

### EpochPolicy trait vs. hard-coded segmentation

**Chosen**: Pluggable `EpochPolicy` trait with a shipped `DefaultEpochPolicy`.
**Rejected**: Hard-coded "always segment on discontinuity."
**Why**: Real PTZ deployments have heterogeneous cameras with different motion characteristics. A camera doing a 2-degree pan adjustment should not force a full epoch segment. Auto-tracking PTZ cameras move constantly but with high-confidence transforms — they should compensate, not segment. The `EpochPolicy` trait is intentionally narrow (single method, one enum return) to avoid framework creep while giving users the control they need. The default policy covers threshold-based segmentation and is sufficient for non-PTZ and simple PTZ deployments.

### MotionReport vs. Option\<PtzTelemetry\> for ViewStateProvider

**Chosen**: `MotionReport` struct with multiple optional fields. `MotionSource` is derived by the view system from field presence, not set by the provider.
**Rejected**: `Option<PtzTelemetry>` return value.
**Why**: `Option<PtzTelemetry>` only works when a camera has PTZ telemetry. Cameras without telemetry — which includes most industrial, retail, sport, wildlife, and robotics cameras that may nonetheless move — had no first-class path for supplying ego-motion estimates. `MotionReport` accommodates telemetry, video-inferred egomotion, and external sources through the same trait. The `MotionSource` on `ViewState` (derived from the report) makes the trust level of the motion information explicit, which is essential for downstream gating.

### ViewVersion vs. relying on ViewEpoch alone

**Chosen**: Both `ViewVersion` (increments on any view change) and `ViewEpoch` (increments only on segment decisions).
**Rejected**: Using `ViewEpoch` as the sole version identifier.
**Why**: Within a single epoch, the view may change many times (continuous PTZ motion with `Degrade` or `Compensate` decisions). A stage that caches data relative to the current camera view needs to know when its cache is stale — even within an epoch. `ViewVersion` is this staleness check. It is a `u64` counter, zero-cost to compare, and saves consumers from doing deep equality checks on `ViewState`.

### Compensation via transform vs. always segmenting

**Chosen**: `EpochDecision::Compensate` as an option alongside `Segment`.
**Rejected**: Always segmenting on motion, leaving compensation to downstream.
**Why**: If the library always segments, consumers who want compensated continuity must re-implement trajectory stitching and coordinate transforms themselves — poorly, inconsistently, and repeatedly. Offering `Compensate` at the temporal-store level means the library handles the coordinate transform once, correctly, on the canonical track data. It only fires when the `EpochPolicy` explicitly opts in with a transform, so there is no risk of silent incorrect compensation.

### Required CameraMode vs. implicit fixed-camera default

**Chosen**: `CameraMode` is a required field on `FeedConfig` (no default). `Fixed` feeds bypass the view system. `Observed` feeds require a `ViewStateProvider` and default to `Unknown`/`Degraded` when the provider returns empty data.
**Rejected**: Defaulting to `CameraMotionState::Stable` / `MotionSource::None` when no provider is configured.
**Why**: The implicit-stable default silently produces incorrect results for any camera that moves — and the user gets no signal that anything is wrong. Requiring `CameraMode` forces the integrator to make a conscious declaration. The cost is one extra line in `FeedConfig`. The benefit is that moving cameras are never accidentally treated as fixed. This follows AGENTS.md rule 10: "Never hide important runtime behavior behind abstractions that make failure semantics unclear."

### ViewBoundContext utility vs. embedded view-binding in core types

**Chosen**: `ViewBoundContext<T>` as a standalone generic utility in `nv-view`. Not embedded into `Detection`, `Track`, or `PerceptionArtifacts`.
**Rejected**: Adding `bound_at: ViewVersion` fields to all core perception types.
**Why**: Most core types (detections, tracks) are already implicitly bound to the frame's view via the frame's `ViewSnapshot`. Adding view-binding fields to every struct would be redundant noise. `ViewBoundContext` is for *user-managed* context that persists across frames — calibration data, region definitions, reference transforms — where the staleness question is genuinely ambiguous. A utility type keeps core types lean and puts the complexity only where it is needed.

### Typed SegmentBoundary vs. untyped/implicit boundaries

**Chosen**: `SegmentBoundary` enum with explicit variants on `TrajectorySegment.opened_by` / `closed_by`.
**Rejected**: Inferring boundary reasons from epoch deltas or segment ordering.
**Why**: Trajectory consumers frequently need to answer "why was this segment split?" — for filtering, for visualization, for quality metrics. Inferring the reason from epoch counters is fragile (an epoch change could be a PTZ jump or a feed restart) and loses causal information that was available at split time. The `SegmentBoundary` enum is small, unambiguous, and written once at the moment the boundary occurs.
