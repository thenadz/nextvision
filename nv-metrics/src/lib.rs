//! Optional OpenTelemetry metrics exporter for the NextVision runtime.
//!
//! `nv-metrics` is a **standalone, optional** crate that bridges
//! [`RuntimeHandle::diagnostics()`](nv_runtime::RuntimeHandle::diagnostics)
//! to the OpenTelemetry metrics ecosystem. The core library crates
//! (`nv-core`, `nv-runtime`, etc.) have **zero** dependency on this crate.
//!
//! # Quick start
//!
//! ```no_run
//! use std::time::Duration;
//! use nv_metrics::MetricsExporter;
//! # async fn example(handle: nv_runtime::RuntimeHandle) -> Result<(), nv_metrics::MetricsError> {
//!
//! let metrics = MetricsExporter::builder()
//!     .runtime_handle(handle)
//!     .service_name("my-video-service")
//!     .otlp_endpoint("http://otel-collector:4317")
//!     .poll_interval(Duration::from_secs(5))
//!     .build()?;
//!
//! // ... application runs ...
//!
//! metrics.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # What gets exported
//!
//! Every poll cycle reads one [`RuntimeDiagnostics`](nv_runtime::RuntimeDiagnostics)
//! snapshot and records:
//!
//! | Namespace | Attributes | Examples |
//! |---|---|---|
//! | `nv.runtime.*` | — | `uptime_seconds`, `feed_count`, `output_lag` |
//! | `nv.feed.*` | `feed_id` | `frames_received`, `tracks_active`, `view_stability_score` |
//! | `nv.batch.*` | `processor_id` | `items_processed`, `avg_processing_ms`, `consecutive_errors` |
//!
//! See [`recorder`] module docs for the full instrument list.
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---|---|---|
//! | `otlp-grpc` | **yes** | OTLP/gRPC exporter via tonic |
//!
//! # Bring your own provider
//!
//! If your application already has an OTel pipeline, pass your
//! existing [`SdkMeterProvider`](opentelemetry_sdk::metrics::SdkMeterProvider)
//! via [`MetricsExporterBuilder::meter_provider()`] instead of
//! configuring an endpoint. The exporter will record into your
//! provider's meter and respect your export/resource configuration.
//!
//! **Ownership note:** when you supply your own provider, `shutdown()`
//! does *not* call `provider.shutdown()` — the caller retains full
//! lifecycle ownership. The exporter only shuts down providers it
//! created internally (i.e. from an OTLP endpoint).
//!
//! # Health-event counters
//!
//! In addition to periodic gauge snapshots, the exporter subscribes to
//! [`RuntimeHandle::health_subscribe()`](nv_runtime::RuntimeHandle::health_subscribe)
//! and increments OTel **counters** on each event:
//!
//! | Counter | Attributes | Trigger |
//! |---|---|---|
//! | `nv.health.source_disconnections` | `feed_id` | `SourceDisconnected` |
//! | `nv.health.source_reconnections` | `feed_id` | `SourceReconnecting` |
//! | `nv.health.stage_errors` | `feed_id`, `stage_id` | `StageError` |
//! | `nv.health.stage_panics` | `feed_id`, `stage_id` | `StagePanic` |
//! | `nv.health.feed_restarts` | `feed_id` | `FeedRestarting` |
//! | `nv.health.feeds_stopped` | `feed_id` | `FeedStopped` |
//! | `nv.health.backpressure_drops` | `feed_id` | `BackpressureDrop` |
//! | `nv.health.view_epoch_changes` | `feed_id` | `ViewEpochChanged` |
//! | `nv.health.view_degradations` | `feed_id` | `ViewDegraded` |
//! | `nv.health.output_lag_events` | — | `OutputLagged` |
//! | `nv.health.sink_panics` | `feed_id` | `SinkPanic` |
//! | `nv.health.sink_timeouts` | `feed_id` | `SinkTimeout` |
//! | `nv.health.sink_backpressure` | `feed_id` | `SinkBackpressure` |
//! | `nv.health.track_admission_rejected` | `feed_id` | `TrackAdmissionRejected` |
//! | `nv.health.batch_errors` | `processor_id` | `BatchError` |
//! | `nv.health.batch_submission_rejected` | `feed_id`, `processor_id` | `BatchSubmissionRejected` |
//! | `nv.health.batch_timeouts` | `feed_id`, `processor_id` | `BatchTimeout` |
//! | `nv.health.batch_in_flight_exceeded` | `feed_id`, `processor_id` | `BatchInFlightExceeded` |
//!
//! # Optional extensions
//!
//! - A Prometheus pull endpoint can be added behind a `prometheus` feature flag.

mod error;
mod exporter;
pub(crate) mod recorder;

pub use error::MetricsError;
pub use exporter::{MetricsExporter, MetricsExporterBuilder};

// Re-export KeyValue for ergonomic resource_attributes() usage.
pub use opentelemetry::KeyValue;
