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
//! # Phase 2 (planned)
//!
//! - Health event counters via `RuntimeHandle::health_subscribe()`
//!   (stage errors, sink panics, backpressure drops, etc.)
//! - Prometheus pull endpoint behind a `prometheus` feature flag

mod error;
mod exporter;
pub(crate) mod recorder;

pub use error::MetricsError;
pub use exporter::{MetricsExporter, MetricsExporterBuilder};

// Re-export KeyValue for ergonomic resource_attributes() usage.
pub use opentelemetry::KeyValue;
