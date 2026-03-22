//! Builder and runtime for the background metrics exporter.
//!
//! [`MetricsExporter`] owns a background tokio task that polls
//! [`RuntimeHandle::diagnostics()`] at a configurable interval and
//! records values into OpenTelemetry instruments. The OTel SDK's
//! [`PeriodicReader`](opentelemetry_sdk::metrics::PeriodicReader)
//! handles batching and export to the configured backend.

use std::time::Duration;

use nv_runtime::RuntimeHandle;
use opentelemetry::metrics::MeterProvider;
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::error::MetricsError;
use crate::recorder::Instruments;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Configures and constructs a [`MetricsExporter`].
///
/// # Minimal example
///
/// ```no_run
/// use std::time::Duration;
/// use nv_metrics::MetricsExporter;
/// # fn example(handle: nv_runtime::RuntimeHandle) -> Result<(), nv_metrics::MetricsError> {
///
/// let metrics = MetricsExporter::builder()
///     .runtime_handle(handle)
///     .otlp_endpoint("http://localhost:4317")
///     .build()?;
///
/// // ... application runs ...
///
/// metrics.shutdown();
/// # Ok(())
/// # }
/// ```
pub struct MetricsExporterBuilder {
    handle: Option<RuntimeHandle>,
    poll_interval: Duration,
    service_name: String,
    resource_attributes: Vec<KeyValue>,
    provider: Option<SdkMeterProvider>,
    #[cfg(feature = "otlp-grpc")]
    otlp_endpoint: Option<String>,
}

impl Default for MetricsExporterBuilder {
    fn default() -> Self {
        Self {
            handle: None,
            poll_interval: Duration::from_secs(5),
            service_name: "nv-runtime".into(),
            resource_attributes: Vec::new(),
            provider: None,
            #[cfg(feature = "otlp-grpc")]
            otlp_endpoint: None,
        }
    }
}

impl MetricsExporterBuilder {
    /// Set the runtime handle to poll for diagnostics.
    ///
    /// Required. The exporter clones this handle internally.
    #[must_use]
    pub fn runtime_handle(mut self, handle: RuntimeHandle) -> Self {
        self.handle = Some(handle);
        self
    }

    /// Set how often to poll `RuntimeHandle::diagnostics()`.
    ///
    /// Default: 5 seconds. Lower values give finer-grained metrics at
    /// the cost of slightly more CPU. Values below 1 second are unusual
    /// for infrastructure metrics.
    #[must_use]
    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Set the OpenTelemetry service name resource attribute.
    ///
    /// Default: `"nv-runtime"`. This appears as `service.name` in your
    /// observability backend.
    #[must_use]
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Add extra OpenTelemetry resource attributes.
    ///
    /// These are merged with the service name to form the OTel
    /// [`Resource`] attached to all exported metrics. Useful for
    /// environment, region, or deployment identifiers.
    #[must_use]
    pub fn resource_attributes(mut self, attrs: Vec<KeyValue>) -> Self {
        self.resource_attributes = attrs;
        self
    }

    /// Set the OTLP gRPC endpoint (e.g., `"http://localhost:4317"`).
    ///
    /// When set, the exporter creates an OTLP/gRPC metrics pipeline
    /// automatically. Mutually exclusive with [`meter_provider()`](Self::meter_provider).
    #[cfg(feature = "otlp-grpc")]
    #[must_use]
    pub fn otlp_endpoint(mut self, url: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(url.into());
        self
    }

    /// Provide a pre-built [`SdkMeterProvider`].
    ///
    /// Use this when your application already has an OTel pipeline and
    /// you want metrics to flow through the same provider.
    ///
    /// When set, `otlp_endpoint()`, `service_name()`, and
    /// `resource_attributes()` are ignored — the provider's
    /// configuration takes precedence.
    #[must_use]
    pub fn meter_provider(mut self, provider: SdkMeterProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Build and start the background metrics exporter.
    ///
    /// Spawns a tokio task that polls diagnostics at the configured
    /// interval. The task runs until [`MetricsExporter::shutdown()`]
    /// is called or the exporter is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`MetricsError::NoRuntimeHandle`] if no handle was set.
    /// Returns [`MetricsError::NoExporter`] if no provider or endpoint
    /// was configured.
    pub fn build(self) -> Result<MetricsExporter, MetricsError> {
        let provider = self.resolve_provider()?;
        let handle = self.handle.ok_or(MetricsError::NoRuntimeHandle)?;
        let meter = provider.meter("nv-metrics");
        let instruments = Instruments::new(&meter);

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let poll_interval = self.poll_interval;

        let task = tokio::spawn(poll_loop(handle, instruments, poll_interval, shutdown_rx));

        debug!(
            poll_interval_ms = poll_interval.as_millis() as u64,
            "nv-metrics exporter started"
        );

        Ok(MetricsExporter {
            _provider: provider,
            shutdown_tx,
            task: Some(task),
        })
    }

    fn resolve_provider(&self) -> Result<SdkMeterProvider, MetricsError> {
        // User-provided provider takes priority.
        if let Some(ref provider) = self.provider {
            return Ok(provider.clone());
        }

        // Build from OTLP endpoint.
        #[cfg(feature = "otlp-grpc")]
        {
            if let Some(ref endpoint) = self.otlp_endpoint {
                return build_otlp_provider(
                    endpoint,
                    &self.service_name,
                    &self.resource_attributes,
                );
            }
        }

        Err(MetricsError::NoExporter)
    }
}

// ---------------------------------------------------------------------------
// OTLP provider construction
// ---------------------------------------------------------------------------

#[cfg(feature = "otlp-grpc")]
fn build_otlp_provider(
    endpoint: &str,
    service_name: &str,
    extra_attrs: &[KeyValue],
) -> Result<SdkMeterProvider, MetricsError> {
    use opentelemetry_otlp::{MetricExporter, WithExportConfig};
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::metrics::PeriodicReader;

    let exporter = MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let reader = PeriodicReader::builder(exporter).build();

    let mut resource_attrs: Vec<KeyValue> = vec![KeyValue::new(
        "service.name",
        service_name.to_owned(),
    )];
    resource_attrs.extend_from_slice(extra_attrs);

    let provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(
            Resource::builder()
                .with_attributes(resource_attrs)
                .build(),
        )
        .build();

    Ok(provider)
}

// ---------------------------------------------------------------------------
// Background poll loop
// ---------------------------------------------------------------------------

async fn poll_loop(
    handle: RuntimeHandle,
    instruments: Instruments,
    interval: Duration,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    debug!("nv-metrics poll loop shutting down");
                    return;
                }
            }
        }

        match handle.diagnostics() {
            Ok(diag) => instruments.record(&diag),
            Err(e) => {
                warn!(error = %e, "nv-metrics: failed to read diagnostics, skipping cycle");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MetricsExporter
// ---------------------------------------------------------------------------

/// Background OpenTelemetry metrics exporter for the NextVision runtime.
///
/// Polls [`RuntimeHandle::diagnostics()`] at a fixed interval and records
/// all metrics as OTel gauges. The OTel SDK handles batched export to the
/// configured backend (OTLP, Prometheus, etc.).
///
/// Create via [`MetricsExporter::builder()`].
///
/// # Lifecycle
///
/// The exporter runs until [`shutdown()`](Self::shutdown) is called.
/// Dropping without shutdown cancels the poll loop but does **not**
/// flush pending metrics. Always call `shutdown()` for clean teardown.
pub struct MetricsExporter {
    /// Kept alive so the OTel pipeline is not dropped prematurely.
    _provider: SdkMeterProvider,
    shutdown_tx: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
}

impl MetricsExporter {
    /// Start building a new exporter.
    #[must_use]
    pub fn builder() -> MetricsExporterBuilder {
        MetricsExporterBuilder::default()
    }

    /// Gracefully stop the poll loop and flush pending metrics.
    ///
    /// Signals the background task to exit, waits for it to finish,
    /// and shuts down the OTel meter provider (which flushes any
    /// buffered exports).
    ///
    /// This method blocks the current **async** context momentarily
    /// while the poll loop completes its current cycle.
    pub async fn shutdown(mut self) -> Result<(), MetricsError> {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
        self._provider.shutdown()?;
        debug!("nv-metrics exporter shut down");
        Ok(())
    }
}

impl Drop for MetricsExporter {
    fn drop(&mut self) {
        // Best-effort: signal the poll loop to stop.
        let _ = self.shutdown_tx.send(true);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_handle() {
        let result = MetricsExporter::builder()
            .meter_provider(SdkMeterProvider::builder().build())
            .build();
        assert!(matches!(result, Err(MetricsError::NoRuntimeHandle)));
    }

    #[test]
    fn builder_defaults() {
        let b = MetricsExporterBuilder::default();
        assert_eq!(b.poll_interval, Duration::from_secs(5));
        assert_eq!(b.service_name, "nv-runtime");
        assert!(b.resource_attributes.is_empty());
        assert!(b.provider.is_none());
        assert!(b.handle.is_none());
    }

    #[test]
    fn builder_no_provider_no_endpoint_errors() {
        let runtime = nv_runtime::Runtime::builder().build().unwrap();
        let result = MetricsExporter::builder()
            .runtime_handle(runtime.handle())
            .build();
        assert!(
            matches!(result, Err(MetricsError::NoExporter)),
            "should fail when neither provider nor endpoint is set"
        );
        runtime.shutdown().unwrap();
    }

    #[test]
    fn builder_method_chaining() {
        let b = MetricsExporterBuilder::default()
            .poll_interval(Duration::from_secs(10))
            .service_name("my-service")
            .resource_attributes(vec![KeyValue::new("env", "test")]);

        assert_eq!(b.poll_interval, Duration::from_secs(10));
        assert_eq!(b.service_name, "my-service");
        assert_eq!(b.resource_attributes.len(), 1);
    }

    #[tokio::test]
    async fn builder_meter_provider_overrides_endpoint() {
        let runtime = nv_runtime::Runtime::builder().build().unwrap();
        // When a custom provider is supplied, build succeeds even
        // without an OTLP endpoint.
        let result = MetricsExporter::builder()
            .runtime_handle(runtime.handle())
            .meter_provider(SdkMeterProvider::builder().build())
            .build();
        assert!(result.is_ok());
        runtime.shutdown().unwrap();
    }

    #[tokio::test]
    async fn shutdown_completes_gracefully() {
        let runtime = nv_runtime::Runtime::builder().build().unwrap();
        let exporter = MetricsExporter::builder()
            .runtime_handle(runtime.handle())
            .meter_provider(SdkMeterProvider::builder().build())
            .poll_interval(Duration::from_millis(50))
            .build()
            .unwrap();

        // Let at least one poll cycle run.
        tokio::time::sleep(Duration::from_millis(80)).await;

        let result = exporter.shutdown().await;
        assert!(result.is_ok(), "shutdown should complete without error");
        runtime.shutdown().unwrap();
    }

    #[tokio::test]
    async fn poll_loop_records_diagnostics() {
        use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader};

        let runtime = nv_runtime::Runtime::builder().build().unwrap();

        let mem_exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(mem_exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();

        let exporter = MetricsExporter::builder()
            .runtime_handle(runtime.handle())
            .meter_provider(provider.clone())
            .poll_interval(Duration::from_millis(50))
            .build()
            .unwrap();

        // Wait for a couple of poll cycles.
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Force a metric collection.
        let _ = provider.force_flush();

        let metrics = mem_exporter.get_finished_metrics().unwrap();
        // The poll loop should have recorded at least one runtime metric.
        let has_uptime = metrics.iter().any(|rm| {
            rm.scope_metrics().any(|sm| {
                sm.metrics().any(|m| m.name() == "nv.runtime.uptime_seconds")
            })
        });
        assert!(has_uptime, "poll loop should record nv.runtime.uptime_seconds");

        exporter.shutdown().await.unwrap();
        runtime.shutdown().unwrap();
    }

    #[tokio::test]
    async fn drop_signals_shutdown() {
        let runtime = nv_runtime::Runtime::builder().build().unwrap();
        let exporter = MetricsExporter::builder()
            .runtime_handle(runtime.handle())
            .meter_provider(SdkMeterProvider::builder().build())
            .build()
            .unwrap();

        // Grab a reference to the shutdown receiver before dropping.
        let rx = exporter.shutdown_tx.subscribe();
        drop(exporter);

        // The Drop impl should have sent `true`.
        assert!(
            *rx.borrow(),
            "dropping MetricsExporter should signal shutdown"
        );
        runtime.shutdown().unwrap();
    }
}
