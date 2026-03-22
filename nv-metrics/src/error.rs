use thiserror::Error;

/// Errors that can occur when building or operating the metrics exporter.
#[derive(Debug, Error)]
pub enum MetricsError {
    /// No [`RuntimeHandle`](nv_runtime::RuntimeHandle) was provided to the builder.
    #[error("no runtime handle provided")]
    NoRuntimeHandle,

    /// No exporter backend is available.
    ///
    /// Enable the `otlp-grpc` feature or supply a pre-built
    /// [`SdkMeterProvider`](opentelemetry_sdk::metrics::SdkMeterProvider)
    /// via [`MetricsExporterBuilder::meter_provider()`](crate::MetricsExporterBuilder::meter_provider).
    #[error("no exporter configured — enable the `otlp-grpc` feature or provide a custom MeterProvider")]
    NoExporter,

    /// The OpenTelemetry SDK returned an error during shutdown or flush.
    #[error("OpenTelemetry SDK error: {0}")]
    OtelSdk(#[from] opentelemetry_sdk::error::OTelSdkError),

    /// `poll_interval` was set to zero, which would create a hot loop.
    #[error("poll_interval must be > 0")]
    ZeroPollInterval,

    /// `build()` was called outside a tokio runtime context.
    #[error("no tokio runtime active — MetricsExporter::build() must be called within a tokio runtime")]
    NoTokioRuntime,

    /// The OTLP exporter could not be constructed.
    #[cfg(feature = "otlp-grpc")]
    #[error("OTLP exporter build error: {0}")]
    ExporterBuild(#[from] opentelemetry_otlp::ExporterBuildError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_runtime_handle_display() {
        let e = MetricsError::NoRuntimeHandle;
        assert_eq!(e.to_string(), "no runtime handle provided");
    }

    #[test]
    fn no_exporter_display() {
        let e = MetricsError::NoExporter;
        assert!(e.to_string().contains("no exporter configured"));
    }

    #[test]
    fn errors_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetricsError>();
    }

    #[test]
    fn zero_poll_interval_display() {
        let e = MetricsError::ZeroPollInterval;
        assert_eq!(e.to_string(), "poll_interval must be > 0");
    }

    #[test]
    fn no_tokio_runtime_display() {
        let e = MetricsError::NoTokioRuntime;
        assert!(e.to_string().contains("no tokio runtime active"));
    }
}
