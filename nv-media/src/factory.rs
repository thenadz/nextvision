//! Concrete [`MediaIngressFactory`] implementation.
//!
//! [`GstMediaIngressFactory`] creates GStreamer-backed [`MediaSource`]
//! instances. The runtime holds one factory and calls [`create()`] for each
//! new feed.

use std::sync::Arc;

use nv_core::error::MediaError;

use crate::ingress::{HealthSink, IngressOptions, MediaIngress, MediaIngressFactory};
use crate::source::MediaSource;

/// Default [`MediaIngressFactory`] that creates GStreamer-backed [`MediaSource`] instances.
///
/// The runtime holds one factory instance and calls [`create()`](MediaIngressFactory::create)
/// for each new feed. An optional [`HealthSink`] can be provided at construction
/// time; it will be attached to every source created by this factory.
pub struct GstMediaIngressFactory {
    health_sink: Option<Arc<dyn HealthSink>>,
}

/// Backend-neutral alias for the default [`MediaIngressFactory`] implementation.
///
/// Currently points to [`GstMediaIngressFactory`] (the GStreamer backend).
/// Downstream code should prefer this alias so that public-facing APIs and
/// documentation read as backend-agnostic.
pub type DefaultMediaFactory = GstMediaIngressFactory;

impl GstMediaIngressFactory {
    /// Create a factory with no health event reporting.
    #[must_use]
    pub fn new() -> Self {
        Self { health_sink: None }
    }

    /// Create a factory that attaches the given health sink to every source.
    #[must_use]
    pub fn with_health_sink(health_sink: Arc<dyn HealthSink>) -> Self {
        Self {
            health_sink: Some(health_sink),
        }
    }
}

impl Default for GstMediaIngressFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl MediaIngressFactory for GstMediaIngressFactory {
    fn create(&self, options: IngressOptions) -> Result<Box<dyn MediaIngress>, MediaError> {
        let mut source = MediaSource::new(
            options.feed_id,
            options.spec,
            options.reconnect,
            options.decode_preference,
        );
        if let Some(ref hs) = self.health_sink {
            source.set_health_sink(Arc::clone(hs));
        }
        if let Some(ptz) = options.ptz_provider {
            source.set_ptz_provider(ptz);
        }
        if let Some(hook) = options.post_decode_hook {
            source.set_post_decode_hook(hook);
        }
        source.event_queue_capacity = options.event_queue_capacity;
        source.device_residency = options.device_residency;
        Ok(Box::new(source))
    }
}
