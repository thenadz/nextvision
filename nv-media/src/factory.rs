//! Concrete [`MediaIngressFactory`] implementation.
//!
//! [`GstMediaIngressFactory`] creates GStreamer-backed [`MediaSource`]
//! instances. The runtime holds one factory and calls [`create()`] for each
//! new feed.

use std::sync::Arc;

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::MediaError;
use nv_core::id::FeedId;

use crate::ingress::{HealthSink, MediaIngress, MediaIngressFactory, PtzProvider};
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
    fn create(
        &self,
        feed_id: FeedId,
        spec: SourceSpec,
        reconnect: ReconnectPolicy,
        ptz_provider: Option<Arc<dyn PtzProvider>>,
    ) -> Result<Box<dyn MediaIngress>, MediaError> {
        let mut source = MediaSource::new(feed_id, spec, reconnect);
        if let Some(ref hs) = self.health_sink {
            source.set_health_sink(Arc::clone(hs));
        }
        if let Some(ptz) = ptz_provider {
            source.set_ptz_provider(ptz);
        }
        Ok(Box::new(source))
    }
}
