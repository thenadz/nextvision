//! Runtime and runtime builder.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use nv_core::error::{NvError, RuntimeError};
use nv_core::id::FeedId;

use crate::feed::{FeedConfig, FeedHandle};

/// Builder for constructing a [`Runtime`].
///
/// # Example
///
/// ```
/// use nv_runtime::Runtime;
///
/// let runtime = Runtime::builder()
///     .max_feeds(16)
///     .build()
///     .expect("failed to build runtime");
/// ```
pub struct RuntimeBuilder {
    max_feeds: usize,
}

impl RuntimeBuilder {
    /// Set the maximum number of concurrent feeds. Default: `64`.
    #[must_use]
    pub fn max_feeds(mut self, max: usize) -> Self {
        self.max_feeds = max;
        self
    }

    /// Build the runtime.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if configuration is invalid.
    pub fn build(self) -> Result<Runtime, NvError> {
        Ok(Runtime {
            max_feeds: self.max_feeds,
            next_feed_id: AtomicU64::new(1),
            feeds: Mutex::new(HashMap::new()),
        })
    }
}

/// The top-level NextVision runtime.
///
/// Manages cross-feed concerns: feed registry, global limits, and shutdown.
/// Create via [`Runtime::builder()`].
///
/// Each feed is tracked by its [`FeedId`]. The runtime enforces a maximum
/// feed count and provides correct add/remove semantics.
pub struct Runtime {
    max_feeds: usize,
    next_feed_id: AtomicU64,
    feeds: Mutex<HashMap<FeedId, FeedEntry>>,
}

/// Internal state tracked per feed.
struct FeedEntry {
    /// The original config is consumed by the runtime when the feed is started.
    /// Stored here so that the runtime can restart feeds per restart policy.
    _config_consumed: bool,
}

impl Runtime {
    /// Create a new [`RuntimeBuilder`].
    #[must_use]
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder { max_feeds: 64 }
    }

    /// Number of currently active feeds.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::RegistryPoisoned` if the internal lock is poisoned.
    pub fn feed_count(&self) -> Result<usize, NvError> {
        let feeds = self.feeds.lock().map_err(|_| {
            NvError::Runtime(RuntimeError::RegistryPoisoned)
        })?;
        Ok(feeds.len())
    }

    /// Maximum allowed concurrent feeds.
    #[must_use]
    pub fn max_feeds(&self) -> usize {
        self.max_feeds
    }

    /// Add a new feed to the runtime.
    ///
    /// The feed begins running immediately. Returns a [`FeedHandle`] for
    /// monitoring and controlling the feed.
    ///
    /// # Errors
    ///
    /// - `RuntimeError::FeedLimitExceeded` if the max feed count is reached.
    /// - `ConfigError` if the feed configuration is invalid.
    pub fn add_feed(&self, _config: FeedConfig) -> Result<FeedHandle, NvError> {
        let mut feeds = self.feeds.lock().map_err(|_| {
            NvError::Runtime(RuntimeError::RegistryPoisoned)
        })?;
        if feeds.len() >= self.max_feeds {
            return Err(NvError::Runtime(RuntimeError::FeedLimitExceeded {
                max: self.max_feeds,
            }));
        }
        let id = FeedId::new(self.next_feed_id.fetch_add(1, Ordering::Relaxed));
        feeds.insert(id, FeedEntry {
            _config_consumed: true,
        });
        // Full runtime will spawn the feed's I/O thread, stage executor,
        // and wire the media ingress → pipeline → output sink chain here.
        Ok(FeedHandle::new(id))
    }

    /// Remove a feed by ID.
    ///
    /// The feed is stopped gracefully (stages receive `on_stop()`).
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::FeedNotFound` if the ID does not exist.
    pub fn remove_feed(&self, feed_id: FeedId) -> Result<(), NvError> {
        let mut feeds = self.feeds.lock().map_err(|_| {
            NvError::Runtime(RuntimeError::RegistryPoisoned)
        })?;
        if feeds.remove(&feed_id).is_some() {
            Ok(())
        } else {
            Err(NvError::Runtime(RuntimeError::FeedNotFound { feed_id }))
        }
    }

    /// Initiate graceful shutdown of all feeds.
    ///
    /// Returns a future that resolves when all feeds have terminated.
    pub async fn shutdown(self) -> Result<(), NvError> {
        let mut feeds = self.feeds.lock().map_err(|_| {
            NvError::Runtime(RuntimeError::RegistryPoisoned)
        })?;
        feeds.clear();
        // Full runtime will send stop signals to each feed thread,
        // await their termination, and drain output sinks.
        Ok(())
    }
}
