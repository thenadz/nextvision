//! Backpressure policy configuration.

/// Controls queue behavior between pipeline stages.
///
/// Applied to the bounded channel between the GStreamer decode thread
/// and the stage execution thread.
#[derive(Debug, Clone)]
pub enum BackpressurePolicy {
    /// Drop the oldest frame in the queue to make room for new ones.
    ///
    /// This is the default — real-time perception prefers fresh frames
    /// over stale ones.
    DropOldest {
        /// Maximum number of frames in the queue.
        queue_depth: usize,
    },

    /// Drop the incoming frame if the queue is full.
    DropNewest {
        /// Maximum number of frames in the queue.
        queue_depth: usize,
    },

    /// Block the producer until space is available.
    ///
    /// Use with caution — can cause GStreamer buffer buildup.
    Block {
        /// Maximum number of frames in the queue.
        queue_depth: usize,
    },
}

impl BackpressurePolicy {
    /// Returns the configured queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        match self {
            Self::DropOldest { queue_depth }
            | Self::DropNewest { queue_depth }
            | Self::Block { queue_depth } => *queue_depth,
        }
    }
}

impl Default for BackpressurePolicy {
    fn default() -> Self {
        Self::DropOldest { queue_depth: 4 }
    }
}
