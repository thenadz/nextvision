//! Retention and eviction policy for the temporal store.

use nv_core::Duration;

/// Controls memory usage of the temporal store via bounded eviction.
///
/// Applied before processing each frame.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum age for a track in `Lost` state before eviction.
    pub max_track_age: Duration,
    /// Maximum observations retained per track (ring buffer depth).
    pub max_observations_per_track: usize,
    /// Hard cap on concurrent tracks. Oldest `Lost` then `Coasted` are evicted first.
    pub max_concurrent_tracks: usize,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_track_age: Duration::from_secs(30),
            max_observations_per_track: 100,
            max_concurrent_tracks: 500,
        }
    }
}
