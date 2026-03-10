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
    /// Hard cap on concurrent tracks.
    ///
    /// Enforced strictly: `commit_track` pre-evicts a victim when
    /// admitting a new track at cap, so the store size never exceeds
    /// this limit after each commit.
    ///
    /// Eviction priority: `Lost` (oldest) → `Coasted` (oldest) →
    /// `Tentative` (oldest). `Confirmed` tracks are never evicted.
    ///
    /// If only `Confirmed` tracks remain at cap, new track admissions
    /// are rejected (existing track updates are always accepted).
    pub max_concurrent_tracks: usize,
    /// Maximum trajectory points retained per track across all segments.
    ///
    /// When exceeded, the oldest points (from the oldest closed segments) are
    /// pruned first. This prevents unbounded memory growth for long-lived
    /// confirmed tracks.
    pub max_trajectory_points_per_track: usize,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_track_age: Duration::from_secs(30),
            max_observations_per_track: 100,
            max_concurrent_tracks: 500,
            max_trajectory_points_per_track: 1000,
        }
    }
}
