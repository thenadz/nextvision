//! Shutdown and restart policy types.

use nv_core::Duration;

/// Controls automatic feed restart behavior after failures.
///
/// # Restart semantics
///
/// - `restart_on == Never` — never restart, regardless of other fields.
/// - `max_restarts == 0` — never restart, regardless of `restart_window`.
/// - `max_restarts > 0` — allow up to this many restarts within
///   `restart_window`. If the feed stays alive longer than the window
///   the counter resets, allowing further restarts.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Maximum number of restarts before giving up.
    ///
    /// `0` means never restart — the `restart_window` cannot override this.
    pub max_restarts: u32,
    /// Reset the restart counter after this much consecutive uptime.
    pub restart_window: Duration,
    /// Delay before restarting.
    pub restart_delay: Duration,
    /// What triggers an automatic restart.
    pub restart_on: RestartTrigger,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            restart_window: Duration::from_secs(300),
            restart_delay: Duration::from_secs(2),
            restart_on: RestartTrigger::SourceOrStagePanic,
        }
    }
}

/// What triggers an automatic feed restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartTrigger {
    /// Restart on source failure only.
    SourceFailure,
    /// Restart on source failure or stage panic.
    SourceOrStagePanic,
    /// Never restart automatically.
    Never,
}
