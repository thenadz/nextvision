//! Shutdown and restart policy types.

use nv_core::Duration;

/// Controls automatic feed restart behavior after failures.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Maximum number of restarts before giving up. `0` = never restart.
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
