//! View-bound context: binding user data to a specific camera view.

use crate::view_state::{ViewEpoch, ViewState, ViewVersion};

/// A piece of user-supplied context data bound to a specific camera view.
///
/// The library does not interpret the contents — it only tracks whether
/// the binding is still valid under the current view. Useful for
/// calibration data, spatial regions, reference transforms, etc.
///
/// # Example
///
/// ```
/// use nv_view::{ViewBoundContext, ViewState, BoundContextValidity};
///
/// struct CalibrationData { /* ... */ }
///
/// let view = ViewState::fixed_initial();
/// let bound = ViewBoundContext::bind(CalibrationData {}, &view);
///
/// // Same view → Current.
/// assert!(matches!(bound.validity(&view), BoundContextValidity::Current));
/// ```
#[derive(Clone, Debug)]
pub struct ViewBoundContext<T> {
    /// The user's data.
    pub data: T,
    /// The `ViewVersion` at which this data was created or last rebind.
    pub bound_at: ViewVersion,
    /// The `ViewEpoch` at which this data was created.
    pub bound_epoch: ViewEpoch,
}

impl<T> ViewBoundContext<T> {
    /// Bind data to the current view state.
    #[must_use]
    pub fn bind(data: T, view: &ViewState) -> Self {
        Self {
            data,
            bound_at: view.version,
            bound_epoch: view.epoch,
        }
    }

    /// Check whether this binding is still valid under the given view state.
    #[must_use]
    pub fn validity(&self, current: &ViewState) -> BoundContextValidity {
        if current.version == self.bound_at {
            BoundContextValidity::Current
        } else if current.epoch == self.bound_epoch {
            BoundContextValidity::StaleWithinEpoch {
                versions_behind: current.version.versions_since(self.bound_at),
            }
        } else {
            BoundContextValidity::InvalidAcrossEpoch {
                epochs_behind: current.epoch.as_u64().saturating_sub(self.bound_epoch.as_u64()),
            }
        }
    }

    /// Rebind this context to the current view.
    ///
    /// The caller asserts the data is still valid at the new view.
    pub fn rebind(&mut self, view: &ViewState) {
        self.bound_at = view.version;
        self.bound_epoch = view.epoch;
    }
}

/// Result of checking a [`ViewBoundContext`]'s validity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundContextValidity {
    /// Same `ViewVersion` — context is exactly current.
    Current,

    /// Same `ViewEpoch` but different `ViewVersion`. View has changed
    /// within the epoch (small motions, compensations). Context may be
    /// approximately valid but should be checked.
    StaleWithinEpoch {
        /// How many versions behind the current view.
        versions_behind: u64,
    },

    /// Different `ViewEpoch` — view has changed fundamentally.
    /// Context should be considered invalid.
    InvalidAcrossEpoch {
        /// How many epochs behind the current view.
        epochs_behind: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_when_same_version() {
        let view = ViewState::fixed_initial();
        let bound = ViewBoundContext::bind(42u32, &view);
        assert_eq!(bound.validity(&view), BoundContextValidity::Current);
    }

    #[test]
    fn stale_within_epoch() {
        let view = ViewState::fixed_initial();
        let bound = ViewBoundContext::bind(42u32, &view);
        let mut updated = view.clone();
        updated.version = ViewVersion::new(5);
        assert_eq!(
            bound.validity(&updated),
            BoundContextValidity::StaleWithinEpoch {
                versions_behind: 5
            }
        );
    }

    #[test]
    fn invalid_across_epoch() {
        let view = ViewState::fixed_initial();
        let bound = ViewBoundContext::bind(42u32, &view);
        let mut updated = view.clone();
        updated.epoch = ViewEpoch::new(3);
        updated.version = ViewVersion::new(10);
        assert_eq!(
            bound.validity(&updated),
            BoundContextValidity::InvalidAcrossEpoch { epochs_behind: 3 }
        );
    }

    #[test]
    fn rebind_resets_validity() {
        let view = ViewState::fixed_initial();
        let mut bound = ViewBoundContext::bind(42u32, &view);
        let mut updated = view.clone();
        updated.version = ViewVersion::new(5);
        bound.rebind(&updated);
        assert_eq!(bound.validity(&updated), BoundContextValidity::Current);
    }
}
