//! Core view state types: [`ViewState`], [`ViewSnapshot`], [`ViewEpoch`], [`ViewVersion`].

use std::fmt;
use std::sync::Arc;

use crate::camera_motion::{CameraMotionState, MotionSource};
use crate::ptz::PtzTelemetry;
use crate::transform::GlobalTransformEstimate;
use crate::transition::TransitionPhase;
use crate::validity::ContextValidity;

/// Opaque monotonic epoch counter.
///
/// Incremented when the view system (via its configured [`EpochPolicy`](crate::EpochPolicy))
/// determines a discontinuity occurred that warrants segmenting temporal state.
///
/// Within a single uninterrupted feed session with `CameraMode::Fixed`,
/// the epoch never changes. Even for fixed feeds, epoch increments on feed restart.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ViewEpoch(u64);

impl ViewEpoch {
    /// The initial epoch (zero).
    pub const INITIAL: Self = Self(0);

    /// Create from a raw value.
    #[must_use]
    pub fn new(val: u64) -> Self {
        Self(val)
    }

    /// Return the next epoch.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Returns the raw value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ViewEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch-{}", self.0)
    }
}

impl fmt::Debug for ViewEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ViewEpoch({})", self.0)
    }
}

/// Monotonic version counter for the view-state itself.
///
/// Incremented on every [`ViewState`] update — even within the same epoch.
/// Allows consumers to cheaply test staleness: `my_version == current.version()`
/// without deep-comparing the full `ViewState`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ViewVersion(u64);

impl ViewVersion {
    /// The initial version (zero).
    pub const INITIAL: Self = Self(0);

    /// Create from a raw value.
    #[must_use]
    pub fn new(val: u64) -> Self {
        Self(val)
    }

    /// Return the next version.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Returns the raw value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Number of versions between `self` and `other`.
    /// Returns 0 if `other >= self`.
    #[must_use]
    pub fn versions_since(self, other: Self) -> u64 {
        self.0.saturating_sub(other.0)
    }
}

impl fmt::Display for ViewVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Debug for ViewVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ViewVersion({})", self.0)
    }
}

/// The current best estimate of the camera's view.
///
/// Updated every frame by the view system. Stages and output consumers
/// receive a [`ViewSnapshot`] (read-only, `Arc`-wrapped) rather than
/// `ViewState` directly.
#[derive(Clone, Debug)]
pub struct ViewState {
    /// Current epoch — incremented on significant view discontinuities.
    pub epoch: ViewEpoch,
    /// Current version — incremented on every view-state update.
    pub version: ViewVersion,
    /// Whether the camera is stable, moving, or unknown.
    pub motion: CameraMotionState,
    /// How the current motion state was determined.
    pub motion_source: MotionSource,
    /// Position in the motion transition state machine.
    pub transition: TransitionPhase,
    /// PTZ telemetry, if available.
    pub ptz: Option<PtzTelemetry>,
    /// Global transform estimate (frame-to-reference), if available.
    pub global_transform: Option<GlobalTransformEstimate>,
    /// Whether temporal state is valid under the current view.
    pub validity: ContextValidity,
    /// Stability score in `[0.0, 1.0]` — 1.0 = fully stable.
    pub stability_score: f32,
}

impl ViewState {
    /// Create the default initial view state for a fixed camera.
    #[must_use]
    pub fn fixed_initial() -> Self {
        Self {
            epoch: ViewEpoch::INITIAL,
            version: ViewVersion::INITIAL,
            motion: CameraMotionState::Stable,
            motion_source: MotionSource::None,
            transition: TransitionPhase::Settled,
            ptz: None,
            global_transform: None,
            validity: ContextValidity::Valid,
            stability_score: clamp_unit(1.0),
        }
    }

    /// Create the default initial view state for an observed camera.
    ///
    /// Starts with `CameraMotionState::Unknown` and `ContextValidity::Degraded`
    /// because no motion data has been received yet. The view system will
    /// upgrade to `Valid` once a provider delivers stable motion data.
    #[must_use]
    pub fn observed_initial() -> Self {
        Self {
            epoch: ViewEpoch::INITIAL,
            version: ViewVersion::INITIAL,
            motion: CameraMotionState::Unknown,
            motion_source: MotionSource::None,
            transition: TransitionPhase::Settled,
            ptz: None,
            global_transform: None,
            validity: ContextValidity::Degraded {
                reason: crate::validity::DegradationReason::Unknown,
            },
                stability_score: clamp_unit(0.0),
        }
    }
}

/// Read-only, cheaply-cloneable snapshot of [`ViewState`].
///
/// Created once per frame by the view system and shared with all stages
/// via `StageContext`. `Clone` is an `Arc` bump.
///
/// Stages and output consumers receive `ViewSnapshot`, not `ViewState`.
#[derive(Clone, Debug)]
pub struct ViewSnapshot {
    inner: Arc<ViewState>,
}

impl ViewSnapshot {
    /// Create a snapshot from a `ViewState`.
    #[must_use]
    pub fn new(state: ViewState) -> Self {
        Self {
            inner: Arc::new(state),
        }
    }

    /// Current epoch.
    #[must_use]
    pub fn epoch(&self) -> ViewEpoch {
        self.inner.epoch
    }

    /// Current version.
    #[must_use]
    pub fn version(&self) -> ViewVersion {
        self.inner.version
    }

    /// Camera motion state.
    #[must_use]
    pub fn motion(&self) -> &CameraMotionState {
        &self.inner.motion
    }

    /// How the motion state was determined.
    #[must_use]
    pub fn motion_source(&self) -> &MotionSource {
        &self.inner.motion_source
    }

    /// Transition phase.
    #[must_use]
    pub fn transition(&self) -> TransitionPhase {
        self.inner.transition
    }

    /// PTZ telemetry, if available.
    #[must_use]
    pub fn ptz(&self) -> Option<&PtzTelemetry> {
        self.inner.ptz.as_ref()
    }

    /// Global transform estimate, if available.
    #[must_use]
    pub fn global_transform(&self) -> Option<&GlobalTransformEstimate> {
        self.inner.global_transform.as_ref()
    }

    /// Context validity under the current view.
    #[must_use]
    pub fn validity(&self) -> &ContextValidity {
        &self.inner.validity
    }

    /// Stability score in `[0.0, 1.0]`.
    #[must_use]
    pub fn stability_score(&self) -> f32 {
        self.inner.stability_score
    }

    /// Borrow the underlying `ViewState`.
    #[must_use]
    pub fn as_view_state(&self) -> &ViewState {
        &self.inner
    }
}

fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ViewEpoch --

    #[test]
    fn view_epoch_initial_is_zero() {
        assert_eq!(ViewEpoch::INITIAL.as_u64(), 0);
    }

    #[test]
    fn view_epoch_next_increments() {
        let e = ViewEpoch::INITIAL;
        assert_eq!(e.next().as_u64(), 1);
        assert_eq!(e.next().next().as_u64(), 2);
    }

    #[test]
    fn view_epoch_ordering() {
        let a = ViewEpoch::new(3);
        let b = ViewEpoch::new(5);
        assert!(a < b);
        assert_eq!(a, ViewEpoch::new(3));
    }

    #[test]
    fn view_epoch_display() {
        assert_eq!(format!("{}", ViewEpoch::new(7)), "epoch-7");
    }

    // -- ViewVersion --

    #[test]
    fn view_version_initial_is_zero() {
        assert_eq!(ViewVersion::INITIAL.as_u64(), 0);
    }

    #[test]
    fn view_version_next_increments() {
        assert_eq!(ViewVersion::INITIAL.next().as_u64(), 1);
    }

    #[test]
    fn view_version_versions_since() {
        let v5 = ViewVersion::new(5);
        let v2 = ViewVersion::new(2);
        assert_eq!(v5.versions_since(v2), 3);
        // Returns 0 when other >= self.
        assert_eq!(v2.versions_since(v5), 0);
    }

    // -- ViewState constructors --

    #[test]
    fn fixed_initial_is_stable_valid() {
        let vs = ViewState::fixed_initial();
        assert_eq!(vs.epoch, ViewEpoch::INITIAL);
        assert_eq!(vs.motion, CameraMotionState::Stable);
        assert_eq!(vs.validity, ContextValidity::Valid);
        assert_eq!(vs.stability_score, 1.0);
        assert_eq!(vs.transition, TransitionPhase::Settled);
    }

    #[test]
    fn observed_initial_is_unknown_degraded() {
        let vs = ViewState::observed_initial();
        assert_eq!(vs.motion, CameraMotionState::Unknown);
        assert!(matches!(vs.validity, ContextValidity::Degraded { .. }));
        assert_eq!(vs.stability_score, 0.0);
    }

    // -- ViewSnapshot --

    #[test]
    fn snapshot_reflects_state() {
        let vs = ViewState::fixed_initial();
        let snap = ViewSnapshot::new(vs);
        assert_eq!(snap.epoch(), ViewEpoch::INITIAL);
        assert_eq!(snap.stability_score(), 1.0);
        assert_eq!(snap.transition(), TransitionPhase::Settled);
    }

    #[test]
    fn snapshot_clone_is_cheap() {
        let snap = ViewSnapshot::new(ViewState::fixed_initial());
        let snap2 = snap.clone();
        // Both should point to the same Arc allocation.
        assert!(std::ptr::eq(snap.as_view_state(), snap2.as_view_state()));
    }
}
