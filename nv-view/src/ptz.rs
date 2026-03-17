//! PTZ telemetry types and discrete control events.

use nv_core::MonotonicTs;

/// Raw PTZ (Pan-Tilt-Zoom) telemetry from the camera.
///
/// Provided by `ViewStateProvider` implementations that have access
/// to PTZ control systems (ONVIF, serial protocols, etc.).
#[derive(Clone, Debug, PartialEq)]
pub struct PtzTelemetry {
    /// Pan angle in degrees.
    pub pan: f32,
    /// Tilt angle in degrees.
    pub tilt: f32,
    /// Zoom level, normalized to `[0, 1]` where `1` = maximum zoom.
    pub zoom: f32,
    /// Timestamp of this telemetry reading.
    pub ts: MonotonicTs,
}

/// A discrete PTZ control event.
///
/// Models push-based PTZ control commands that the camera may receive
/// externally (from an operator, a tour scheduler, or an API call).
/// These supplement the polling-based [`PtzTelemetry`] with explicit
/// causal information about **why** the camera moved.
///
/// Providers include these in [`MotionReport::ptz_events`](crate::MotionReport::ptz_events)
/// when they have access to a PTZ command stream (e.g., ONVIF Events).
#[derive(Clone, Debug, PartialEq)]
pub enum PtzEvent {
    /// A continuous move started (pan/tilt/zoom speeds set).
    MoveStart {
        /// Timestamp of the command.
        ts: MonotonicTs,
    },
    /// A continuous move stopped.
    MoveStop {
        /// Timestamp of the command.
        ts: MonotonicTs,
    },
    /// Camera moved to a preset position.
    PresetRecall {
        /// Numeric preset identifier.
        preset_id: u32,
        /// Timestamp of the command.
        ts: MonotonicTs,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptz_telemetry_clone_eq() {
        let t = PtzTelemetry {
            pan: 45.0,
            tilt: -10.0,
            zoom: 0.5,
            ts: MonotonicTs::from_nanos(1_000_000),
        };
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    #[test]
    fn ptz_event_variants() {
        let ts = MonotonicTs::from_nanos(100);

        let start = PtzEvent::MoveStart { ts };
        let stop = PtzEvent::MoveStop { ts };
        let preset = PtzEvent::PresetRecall { preset_id: 3, ts };

        // Each variant is distinct.
        assert_ne!(start, stop);
        assert_ne!(stop, preset.clone());

        // Clone round-trips.
        assert_eq!(preset.clone(), preset);
    }
}
