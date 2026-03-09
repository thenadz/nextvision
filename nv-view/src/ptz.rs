//! PTZ telemetry types.

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
    /// Timestamp at which this telemetry was captured.
    pub ts: MonotonicTs,
}
