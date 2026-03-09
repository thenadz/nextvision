//! Generic derived signals — named scalar/vector/boolean/categorical values.
//!
//! Signals are domain-agnostic. Stage authors define the names and semantics.
//! Multiple stages may emit different signals per frame.

use nv_core::MonotonicTs;

/// A generic named signal produced by a perception stage.
///
/// Signals carry a name, a value, and the timestamp at which they were computed.
/// The name is a `&'static str` — signal names are fixed at compile time in
/// the stage implementation.
///
/// # Examples
///
/// ```
/// use nv_perception::{DerivedSignal, SignalValue};
/// use nv_core::MonotonicTs;
///
/// let signal = DerivedSignal {
///     name: "scene_complexity",
///     value: SignalValue::Scalar(0.73),
///     ts: MonotonicTs::from_nanos(1_000_000),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct DerivedSignal {
    /// Signal name — compile-time constant chosen by the stage author.
    pub name: &'static str,
    /// The signal value.
    pub value: SignalValue,
    /// Timestamp at which this signal was computed.
    pub ts: MonotonicTs,
}

/// The value of a derived signal.
#[derive(Clone, Debug)]
pub enum SignalValue {
    /// A single scalar value.
    Scalar(f64),
    /// A numeric vector (feature vector, histogram, etc.).
    Vector(Vec<f64>),
    /// A boolean flag.
    Boolean(bool),
    /// A categorical value — a compile-time label.
    Categorical(&'static str),
}
