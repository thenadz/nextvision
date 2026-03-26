//! # nv-test-util
//!
//! Test utilities for the NextVision runtime.
//!
//! Provides:
//!
//! - **[`synthetic`]** — factory functions for synthetic `FrameEnvelope`s with
//!   known content (solid color, gradient, etc.).
//! - **[`mock_stage`]** — configurable mock `Stage` implementations for testing
//!   pipelines without real perception models.
//! - **[`harness`]** — a lightweight [`StageHarness`] for
//!   testing stages in isolation without the full runtime.
//! - **[`clock`]** — a controllable test clock for deterministic timestamp generation.
//!
//! This crate is intended as a `dev-dependency` only.

pub mod clock;
pub mod harness;
pub mod mock_stage;
pub mod synthetic;

pub use harness::StageHarness;
pub use mock_stage::NullTemporalAccess;
