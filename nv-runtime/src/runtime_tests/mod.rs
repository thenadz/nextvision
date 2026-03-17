//! Integration tests for the runtime: feed lifecycle, restart, shutdown,
//! output subscription, backpressure, pause/resume, provenance, and
//! sentinel-based lag detection.

mod config;
mod diagnostics;
mod feed;
mod harness;
mod lifecycle;
mod output;
mod sink;
