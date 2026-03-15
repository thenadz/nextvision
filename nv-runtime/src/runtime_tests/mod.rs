//! Integration tests for the runtime: feed lifecycle, restart, shutdown,
//! output subscription, backpressure, pause/resume, provenance, and
//! sentinel-based lag detection.

mod harness;
mod lifecycle;
mod output;
mod feed;
mod sink;
mod config;
mod diagnostics;
