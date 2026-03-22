# Contributing to NextVision

Thank you for your interest in contributing! This document provides guidelines
for contributing to the NextVision workspace.

## Getting started

### Prerequisites

- Rust 1.92+ (install via [rustup](https://rustup.rs))
- GStreamer 1.16+ development libraries:
  ```bash
  # Debian/Ubuntu
  sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  ```

### Building

```bash
cargo build --workspace
```

### Testing

```bash
cargo test --workspace
```

### Linting

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Code standards

This project follows the coding standards documented in [AGENTS.md](AGENTS.md).
Key principles:

- **Event-driven first**: Behavior must be triggered by events (frame arrival,
  track transitions, health signals), not arbitrary timers or polling loops.
  Timers are acceptable only when no event source exists.
- **Domain-agnostic**: No vertical-specific concepts (traffic, retail, security)
  in core library crates.
- **GStreamer is an implementation detail**: GStreamer types must not leak into
  the public API.
- **DRY**: Extract shared helpers; avoid repeating logic.
- **Readability over cleverness**: Straightforward, explicit, logically organized.
- **Performance is a design concern**: Avoid unnecessary allocation, cloning, and
  lock contention on hot paths.
- **Bounded everything**: No unbounded queues in critical runtime paths.

## Pull request process

1. **Fork and branch** from `main`.
2. **Write tests** for meaningful behavior changes.
3. **Run the full check suite** before submitting:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo doc --workspace --no-deps
   ```
4. **Update documentation** if the change affects public API or behavior.
5. **Update CHANGELOG.md** under the `[Unreleased]` section.
6. **Keep commits focused**: one logical change per commit.
7. **Write a clear PR description** explaining what and why.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full specification — crate
boundaries, dependency graph, data flow, and design decisions.

## Benchmarks

If a change affects hot paths (frame handoff, queue operations, stage execution,
temporal-state retention), add or update benchmarks:

```bash
cargo bench -p nv-frame
cargo bench -p nv-temporal
cargo bench -p nv-runtime
```

## Licensing

This project is dual-licensed under MIT and Apache 2.0. By submitting a
contribution, you agree that your contribution will be licensed under both
licenses.
