# AGENTS.md

## Purpose

This repository contains a **domain-agnostic, production-oriented, high-performance Rust video perception runtime** built on top of **GStreamer**.

This library is intended to be a reusable foundation for multiple domains. It must remain independent of any single vertical such as transportation, transit, retail, security, sports, robotics, or industrial safety.

Agents working in this repository must preserve:

- strong architectural boundaries
- clean public APIs
- readable and maintainable code
- high-performance design
- minimal duplication
- domain-agnostic semantics
- operational robustness

This file defines the coding, architecture, and API standards that all contributions must follow.

---

## Core architectural principles

### 1. The library is domain-agnostic
Do not introduce domain-specific concepts into the core library.

Examples of forbidden concepts in the core:
- traffic lanes
- crosswalks
- stop bars
- ATMS-specific events
- retail-specific zones
- security-specific alarm workflows
- domain-specific event taxonomies

Allowed concepts are generic:
- frame
- detection
- track
- trajectory
- motion features
- scene features
- region
- coordinate space
- transform
- view state
- camera motion state
- health event
- provenance

If a concept would feel strange or overly specific to users in other domains, it probably does not belong in the core library.

---

### 2. GStreamer is an implementation detail of the media backend
The library is GStreamer-backed, but GStreamer types must not leak into the public API unless there is a compelling, explicit, reviewed reason.

Public users should interact with:
- library-defined source specs
- library-defined frame types
- library-defined runtime handles
- library-defined outputs and events

They should not need to understand GStreamer internals to use the library.

---

### 3. The public API is a first-class product
The public interface of this library is one of its most important assets.

All public APIs must be:
- intuitive
- coherent
- well-documented
- stable-minded
- minimal but expressive
- unsurprising

Avoid exposing internal implementation details.
Prefer strong, meaningful types over loose bags of options.
Prefer explicitness over cleverness.

Every new public type or trait should answer:
- why should this exist publicly?
- is the name intuitive?
- is this the smallest useful abstraction?
- will a new user understand how to use this from the docs alone?

---

### 4. DRY is mandatory
Do not repeat logic unnecessarily.

When similar logic appears in multiple places:
- extract shared helpers
- extract reusable internal abstractions
- unify duplicated patterns
- consolidate repeated lifecycle or conversion code

However, do not create abstractions that are more confusing than the duplication they replace.

Use judgment:
- eliminate wasteful repetition
- avoid premature abstraction
- prefer abstractions that clarify the design

---

### 5. Abstractions must earn their existence
Do not introduce layers of traits, wrappers, or indirection without clear benefit.

Every abstraction should improve one or more of:
- clarity
- code reuse
- testability
- performance isolation
- boundary enforcement
- future extensibility without complexity explosion

Avoid:
- abstraction for its own sake
- generic frameworks inside the library
- type-system gymnastics that obscure runtime behavior
- “enterprise architecture” style indirection

Prefer a small number of logical, durable abstractions.

---

### 6. Readability matters
This repository values readable code highly.

Code should be:
- straightforward
- explicit
- logically organized
- reasonably small in function and file size
- easy for another engineer to maintain

Avoid:
- clever tricks
- unnecessary macro-heavy designs
- giant files
- deeply nested control flow
- hidden side effects
- surprising ownership patterns

Performance is important, but readability should not be sacrificed for speculative micro-optimizations.

---

### 7. Performance is a design concern, not an afterthought
This is a high-performance runtime. Code should avoid needless overhead.

Be careful about:
- unnecessary allocation
- unnecessary cloning
- unnecessary copies of frame data
- lock contention
- unbounded buffering
- per-frame expensive logging
- hidden hot-path work in convenience helpers

When performance-sensitive code is introduced:
- make the hot path obvious
- document why the design is efficient
- avoid speculative complexity
- prefer profiling-guided improvements over guesswork

---

### 8. Temporal state is first-class
This library is not just a frame-processing toolkit.

Temporal concepts are first-class:
- tracks
- track observations
- trajectories
- motion features
- continuity state
- degradation state
- provenance over time

Agents must preserve the quality and integrity of temporal abstractions.
Do not collapse everything into frame-local logic.

---

### 9. PTZ/view-state is first-class and generic
The library must intelligently model changing camera views in a domain-agnostic way.

This includes:
- view state
- camera motion state
- explicit PTZ/control events if available
- inferred view-motion
- continuity degradation
- trajectory segmentation
- context validity under changing view state

Do not pretend PTZ/free-moving cameras are equivalent to fixed cameras.
Do not implement domain-specific policies for PTZ behavior in the core library.

---

### 10. Operational clarity matters
This library is intended for real systems.

Agents must preserve clear operational behavior around:
- startup
- shutdown
- restart
- backpressure
- queue limits
- error propagation
- feed isolation
- health/status reporting

Never hide important runtime behavior behind abstractions that make failure semantics unclear.

---

## Repository standards

### 11. Crate boundaries matter
Keep crate/module responsibilities clean.

Examples of appropriate separation:
- core shared types
- frame abstractions
- media abstractions
- GStreamer backend
- perception artifacts
- temporal state
- pipeline/runtime orchestration
- observability
- testing utilities

Do not let modules become junk drawers.

If a file or crate starts accumulating unrelated responsibilities, refactor it.

---

### 12. Public traits should be intentionally small
Traits in the public API should be focused and understandable.

Avoid:
- bloated traits
- catch-all mega-traits
- highly generic trait hierarchies that are difficult to implement
- public traits that expose internal assumptions

Prefer:
- narrow interfaces
- clear contracts
- explicit inputs and outputs
- useful documentation and examples

---

### 13. Errors must be typed and informative
Error handling should be:
- explicit
- typed
- contextual
- operationally useful

Avoid opaque stringly errors in core flows.
Use typed errors for meaningful failure categories.
Include enough context to debug real failures.

---

### 14. Tests are required for meaningful changes
Important changes must include tests.

At minimum, add or update tests for:
- public API behavior
- lifecycle behavior
- queue/backpressure behavior
- restart/shutdown behavior
- temporal-state behavior
- PTZ/view-state behavior
- provenance behavior
- failure isolation

Do not rely on manual reasoning alone for correctness in runtime code.

---

### 15. Benchmarks are expected for hot-path changes
If a change affects likely hot paths, add or update benchmarks where practical.

Examples:
- frame handoff
- queue operations
- stage execution
- temporal-state retention
- output construction

Keep benchmarks focused and useful.

---

### 16. Documentation is part of the implementation
Public APIs must be documented.
Crates should have meaningful crate-level docs.
The README should reflect reality.

When adding functionality, update docs as needed.

Documentation should explain:
- what the abstraction is for
- how to use it
- major lifecycle expectations
- important invariants or caveats

---

## Design rules for contributions

### 17. Prefer library-defined semantic types over generic maps
Avoid shoving meaning into:
- `HashMap<String, String>`
- untyped JSON blobs
- ambiguous metadata bags

Use strong types where there is durable meaning.

Metadata bags are acceptable only where genuinely open-ended extensibility is needed, and even then they should not replace first-class typed fields.

---

### 18. Avoid leaking implementation detail into the conceptual model
Do not let internal implementation choices shape the user-facing model unnecessarily.

For example:
- queue internals should not distort output semantics
- GStreamer concepts should not dominate public API names
- temporary runtime shortcuts should not become permanent type design

The conceptual model should reflect the library’s purpose, not its incidental implementation.

---

### 19. Backpressure and boundedness must be explicit
No unbounded queues in critical runtime paths unless there is an extremely strong reason.
Memory growth must be controlled.
Frame dropping/sampling/backpressure behavior must be explicit in config and code.

Hidden unbounded growth is unacceptable.

---

### 20. Preserve feed isolation
Per-feed failures, state, and lifecycle should remain isolated wherever practical.

Do not introduce shared-state shortcuts that make one feed able to poison the entire runtime unless there is a strong and documented reason.

---

### 21. Prefer composition over sprawling inheritance-like hierarchies
In Rust terms:
- prefer focused structs and traits
- prefer explicit composition
- avoid sprawling webs of generic wrappers and adapters

The architecture should feel modular, not tangled.

---

### 22. Do not build a framework inside the framework
This repository is a video perception runtime, not a general workflow engine.

Avoid turning the stage system into:
- a giant DAG engine
- a plugin meta-framework
- a general orchestration platform
- a research sandbox

Keep the system focused and practical.

---

## Change evaluation checklist

Before finalizing any significant code change, ask:

1. Does this preserve domain-agnostic design?
2. Does this improve or at least preserve public API clarity?
3. Is any new abstraction justified?
4. Did I reduce duplication where appropriate?
5. Did I avoid introducing unnecessary complexity?
6. Is runtime behavior still operationally clear?
7. Did I preserve performance expectations?
8. Did I avoid hidden allocations/clones/locks in hot paths?
9. Did I add tests for meaningful behavior?
10. Did I update docs if the behavior or API changed?

If the answer to any of these is no, revise the change.

---

## Guidance for agents generating code

When generating code in this repository:

- prefer coherent, compileable, realistic implementations
- prefer smaller complete slices over broad fake scaffolding
- avoid TODO-heavy architecture skeletons
- keep public APIs intentional
- keep internal boundaries clean
- preserve domain neutrality
- preserve PTZ/view-state correctness
- preserve temporal-state quality
- prefer operational clarity over elegance theater

If forced to choose between:
- cleverness and readability, choose readability
- broad scaffolding and working implementation, choose working implementation
- abstraction purity and user ergonomics, choose user ergonomics
- speculative optimization and clean design, choose clean design unless profiling clearly justifies otherwise

---

## Non-goals for the core library

The following are explicitly out of scope for the core library:
- domain event engines
- domain reporting systems
- ATMS/TMC integrations
- alert workflows
- business rules for specific verticals
- UI concerns
- operator workflow semantics
- product-layer calibration semantics beyond generic context/view-state support

These may be built above the library, but must not contaminate the core.

---

## Final instruction

Keep this library something that an excellent engineer in an unrelated domain would respect, understand, and be willing to adopt.

That means:
- strong fundamentals
- clean boundaries
- low nonsense
- no domain contamination
- no architecture cosplay