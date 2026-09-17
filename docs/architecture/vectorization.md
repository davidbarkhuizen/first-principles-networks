# vectorization

[← back to README](../../README.md)

`docs/structure.md`'s "possible next steps" flags NumPy vectorization as a values question ("no
ML framework dependency, everything hand-built" is this repo's own stated identity) rather than a
given; this document, and the three it links to below, are the detailed design and measurement
behind that choice.

## why not just adopt real NumPy

Installing `numpy` would obviously work, and would be the pragmatic choice for anyone whose only
goal is faster training. This document exists because this repo's own identity treats even a
"just fast math" dependency as a deliberate choice, not a default - the same posture that's kept
scikit-learn as a one-time, offline, non-runtime extraction tool (`digits_data.py`) rather than a
real dependency, and pyarrow as a one-time conversion step
(`mnist_data.convert_parquet_to_binary`) rather than something training itself ever imports.

## the three parts

1. **[vectorized array-based classes](vectorized-array-classes.md)** - what the model classes
   look like when this codebase's forward/backward/gradient math is rewritten around whole-layer
   arrays instead of individual node objects, built and validated against real `numpy`. Covers
   the central architectural finding this effort turns on: an array-based network can't reuse
   `BackpropNetworkBase`'s existing per-node orchestration at all, so it's a genuinely standalone
   class, sharing only the external contract (`learn`, `classify_state`, `snapshot`, ...) every
   other sibling network already shares.
2. **[the numpy interface subset](numpy-interface-subset.md)** - the precise, minimal set of
   numpy operations that document's classes actually call, derived directly from their design -
   the formal contract the third document satisfies.
3. **[the Rust array core](rust-array-core.md)** - a hand-built, tightly-scoped array core in
   Rust, wrapped for Python via PyO3/maturin, implementing exactly document 2's contract - this
   codebase's production array backend (see "decision" below).

## decision

`VectorizedMultiClassBackpropClassifierNetwork` and its real-numpy backend
([vectorized array-based classes](vectorized-array-classes.md)) are kept indefinitely as a
standing performance-benchmarking mirror - the reference point any backend's own speed claim gets
measured against - not a prototype superseded once something faster exists.
[The Rust array core](rust-array-core.md) is this codebase's production array backend - actual
training/demo code routes through it via `RustArrayLayer`/
`RustArrayMultiClassBackpropClassifierNetwork` (see [the production cutover
plan](rust-production-cutover.md), done through phase 2: a real, measured speedup over numpy at
UCI digits/real MNIST scale, statistically indistinguishable accuracy - first measured at
3.40x/1.31x, moved to ~3.6x/~2.6x by later matmul SIMD work, see [structure](../project/structure.md#possible-next-steps)).
Both are kept side by side, permanently, each for a different purpose - neither replaces the
other.

## expected effect

A throwaway numpy benchmark at this codebase's real architecture (`dimension=784`, `hidden=16`,
`output=10`) measured **150.7x** forward-pass speedup over the current pure-Python
implementation, correctness-checked first (max difference 5.27e-16 against the pure-Python
reference) - see [the Rust core](rust-array-core.md#expected-vs-measured-performance) for the
full benchmark. Treated as a ceiling, not a target: a naive, unoptimized Rust core should land
well below it, with 15-45x (10-30% of the measured ceiling) still a large, practically
significant win.

Real, not extrapolated, measurements exist for both. For the numpy-backed classes: one full
real-MNIST epoch (`[30]`-hidden-layer architecture, batch-size-1 SGD) measured **33.87x** (32.7s
vectorized vs. 18.45 min pure-Python), and UCI digits measured **9.00x** (8.57s vs. 77.16s) - both
landing inside the 15-45x practical-win range, at real, practical batch-size-1 SGD rather than a
forward-pass-only microbenchmark. See [vectorized array-based
classes](vectorized-array-classes.md#measured-results) for the full numbers. For the Rust core,
now that [the production cutover](rust-production-cutover.md) is done (see "decision" above): a
genuine speedup over numpy on this codebase's actual `learn()`-shaped (`batch_size=1`) training
paths, first measured at **3.40x** at UCI digits scale and **1.31x** at real MNIST scale (see [the
Rust core](rust-array-core.md) and [research and
analysis](../research/research-rust-performance.md#phase-2-tier-2-real-per-example-training-is-a-genuine-win-at-both-scales-measured)),
moved to **~3.6x**/**~2.6x** by later matmul SIMD work targeting the `batch_size=1` shape directly
(see [research and
analysis](../research/research-rust-performance.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)).
