# research and analysis: Rust core performance

[← back to research and analysis](research-and-analysis.md)

The investigation thread behind [the Rust production cutover](rust-production-cutover.md)'s
performance work - the debug-build discovery, fusing, and the `batch_size >= 32` matmul gap's
closing sequence (blocking, threading, SIMD) - see [research and
analysis](research-and-analysis.md) for what this collection of docs is for.

## the Rust array core's 65-335x-slower-than-numpy finding was a debug-build artifact

[The production cutover plan](rust-production-cutover.md)'s own gating benchmark measured the
Rust core's naive per-op forward pass (a line-for-line port composed from individual `Array`
operator calls) as 65.7x-335.4x slower than numpy at this codebase's real layer size
(`dimension=784, hidden=16`), widening with batch size - the premise phase 0's entire fix-and-
gate structure was built around. Re-measuring it immediately before starting phase 0a's matmul
loop reorder surfaced the actual cause: `./cli setup`/`./cli build-rust` built the crate with a
plain `maturin develop` - a debug (unoptimized) build - and that debug build, not anything
inherent to the per-op composition approach, was almost the entire gap.

Same forward pass, same architecture, four build/loop-order combinations, batch sizes 1/8/32/128/512
(ratio = Rust wall-clock / numpy wall-clock; correctness held throughout, max elementwise
difference 1e-15 to 2e-14, float64 noise at every combination):

| batch | debug build, original loop order | release build, original loop order | release build, after 0a's loop reorder |
|---|---|---|---|
| 1 | 71.5x | 3.4x | 2.4x |
| 8 | 160.2x | 3.8x | 1.7x |
| 32 | 228.4x | 5.0x | 2.6x |
| 128 | 427.6x | 10.4x | 6.5x |
| 512 | 374.9x | 9.5x | 4.7x |

The debug-build column reproduces the plan's original 65.7x-335.4x figures closely, confirming
the cause. Under a release build - the only build any real deployment would ship - the naive
per-op composition was already only ~3-10x slower than numpy before any of phase 0's fixes, and
the matmul loop reorder alone (0a) brings it to ~2-7x, without yet fusing per-layer calls (0b).

**Decision:** `./cli build-rust`/`./cli setup` now build with `maturin develop --release` (a
one-line fix - see the `cli` script and [setup](setup.md)) - a benchmark's headline number is only
as good as the build it was measured against, and this one went unquestioned through two doc
revisions before being re-run.

## phase 0b: fusing closes some further ground, but matmul is now the binding constraint

With the release-build baseline corrected and 0a's loop reorder in, phase 0b fused each
`ArrayLayer` method into a single Rust call (`fused.rs` - `layer_forward`, `layer_forward_batch`,
`layer_output_delta`, `layer_hidden_delta`/`_batch`, `layer_accumulate_gradient`/`_batch`,
`layer_apply_accumulated_gradient`), eliminating the per-op FFI-crossing count entirely (one
Python-to-Rust call per layer method, not four to six). Re-measuring `layer_forward_batch` alone
against numpy, same architecture as every prior table here (`dimension=784, hidden=16`):

| batch size | numpy | fused Rust | ratio |
|---|---|---|---|
| 1 | 14.4 us | 27.7 us | 1.9x slower |
| 8 | 21.9 us | 45.1 us | 2.1x slower |
| 32 | 57.6 us | 135.1 us | 2.4x slower |
| 128 | 154.2 us | 666.4 us | 4.3x slower |
| 512 | 489.1 us | 2082.5 us | 4.3x slower |

### a fuller comparison: one whole mini-batch training step

Fusing closed some further ground over 0a alone (batch 128 was 6.5x slower after 0a alone;
4.3x after 0b) but not decisively - matmul, still a naive triple loop with no SIMD/blocking, is
now the dominant remaining cost at these batch sizes, and fusing doesn't touch it. A fuller
comparison - one whole mini-batch training step (`layer_forward_batch` + `layer_output_delta` +
`layer_accumulate_gradient_batch` + `layer_apply_accumulated_gradient` vs. the equivalent numpy
`ArrayLayer` method sequence, `dimension=784, hidden=10` - this codebase's real output-layer
shape) tells a more textured story than the forward-pass-only table above:

| batch size | numpy | fused Rust | ratio |
|---|---|---|---|
| 1 | 68.0 us | 30.5 us | **0.45x - Rust is faster** |
| 8 | 67.2 us | 64.6 us | ~parity |
| 32 | 108.3 us | 189.6 us | 1.75x slower |
| 128 | 201.5 us | 798.9 us | 4.03x slower |
| 512 | 1740.3 us | 3478.3 us | 2.00x slower |

(Correctness held throughout both tables - max weight difference after a full training step ≤
5e-16, float64 noise; every fused function checked against `array_layer.py`'s own `ArrayLayer`
methods directly, not an independently-written reference formula - see
`rust/indrajala_ml_array/tests/test_fused_layer_ops.py`.)

### the split result, and the decision to build phase 1 anyway

**The split result:** the fused Rust core beats numpy for per-example training (`batch_size=1`,
`ArrayLayer.learn`'s own call shape) but loses to it, by a widening margin, for realistic
mini-batch training (`batch_size >= 32`, `learn_batch`'s shape) - the opposite of what [the
production cutover plan](rust-production-cutover.md#risks-and-open-questions)'s own "risks and
open questions" guessed before this was measured (it expected batching to be the Rust core's
strength and per-example calls its weakness, on the theory that a single Python↔Rust call still
costs something no pure-Python-calling-numpy path pays - true, but overwhelmed at larger batches
by numpy's BLAS-backed matmul pulling further ahead of this crate's still-naive one). **Decision:
build phase 1 in full anyway.** The user clarified (2026-09-16) that adoption of the Rust core as
the production backend is unconditional, not gated on beating numpy first - numpy stays on
permanently as the benchmark comparison, and the `batch_size >= 32` gap above is tracked as
follow-on optimization work (SIMD, blocked/tiled matmul, threading), not a reason to withhold
`RustArrayMultiClassBackpropClassifierNetwork` or scope it down to the per-example path alone. See
[the production cutover plan](rust-production-cutover.md#the-decisive-finding-this-plan-has-to-answer-first)
("adoption is unconditional, not gated on this benchmark") for the full framing change.

## phase 2 tier 2: real per-example training is a genuine win at both scales measured

Good news the synthetic phase-0b benchmark only hinted at: both of this codebase's actual
production training paths (`demo_vectorized_uci_digit_recognition.py`,
`demo_vectorized_mnist_recognition.py`) train via `train_linear_classifier_network`, i.e.
`learn()`'s per-example (`batch_size=1`) path exclusively - checked directly, neither demo calls
`learn_batch`. That is exactly the regime phase 0b's benchmark found the fused Rust core *beats*
numpy in, not the `batch_size >= 32` regime it loses in. Tier 2 (docs/rust-production-cutover.md's
phase 2) measures this on real training runs, not synthetic per-step timings: full
`randomize()`-to-trained-accuracy runs, multiple independent seeds per class (the RNG mismatch
means seeds can't be matched across classes - see [the Rust core](rust-array-core.md#the-rng-exception) -
so seeds are independent draws, compared by distribution, not by seed-for-seed identity).

### UCI digits: 3.40x faster

**UCI digits** (`[32]` hidden, `dimension=64`, same fixed `split_train_test(seed=0)` split
`demo_uci_digit_recognition.py` uses, `learning_rate=0.5`, `epochs=30`, 8 seeds each):

| network | train accuracy | test accuracy | wall-clock |
|---|---|---|---|
| numpy (`VectorizedMultiClassBackpropClassifierNetwork`) | mean 99.57%, stdev 0.07% | mean 96.62%, stdev 0.47% | 4.27s |
| Rust (`RustArrayMultiClassBackpropClassifierNetwork`) | mean 99.58%, stdev 0.03% | mean 96.90%, stdev 0.22% | 1.26s |

**3.40x faster**, with test accuracy distributions overlapping well inside each other's stdev band
- statistically indistinguishable, matching this codebase's own established
"correctness/accuracy first, then compare" standard.

### real MNIST: 1.31x faster

**Real MNIST** (`[30]` hidden, `dimension=784` - `demo_vectorized_mnist_recognition.py`'s own
already-documented architecture, `learning_rate=0.5`, 1 real epoch over the full 60000-example
training set, 3 seeds each - fewer seeds than UCI digits purely because each one costs over ten
times as long):

| network | train accuracy | test accuracy | wall-clock |
|---|---|---|---|
| numpy | mean 93.13%, stdev 0.01% | mean 92.96%, stdev 0.14% | 15.1s |
| Rust | mean 92.86%, stdev 0.11% | mean 92.66%, stdev 0.17% | 11.6s |

**1.31x faster**, smaller than UCI digits' win - real MNIST's much larger `dimension=784` matmul
gives the naive triple loop more relative work per call even at `batch_size=1`, eating into the
per-call-overhead advantage that dominates at UCI digits' smaller `dimension=64` - but still a
genuine, positive speedup, with test accuracy within 0.3 points, comparable to each network's own
seed-to-seed spread. Both figures are honest, measured numbers from real training runs on real
data, not extrapolated from the synthetic phase 0b benchmark.

## build-flag tuning measured as a null

First candidate tried for closing the naive matmul's remaining `batch_size >= 32` gap (see
[structure](structure.md#possible-next-steps)): release-profile/codegen tuning, the cheapest
possible lever since it changes no algorithm - motivated directly by this same document's own
debug-vs-release discovery, which showed build configuration can matter more than assumed here.
Two variants measured against the same full mini-batch training-step benchmark phase 0b used
(`layer_forward_batch` + `layer_output_delta` + `layer_accumulate_gradient_batch` +
`layer_apply_accumulated_gradient`, `dimension=784, hidden=10`), each run twice to separate a
real effect from noise:

| variant | batch 1 | batch 8 | batch 32 | batch 128 | batch 512 |
|---|---|---|---|---|---|
| baseline (`--release`) | 0.39x, 0.50x | 0.99x, 1.09x | 1.79x, 1.79x | 3.28x, 3.77x | 2.23x, 1.80x |
| `+ lto=true, codegen-units=1` | 0.43x | 1.08x | 1.76x | 4.14x | 1.93x |
| `+ RUSTFLAGS="-C target-cpu=native"` (AVX2 available) | 0.37x, 0.52x | 1.14x, 1.06x | 1.55x, 1.63x | 3.88x, 3.70x | 2.15x, 2.09x |

(ratio = Rust wall-clock / numpy wall-clock; lower is better for Rust.) Every variant's numbers
fall inside the baseline's own run-to-run spread (e.g. batch 128 alone ranges 3.28x-4.14x across
runs of the *identical* binary) - neither `lto`/`codegen-units` nor `target-cpu=native` produced a
change distinguishable from noise, at any batch size. **Decision: not adopted** - the
`Cargo.toml`/`RUSTFLAGS` changes were reverted rather than kept for no measured benefit, the same
"correctness-validated, not adopted for performance reasons" treatment momentum/L2/Xavier-Glorot
got. Plausible reason: this crate is small enough (~900 lines) that a normal `--release` build's
default codegen-unit count already gives LLVM everything it needs to inline across module
boundaries, so `lto`/`codegen-units=1` had nothing further to unlock; `target-cpu=native`'s lack
of effect suggests the matmul inner loop (`linalg.rs`, indexed slice access,
`out_row[col] += a_value * b_row[col]`) isn't auto-vectorizing even with AVX2 available - the next
candidates (blocking, threading, explicit SIMD intrinsics) target that directly instead of hoping
the compiler finds it unassisted.

## cache-blocked matmul: a real, shape-dependent win, gated by size

Second candidate for the `batch_size >= 32` gap: cache-block `linalg.rs::matmul`'s 2D×2D case
over `row` and `k` (fixed-size slabs reused across multiple output rows before moving on),
instead of the plain `row -> k -> col` loop phase 0a left in place. Measured directly with raw
matmul calls (not the full training-step benchmark) at three shapes, to isolate the effect from
noise: two of this codebase's own real matmul shapes, plus one exaggerated size to confirm
blocking works at all before tuning it for this codebase's scale:

| shape | `b` size | unblocked | unconditionally blocked | ratio |
|---|---|---|---|---|
| `forward_batch`-like: `(512,784)@(784,16)` | ~100KB | 1.94ms | 2.21ms | **0.87x - a regression** |
| `accumulate_gradient_batch`-like: `(10,512)@(512,784)` | ~3.2MB | 1.28ms | 0.96ms | **1.33x faster** |
| exaggerated: `(2048,2048)@(2048,2048)` | 32MB | 5251ms | 2527ms | **2.08x faster** |

Blocking only helps once `b` (the operand that gets re-streamed once per output row in the
unblocked loop) is big enough to not already fit in cache - at `dimension=784, hidden=16`, `b` is
only ~100KB, already cache-resident, so blocking's extra bookkeeping was pure overhead with
nothing to relieve. **Fix, not abandonment**: gate blocking on `b`'s size
(`c1 * c2 * size_of::<f64>()`), with a 256KB threshold comfortably below a typical machine's L2
cache - below it, run the plain unblocked loop (verified back to baseline speed, not just
"not worse"); at or above it, block. Re-measured on the full training-step benchmark
(`dimension=784, hidden=10`, 3 runs): `batch_size=512`'s ratio improved from a 1.80x-2.23x range
(no blocking) to a 1.53x-1.74x range (size-gated blocking) - a real, reproducible ~15-25%
reduction in that batch size's gap, with `batch_size=1`/`8` unaffected (matches expectation -
`batch_size=1`'s `learn()` path never even reaches the 2D×2D matmul case; `layer_forward_batch`'s
own matmul stays under the threshold at this codebase's real `dimension=784` architecture
regardless of batch size, since `c1`/`c2` there are `dimension`/`hidden`, not `batch`). **Decision:
adopted** - both branches verified bit-identical to the previous single-path implementation (525
existing tests unchanged; blocking only restructures loop order, not summation order), and this is
a genuine, if modest, step in closing docs/structure.md's tracked follow-on gap, not a full
resolution of it. Threading and SIMD intrinsics remain open next steps for the rest of that gap.

## threaded matmul: a real further win, one real bug caught, one refinement rejected

Third candidate for the `batch_size >= 32` gap, on top of blocking above: split
`linalg.rs::matmul`'s 2D×2D case's output rows across `std::thread::scope` workers once there's
enough total work (`r1 * c1 * c2 >= 4,000,000`, a deliberately conservative floor) to plausibly
pay for thread spawn overhead. Row-splitting needs no cross-thread reduction (each worker owns
complete output rows end to end), so results are bit-identical to the single-threaded path
regardless of thread count - confirmed by all 525 existing tests passing unchanged.

### a real implementation bug caught before it shipped

**A real implementation bug caught before it shipped**: the first version called
`std::thread::available_parallelism()` on every matmul dispatch (to decide the thread count) -
measured directly at **~50 microseconds per call** (not cached by the standard library,
presumably a cgroup/proc filesystem read each time). Since this codebase's own actual per-call
matmul times are themselves in the tens of microseconds, this silently dominated everything:
`batch_size=1` on the full training-step benchmark went from Rust *beating* numpy (0.38x-0.50x) to
Rust *losing badly* (2.20x-2.31x) - a regression at the one batch size that mattered most,
caught only because the full benchmark suite (not just the new matmul's own isolated shapes) was
re-run before trusting the change. Fixed by caching the result once behind a `OnceLock` -
`available_parallelism_cached()` - and by checking the (cheap) flops threshold *before* ever
reading it, so tiny matmuls never pay even the cached read's small remaining cost.

### a refinement that looked right in isolation, but wasn't

**A refinement that looked right in isolation but wasn't, measured against the real target
metric**: an isolated raw-matmul-only benchmark at three shapes -

| shape | `b` size | single-threaded (blocked) | + threading |
|---|---|---|---|
| `forward_batch`-like: `(512,784)@(784,16)` | ~100KB | 1.94-2.21ms | **0.87-1.07ms - ~2x faster** |
| `accumulate_gradient_batch`-like: `(10,512)@(512,784)` | ~3.2MB | 0.94-1.09ms | 0.91-1.16ms - no clear change |
| exaggerated: `(2048,2048)@(2048,2048)` | 32MB | 2.35-2.67s | 2.06-2.68s - no real change |

- showed the small-`r1` shape (`r1=10`) getting no benefit from 8 threads doing barely more than
one row each, which looked like the natural fix: require a minimum rows-per-thread (32) before
threading engages at all, so `r1=10` falls back to single-threaded. That change *did* clean up
the isolated shape's own number - but re-measured against the actual target metric (the full
mini-batch training-step benchmark, not one matmul shape in isolation), it made
`batch_size=512`'s real ratio *worse* (1.24x-1.79x, back up from where unrestricted threading had
it), reproducibly across three separate runs. **Not adopted** - the isolated shape's own
regression didn't generalize to the composite workload it's actually part of, so the simpler
`min(available_parallelism, 8, r1)` thread count (no rows-per-thread floor) was kept instead. A
concrete instance of this codebase's own "measure, don't assume" standard applying recursively -
even a measurement-driven refinement of an earlier measurement needs checking against the real
metric, not just the proxy that motivated it.

### a memory-bandwidth ceiling, confirmed directly

The exaggerated `2048x2048` shape's lack of improvement from threading was itself directly
verified as a memory-bandwidth ceiling, not a threading bug: `os.times()`-measured CPU-time-to-
wall-time ratio was **7.17x** (near-perfect use of 8 cores) while wall-clock stayed flat - the
matrix (32MB) doesn't fit any per-core cache, so 8 cores contending for the same DRAM bandwidth
gain little from parallelism regardless of how well the threads themselves are balanced. Not a
concern for this codebase's actual matrix sizes (`dimension<=784`), included here only because
it's what surfaced the effect clearly enough to identify it.

### net result and decision

**Net result, full training-step benchmark** (`dimension=784, hidden=10`, three runs):
`batch_size=512`'s ratio improved from blocking-alone's 1.53x-1.74x range to **1.19x-1.21x** - a
further, real, reproducible ~25-30% reduction on top of blocking's own earlier win.
`batch_size=1`/`8`/`32` are unaffected (`batch_size=1`'s `learn()` path never reaches the 2D×2D
matmul case at all; the other two stay below the flops threshold at this architecture).
**Decision: adopted**, with the rows-per-thread refinement explicitly rejected per the
measurement above. Explicit SIMD intrinsics remain the one open candidate left for the rest of
this gap.

## explicit SIMD intrinsics: a real further win, with fused multiply-add kept consistent across every path

Fourth and (per docs/rust-production-cutover.md's original reasoning) last candidate for the
`batch_size >= 32` gap: `linalg.rs::matmul_2d_row_range`'s innermost accumulate step
(`out_row[col] += a_value * b_row[col]`) is a textbook AXPY pattern, vectorizable 4 `f64` lanes at
a time on this machine's AVX2. Rather than rely on LLVM's autovectorizer opportunistically doing
this (which the earlier `target-cpu=native` build-flag experiment, measured as a null, gave no
evidence was happening reliably), the accumulate step was factored into one shared helper
(`axpy_row`, called from both the blocked and unblocked loops - previously duplicated 3-line
loops) with an explicit AVX2 path using `std::arch::x86_64::_mm256_fmadd_pd`, runtime-gated by
`is_x86_feature_detected!("avx2", "fma")` with a portable scalar fallback for machines without it.

### the design question: keeping FMA semantics consistent across paths

**The design question this raised**: `_mm256_fmadd_pd` is a genuine fused multiply-add (one
rounding for `a*b+c`), not a separate multiply-then-add (two roundings) like the old `+=` loop.
Used naively, that would have made the AVX2 path produce different last-bit results than the
scalar fallback - silently breaking the bit-identical invariant that blocking (above) and
threading (above) were both explicitly built to preserve, and that a machine without AVX2 would
then train slightly different numbers than one with it, given enough steps. **Resolved by making
FMA the accumulate semantics on *every* path**, not just the new one: the scalar fallback
(`axpy_row_scalar`) now uses `f64::mul_add` (Rust's portable FMA, lowering to the same hardware
instruction when available) instead of `+=`, so the AVX2 path is a wider version of what the
scalar path already does, not a numerically different one.

### verification

**Verified directly, not just argued**: this crate has no native `cargo test` support (`pyo3`'s
`extension-module` feature doesn't link libpython, so a `cargo test` binary fails at link time -
this codebase's whole correctness suite runs through pytest instead, which is why nothing
previously caught this class of regression). Checked empirically instead: built once with the
AVX2 path forced off, captured `matmul` output as exact IEEE-754 bit patterns (`struct.pack("<d",
...).hex()`, not `pytest.approx`) across five shapes spanning both the blocked/unblocked and
remainder/exact-multiple-of-4 boundaries, including this codebase's real `batch_size=512` shape;
rebuilt with the AVX2 path restored and re-captured - **byte-for-byte identical across all five
shapes**. All 845 top-level tests (525 crate-level) also pass unchanged on the final build.

### net result, decision, and AVX-512

**Net result, full training-step benchmark** (`dimension=784, hidden=10`, `batch_size=512`,
`learn_batch`, three runs, AVX2-forced-off vs AVX2-on): per-step wall time dropped from
~31.4-33.1ms to ~25.7-26.6ms, roughly a **17-20% reduction** on top of blocking+threading's
earlier combined win. Measured directly against numpy on the same shape (three trials): the
Rust/numpy ratio moved to **0.84x-1.04x** - Rust now matches or beats numpy at `batch_size=512` in
most trials, versus the 1.19x-1.21x (Rust slower) recorded after threading alone. **Decision:
adopted** - this was the last candidate docs/rust-production-cutover.md identified for closing
the `batch_size >= 32` gap, and unlike blocking/threading it closes it in Rust's favor rather than
just narrowing it.

**AVX-512 checked and ruled out - not a further candidate on this machine**: `/proc/cpuinfo`/
`lscpu`'s flags list `avx`/`avx2` but no `avx512*` variant. This machine's CPU (AMD Ryzen 7 3700U,
Zen+/"Picasso", a 2019 mobile part) predates AMD's AVX-512 support entirely - that arrived with
Zen 4 in 2022. AVX2's 256-bit registers (4 `f64` lanes/instruction) are this hardware's actual
ceiling, so `axpy_row_avx2_fma` is already using the widest vector width available here; there is
no `_mm512_fmadd_pd` upside to chase on this machine. (Would need re-checking on different
hardware - `axpy_row`'s runtime `is_x86_feature_detected!` gate means a future AVX-512-capable
machine falls back to the AVX2 path today, not a crash, but also not the extra width until an
AVX-512 path is added.)

## SIMD for the matvec production path: a bigger win than the batch>=32 work it followed

The `batch_size >= 32` work above targeted `linalg.rs::matmul`'s 2D×2D case - a shape no current
production path actually uses. Once that work closed out, a stage-0-style measurement (same
"quantify before writing code" discipline as docs/rust-production-cutover.md's own "the decisive
finding") checked whether the *other* two matmul cases (`Matrix @ Vector`, `Vector @ Matrix`) -
untouched by any of the blocking/threading/SIMD work above - had any real headroom left, rather
than trusting `linalg.rs`'s own prior doc comment, which asserted (unverified) that they "never
exercised... at a size where it would matter."

### the assumption was wrong: real headroom existed

**That assumption was wrong.** At the real production shape (`dimension=784, hidden=16`), a
microbenchmark (200,000 reps, warmed up first) found `W @ x` (12,544 FLOPs) cost **~10.2us**, of
which **~97% was the matmul itself**, not PyO3 call overhead (a bare 1-element-read FFI round
trip measured ~0.29us) - and this one matvec call was **~3.5x slower than numpy** at this exact
shape, accounting for ~97% of a fused `layer_forward` call's cost and ~27-29% of a full `learn()`
step's wall-clock. The `Vector @ Matrix` case (unused by the current class design, per the
interface subset's own note) wasn't checked at production scale for the same reason it was never
optimized before: nothing calls it today.

### why matrix@vector needed its own reduction scheme

**Why the matrix@vector case couldn't just reuse `axpy_row`.** `axpy_row` vectorizes across the
*output* dimension while keeping the reduction over `k` strictly sequential - that's what makes
it bit-identical regardless of which path runs. A per-row dot product's reduction dimension *is*
the dimension SIMD would vectorize, so there's no grouping of 4 parallel lanes that's also
bit-identical to a naive left-to-right sequential sum (float64 addition isn't associative - a
different grouping is a different value, typically by ~1 ULP). Resolved by picking one canonical
grouping - 4 interleaved partial sums (lane `j` accumulates indices `j, j+4, j+8, ...`), combined
pairwise at the end - and using that *same* grouping in both the scalar fallback
(`dot_product_scalar`) and the new AVX2+FMA path (`dot_product_avx2_fma`), via a shared
`combine_lanes`/`dot_product_tail` so the two paths can't accidentally diverge. This does change
the *value* matmul produces from the old naive-sequential-sum baseline (last-few-ULPs noise) - the
same category of already-accepted risk as numpy's own internal reduction order not matching
Python's sequential sum (see "summation-order rounding" in
[vectorized array-based classes](vectorized-array-classes.md#numerical-parity-validation)), not a
new one, and every parity check against numpy/the pure-Python reference already tolerates it via
`rtol`, not exact equality. The `Vector @ Matrix` case needed no new reduction logic at all - it
turned out to be structurally identical to `axpy_row`'s own row-scaling accumulate (one output
"row" instead of many, `b`'s stride over `k` staying row-contiguous), so it was wired straight
through the existing, already-bit-identical `axpy_row` instead.

### verification

**Verified the same way as the earlier SIMD work**: built once with the AVX2 path forced off,
captured `matmul`'s `Matrix @ Vector` output as exact IEEE-754 bit patterns across 7 shapes
(spanning exact-multiples-of-4, remainders, and a 1x1 degenerate case, including the real
`16x784`/`10x16` production shapes); rebuilt with AVX2 restored and re-captured - **byte-for-byte
identical across all 7 shapes**. All 845 tests (525 crate-level) pass unchanged.

### net result and decision

**Net result:**

| measurement | before | after |
|---|---|---|
| raw matvec, `16x784 @ 784` (production shape) | ~10.2us | **~4.3us** (2.4x faster) |
| Rust/numpy ratio, same shape | 3.5x slower | **1.48x slower** |
| full `learn()` step | ~73-75us | ~67us (~9% faster) |
| UCI digits training run, Rust/numpy | 3.40x | **~3.6x** (3 trials: 3.53x-3.64x) |
| real MNIST training run, Rust/numpy | 1.31x | **~2.6x** (5 trials: 2.59x-2.66x) |

Test accuracy stayed statistically indistinguishable (UCI digits: 96.94% numpy vs. 96.66% Rust;
real MNIST: 92.88% numpy vs. 93.18% Rust - both within the same RNG-mismatch noise band this
codebase's other Rust-vs-numpy comparisons already show, not a regression). The real-MNIST result
is the more striking one: `dimension=784, hidden=30` is exactly the shape where this matvec
dominates most, and the Rust/numpy ratio nearly doubled. **Decision: adopted.** Unlike the
`batch_size >= 32` work above (which targeted a shape no production path uses), this closes a gap
in the shape *every* production training step actually runs.
