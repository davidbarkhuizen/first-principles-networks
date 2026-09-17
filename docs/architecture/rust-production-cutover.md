# Rust production cutover

[← back to README](../../README.md)

Workplan for making [the Rust array core](rust-array-core.md) (`indrajala_ml_array`) the primary
array backend actual training/demo code uses, per
[vectorization](vectorization.md#decision)'s own recorded decision - while keeping
[the numpy-backed vectorized classes](vectorized-array-classes.md) permanently available as the
comparison point, for both wall-clock performance and, to the extent the RNG mismatch allows,
trained results. Nothing in `indrajala_ml/` changes as a result of this document itself - it's the
plan, not the implementation.

## the decisive finding this plan has to answer first

Before any wiring work, a throwaway benchmark measured the *current* Rust core - real
`indrajala_ml_array.Array`, called from Python exactly the way a line-for-line port of
`array_layer.py` would call it (`W @ x + b`, then `1.0 / (1.0 + exp(-z))` built from individual
`Array` operators) - against the same forward pass in numpy, at this codebase's real production
layer size (`dimension=784`, `hidden=16`):

| batch size | numpy | naive Rust (per-op composition) | ratio (Rust/numpy) |
|---|---|---|---|
| 1 | 12.2 us | 800 us | 65.7x **slower** |
| 8 | 25.0 us | 3464 us | 138.3x **slower** |
| 32 | 60.6 us | 12702 us | 209.7x **slower** |
| 128 | 164.2 us | 50179 us | 305.5x **slower** |
| 512 | 606.9 us | 203550 us | 335.4x **slower** |

(Correctness was checked alongside timing - max elementwise difference ≤ 1e-15 at every batch
size, floating-point noise, not a bug.) This is the opposite of "primary in production": as
currently composed, the Rust core appeared to be two to three orders of magnitude slower than
numpy, with the gap widening with batch size.

**Corrected: this benchmark was run against a debug build.** `./cli setup`/`./cli build-rust` ran
a plain `maturin develop` - a debug (unoptimized) build - and re-measuring during phase 0
(see [research and analysis](../research/research-rust-performance.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact)
for the full four-way table) found that build mode, not the per-op composition itself, was almost
the entire gap: the *same* pre-fix code under a **release** build was only ~3-10x slower, not
65-335x. `./cli build-rust`/`./cli setup` now build with `maturin develop --release`; the table
above is kept as-recorded (it's what was actually measured, and the debug-build discovery is
itself a real finding worth keeping legible) rather than silently edited, but it must not be read
as the current gap. The two causes below are both real and both worth fixing, just at a much
smaller scale than first measured:

1. **Per-call FFI/marshaling overhead, multiplied by call count.** A line-for-line port composes
   a forward pass from 4-6 separate `Array` method/operator calls (`@`, `+`, a subtraction for
   `-z`, `exp`, another `+`, a `/`), each crossing the Python/Rust boundary and allocating a new
   `Array`. This is the same granularity mistake vectorization itself was built to fix ("one
   Python object, one method call, per node") recurring one level down ("one Rust array, one FFI
   call, per operation").
2. **`linalg.rs::matmul`'s 2D×2D loop order was cache-hostile.** It iterated `row -> col -> k`,
   reading `b.data[k*c2+col]` on the innermost loop - a stride-`c2` access, the worst of the six
   possible loop orderings for cache locality. **Fixed** (stage 1 below): reordered to
   `row -> k -> col`, accumulating into a whole output row at once and reading both operands
   row-contiguously. Re-measured on the corrected release-build baseline: ~3-10x slower before the
   reorder, ~2-7x slower after it, at batch sizes 1/8/32/128/512 (see research and analysis for
   the full numbers) - a real improvement, not decisive on its own.

**Consequence for this plan:** "retarget production to the Rust core" cannot mean "swap `numpy` →
`Array` call-for-call carelessly" - but the corrected baseline means that bar may already be much
closer than this plan originally assumed. Phase 0 remains required regardless: fix what can
cheaply be fixed (0a, 0b) and measure honestly, so phase 1 is built on top of the best cheap
version of this core, not the accidentally-debug one.

**Adoption is unconditional, not gated on this benchmark - clarified 2026-09-16.** The original
framing below ("only proceed... if it beats numpy," "if phase 0 doesn't close the gap") predates
this clarification and is kept only as a record of the plan's own evolving reasoning, not as the
live decision rule. The actual intent: the Rust core replaces numpy as the production backend
*always*, regardless of the current benchmark split; numpy stays on permanently as the benchmark
comparison point (unchanged from the original decision); and any remaining performance gap (see
phase 0b's split result below - naive matmul losing to numpy's BLAS at `batch_size >= 32`) is
closed incrementally over time (SIMD, blocking/tiling, threading), not resolved before shipping
phase 1. Phase 1 proceeds on that basis.

## scope decision: a new sibling class, not a retrofit

Build `RustArrayLayer`/`RustArrayMultiClassBackpropClassifierNetwork` as new, standalone classes -
the same relationship `VectorizedMultiClassBackpropClassifierNetwork` has to
`MultiClassBackpropClassifierNetwork`, not a modification of it. `ArrayLayer`/
`VectorizedMultiClassBackpropClassifierNetwork` stay exactly as they are, permanently: that's
[the recorded decision](vectorization.md#decision) for numpy's role as a standing benchmark
mirror, and the pattern every prior addition in this codebase already follows (never retrofit,
always add a sibling - ReLU, softmax, momentum, L2, conv, and the numpy-backed classes themselves
were all built this way). Retrofitting `ArrayLayer` to take a backend parameter was considered
and rejected: it would mix two array libraries' semantics and error modes into one class, and
undo the "numpy stays untouched" half of the recorded decision for no real benefit.

## phase 0: fix the two measured bottlenecks in the Rust core itself (required gate)

### 0a. reorder `matmul`'s loop nest

Change the 2D×2D case in `linalg.rs::matmul` from `row -> col -> k` (reads `b` with a
stride-`c2` access on the innermost loop) to `row -> k -> col` (accumulates into a whole output
row at once, reading both `a` and `b` row-contiguously) - a well-known, zero-dependency,
mechanical reordering, matching this crate's own "naive but correct first, cache-friendlier loop
order as a later, separately-measured refinement" framing it already anticipated. Re-run the
benchmark above immediately after this one change, in isolation, before touching anything else -
measure the delta, don't assume it closes the gap. Set expectations honestly regardless: no naive
loop (no SIMD, no blocking, no threads) will approach OpenBLAS - the open question this step
answers is how much of the *current* multi-hundred-x gap a well-ordered loop closes, not whether
it reaches parity.

### 0b. fuse each layer operation into one Rust call

**Done.** Fused per-layer operations added to the crate (`fused.rs`) - `#[pyfunction]`s doing the
*entire* computation in one Rust function, returning one `Array`, instead of composing it from
several `Array` operator calls in Python:

| function | replaces this Python-level composition |
|---|---|
| `layer_forward(W, x, b) -> Array` | `sigmoid(W @ x + b)`, single-example |
| `layer_forward_batch(W, X, b) -> Array` | `sigmoid(X @ W.T + b)`, batched |
| `layer_output_delta(a, reference) -> Array` | `(a - reference) * a * (1 - a)` - one formula, used for both single-example and batched (shape-agnostic) |
| `layer_hidden_delta(next_W, next_delta, a) -> Array` | `(next_W.T @ next_delta) * a * (1 - a)`, single-example |
| `layer_hidden_delta_batch(next_W, next_delta_batch, a_batch) -> Array` | `(next_delta_batch @ next_W) * A * (1 - A)`, batched - a genuinely different call shape from the single-example version above (no transpose on `next_W`, operand order swapped), not just a shape-agnostic reuse, so it needed its own function (added beyond the plan's original table, which only sketched the single-example case) |
| `layer_accumulate_gradient(delta, input_activation, grad_W, grad_b) -> (Array, Array)` | `grad_W += outer(delta, input_activation); grad_b += delta`, single-example |
| `layer_accumulate_gradient_batch(delta_batch, input_activation_batch, grad_W, grad_b) -> (Array, Array)` | `grad_W += delta_batch.T @ input_activation_batch; grad_b += delta_batch.sum(axis=0)` (added beyond the plan's original table for the same reason as `layer_hidden_delta_batch`) |
| `layer_apply_accumulated_gradient(W, b, grad_W, grad_b, learning_rate, batch_size) -> (Array, Array)` | `W -= lr * grad_W / batch_size; b -= lr * grad_b / batch_size` - shape-agnostic, covers both |

This directly targets cause 1 above (FFI-crossing count), and sidesteps needing `__neg__`/
`__radd__`/`__rtruediv__` on `Array` at all (`fused.rs` inlines its own `1.0 / (1.0 + (-z).exp())`
sigmoid rather than composing it from `Array` operators) - see "operators the current `Array` is
missing" below for why those matter if this fused approach is *not* taken. Each function has the
same parity-test treatment as every existing operation in this crate
(`tests/test_fused_layer_ops.py`): a randomized sweep checked directly against
`indrajala_ml/model/array_layer.py`'s own `ArrayLayer` methods, the actual production reference
these functions replace, not just against a formula written independently.

**The go/no-go benchmark, on the corrected release-build baseline** (see "the decisive finding"
above): `layer_forward_batch` alone vs. numpy, same architecture (`dimension=784, hidden=16`):

| batch size | numpy | fused Rust | ratio |
|---|---|---|---|
| 1 | 14.4 us | 27.7 us | 1.9x slower |
| 8 | 21.9 us | 45.1 us | 2.1x slower |
| 32 | 57.6 us | 135.1 us | 2.4x slower |
| 128 | 154.2 us | 666.4 us | 4.3x slower |
| 512 | 489.1 us | 2082.5 us | 4.3x slower |

Fusing closed some further ground over 0a alone but not much - matmul, still a naive triple loop,
is now the dominant remaining cost at these batch sizes, and fusing doesn't touch it. A fuller
comparison - one whole mini-batch training step (`layer_forward_batch` + `layer_output_delta` +
`layer_accumulate_gradient_batch` + `layer_apply_accumulated_gradient`, `dimension=784,
hidden=10`, this codebase's real output-layer shape) vs. the equivalent numpy `ArrayLayer` method
sequence, tells a more textured story:

| batch size | numpy | fused Rust | ratio |
|---|---|---|---|
| 1 | 68.0 us | 30.5 us | **0.45x - Rust is faster** |
| 8 | 67.2 us | 64.6 us | ~parity |
| 32 | 108.3 us | 189.6 us | 1.75x slower |
| 128 | 201.5 us | 798.9 us | 4.03x slower |
| 512 | 1740.3 us | 3478.3 us | 2.00x slower |

(Correctness held throughout both tables - max weight difference ≤ 5e-16, float64 noise.) This is
the split outcome "risks and open questions" below already flagged as a real possibility before
it was measured: **the fused Rust core beats numpy for per-example training (`batch_size=1`,
`ArrayLayer.learn`'s own shape) but loses to it, by a widening margin, for realistic mini-batch
training (`batch_size >= 32`, `learn_batch`'s shape)** - the opposite split from what the risks
section guessed (it expected batching to be Rust's strength and per-example its weakness). See
[research and analysis](../research/research-rust-performance.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact)
for the full numbers and how they were produced. **Resolved 2026-09-16 (see "the decisive
finding" above): phase 1 is built in full** (both `learn` and `learn_batch`), not scoped to only
the per-example path this benchmark currently favors - adoption is unconditional, and the
`batch_size >= 32` gap is tracked as follow-on optimization work, not a reason to withhold the
batched path.

### operators the current `Array` is missing (relevant only if phase 0's fused approach is skipped)

Empirically confirmed missing, not assumed: `Array` has `__add__`/`__sub__`/`__rsub__`/`__mul__`/
`__rmul__`/`__truediv__` but no `__neg__`, `__radd__`, or `__rtruediv__` - so `-z`, `1.0 + arr`,
and `1.0 / arr` all raise `TypeError` today (`0.0 - z` works as a substitute for `-z`, via the
existing `__rsub__`). Numpy's `ndarray` supports the full reflected set for free, which is why
`array_layer.py`'s numpy code never had to think about this. If phase 0 is skipped or deferred
and a line-for-line port is built anyway (accepting the measured regression, e.g. for a
correctness-only milestone), these three operators would need adding to the crate first, or the
Python-side formula rewritten to avoid them (`0.0 - z` instead of `-z`; restructuring `1/(1+e)` as
`e.__add__(1.0).__rtruediv__`-style workarounds, which is exactly the kind of awkward code the
fused approach in 0b avoids needing at all).

### the matmul gap that remains

A real, measured outcome, not argued away: even after 0a and 0b, `batch_size >= 32` training
still loses to numpy, by a widening margin, because matmul is still a naive triple loop with no
SIMD/blocking/threading (see phase 0b's benchmark table above). Per the 2026-09-16 clarification
above, this is not a reason to withhold adoption - it's tracked as follow-on optimization work
(SIMD intrinsics, a blocked/tiled matmul, batching every operation across the whole training set
instead of per-layer), a separate, later piece of work from phase 1 itself, recorded honestly in
[research and analysis](../research/research-and-analysis.md) rather than glossed over.

## phase 1: `RustArrayLayer` / `RustArrayMultiClassBackpropClassifierNetwork`

**Done** (`indrajala_ml/model/rust_array_layer.py`, `indrajala_ml/model/rust_array_multiclass_backprop_classifier_network.py`),
built unconditionally per the 2026-09-16 clarification above, not gated on phase 0's benchmark
split. Mirrors `ArrayLayer`/
`VectorizedMultiClassBackpropClassifierNetwork`'s design exactly (same method names, same
external contract: `learn`, `learn_batch`, `randomize`/`randomized`, `classify_state`,
`predict_probabilities`, `snapshot`/`restore`, `save`/`load`), but every method body is a single
call into one of phase 0's fused Rust functions instead of a composition of `Array` operators.
Reuses `indrajala_ml/train.py`'s `train_linear_classifier_network`/`train_backprop_network_mini_batch`
completely unchanged - both already work via duck typing against exactly this contract, the same
way every prior sibling network (including `VectorizedMultiClassBackpropClassifierNetwork`
itself) plugs in.

- `randomize()` calls `indrajala_ml_array.uniform` - the RNG-non-reproducibility caveat
  ([the Rust core](rust-array-core.md#the-rng-exception)) applies directly here and shapes the
  parity strategy below.
- `save()`/`load()`: own JSON envelope via `Array.tolist()`/`Array(nested_list)`, the same
  pattern `VectorizedMultiClassBackpropClassifierNetwork`/`ConvMultiClassBackpropClassifierNetwork`
  already use for a class `save_model_json` can't serve.

## phase 2: numerical parity, given the RNG mismatch can't be closed

Two tiers, matching what's actually checkable:

- **Tier 1 - exact, always checkable. Done.** `tests/test_rust_array_multiclass_backprop_model.py`
  injects identical fixed weights/inputs (never `randomize()`) into both
  `VectorizedMultiClassBackpropClassifierNetwork`'s pure-Python-equivalent reference
  (`MultiClassBackpropClassifierNetwork`) and `RustArrayMultiClassBackpropClassifierNetwork`
  (Rust) and checks bit-close (`rtol=1e-9`) agreement after every single-example `learn()` step
  (`batch_size=1`) and every `learn_batch()` call (`batch_size=8`), not just at the end.
- **Tier 2 - statistical, not exact. Done.** Full training runs from each class's own
  `randomize()`, same dataset/hyperparameters, multiple independent seeds per class, compared by
  final-accuracy distribution (mean/stdev across seeds) rather than seed-for-seed weight identity
  - the honest substitute for "same seed → same weights," which the RNG mismatch makes
  structurally impossible (see [the Rust core](rust-array-core.md#the-rng-exception)). Measured at
  both UCI digits (8 seeds) and real MNIST (3 seeds, 1 epoch each) scale - see [research and
  analysis](../research/research-rust-performance.md#phase-2-tier-2-real-per-example-training-is-a-genuine-win-at-both-scales-measured):
  **3.40x faster at UCI digits, 1.31x faster at real MNIST**, both with statistically
  indistinguishable test accuracy. This landed as a genuine, not just unconditionally-accepted,
  win - both of this codebase's actual production training paths use `learn()`'s per-example
  (`batch_size=1`) shape exclusively (checked directly - neither existing vectorized demo calls
  `learn_batch`), which is exactly the regime phase 0b's benchmark found the Rust core ahead in.

## phase 3: the benchmark harness

Extends the existing pattern rather than inventing a new one - `demo_vectorized_uci_digit_recognition.py`
already trains pure-Python and numpy side by side, at the same seed/hyperparameters, reporting
accuracy and wall-clock together. Add a third column, not a new pattern:

- `demo_rust_vs_vectorized_uci_digit_recognition.py` - pure-Python /
  `VectorizedMultiClassBackpropClassifierNetwork` (numpy) /
  `RustArrayMultiClassBackpropClassifierNetwork` (Rust), same architecture as the existing UCI
  demo, reporting all three wall-clock times and accuracies side by side.
- A real-MNIST counterpart mirroring `demo_vectorized_mnist_recognition.py`'s one-real-epoch
  shape, once the UCI-scale comparison is trusted.
- The array-backend comparison stays in its own demo pair, not folded into
  `demo_backprop_variant_comparison.py` - that demo's scope is model/loss/init variants, a
  different axis of comparison from array backend/performance.
- Once real numbers exist, record them in [research and analysis](../research/research-and-analysis.md), the
  same place every other measured comparison in this codebase lives - not the extrapolated
  figures this plan itself contains.

## phase 4: the actual cutover

**Done.** Before this phase, only `demo_vectorized_uci_digit_recognition.py` and
`demo_vectorized_mnist_recognition.py` instantiated the array-based classes (checked directly -
nothing else in `indrajala_ml/` did), so "production" had to concretely mean something about those
two demos' role. Resolved via phase 3's own new demos rather than by mutating the old ones:
`demo_rust_vs_vectorized_uci_digit_recognition.py`/`demo_rust_vs_vectorized_mnist_recognition.py`
now train/save/interactively-demo `RustArrayMultiClassBackpropClassifierNetwork` as the primary
network (with both pure-Python and numpy trained alongside for comparison, one column further
than the demos they extend), which already *is* "the default network is Rust, numpy kept as the
explicit, permanent comparison" - stage 9's own goal - without needing to also rewrite the
original two-way demos, which stay exactly as they are (numpy-specific historical record, not
deprecated, same "kept alongside, never removed" treatment numpy itself gets). `structure.md`'s
"vectorized array-based classes" section is rewritten to describe the Rust class as the primary
path and numpy as the comparison - see [structure](../project/structure.md#vectorized-array-based-classes).

## risks and open questions

- Whether phase 0's fused-call approach can close enough of the gap given matmul is still naive
  (no SIMD/blocking/threading) - genuinely unmeasured until it's done.
- Whether per-example (`batch_size=1`) training can ever beat numpy even after fusing, since a
  single Python↔Rust call still costs something no pure-Python-calling-numpy path pays - if not,
  "primary in production" may end up meaning "for `learn_batch`-shaped batched training," not
  `learn()`'s per-example path, which would need stating plainly rather than glossed over.
- The RNG mismatch means "primary in production" can never mean bit-identical outcomes to a numpy
  run from the same seed - only statistically comparable ones (phase 2, tier 2). Anyone expecting
  reproducible-to-the-bit swaps between backends needs to know this going in.

## delivery stages (each its own PR, per this repo's practice)

1. **Done.** Matmul loop reorder (0a) + re-benchmark. Along the way, caught and fixed a bigger
   issue than the reorder itself: `./cli build-rust`/`./cli setup` were building this crate in
   debug mode, which is what made the original gate benchmark read as 65-335x slower than numpy
   instead of the real ~3-10x - see "the decisive finding" above and
   [research and analysis](../research/research-rust-performance.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact).
2. **Done.** Fused forward functions (`layer_forward`, `layer_forward_batch`) + parity tests.
3. **Done.** Fused backward/gradient functions (`layer_output_delta`, `layer_hidden_delta`,
   `layer_hidden_delta_batch`, `layer_accumulate_gradient`, `layer_accumulate_gradient_batch`,
   `layer_apply_accumulated_gradient`) + parity tests - delivered together with stage 2 rather
   than as a separate PR, since both were built in the same pass once the debug-build fix (stage
   1) reset the whole gate's premise.
4. **Done.** Go/no-go benchmark re-run against numpy at realistic batch sizes; recorded in "0b.
   fuse each layer operation into one Rust call" above and
   [research and analysis](../research/research-rust-performance.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact) -
   a split result (faster at `batch_size=1`, slower and widening from `batch_size=32` up).
   Per the 2026-09-16 clarification above, adoption is unconditional, so this data is a progress
   record, not a gate phase 1 had to clear.
5. **Done.** `RustArrayLayer`/`RustArrayMultiClassBackpropClassifierNetwork` + tier-1 exact
   parity tests (`tests/test_rust_array_multiclass_backprop_model.py`) - bit-close (`rtol=1e-9`)
   against `MultiClassBackpropClassifierNetwork`'s pure-Python reference after every single-example
   `learn()` step (100 steps) and every `learn_batch()` call (20 batches of 8), not just at the
   end.
6. **Done.** Tier-2 statistical parity + accuracy validation (UCI digits first, then real
   MNIST) - see phase 2 above.
7. **Done.** The benchmark demo(s) (phase 3) - real measured results: 3.30x faster at UCI digits,
   1.39x faster at real MNIST.
8. **Done.** `structure.md`/`vectorization.md`/`rust-array-core.md`/`README.md` updated to
   describe the shipped result.
9. **Done, via phase 3's new demos rather than mutating the old ones** - see phase 4 above.

**All nine delivery stages are done as of 2026-09-16.** `RustArrayMultiClassBackpropClassifierNetwork`
is this codebase's production array-backed network; `VectorizedMultiClassBackpropClassifierNetwork`
(numpy) remains permanently as the benchmark comparison. The follow-on work flagged above -
closing the naive matmul's remaining gap at `batch_size >= 32` - was never part of this plan, and
is itself now done too (same day, cache-blocking + threading + explicit SIMD intrinsics, see
[structure](../project/structure.md#possible-next-steps) and
[research and analysis](../research/research-rust-performance.md#explicit-simd-intrinsics-a-real-further-win-with-fused-multiply-add-kept-consistent-across-every-path)):
`batch_size=512`'s Rust/numpy ratio moved from a widening multi-x loss to 0.84x-1.04x. A second,
larger follow-on win landed the same day on the `Matrix @ Vector` matmul case - `self.W @ x`, the
shape phase 2's own `batch_size=1` benchmark above already runs on every `learn()` call - moving
the real per-example training-run speedup this phase measured (3.40x/1.31x) to ~3.6x/~2.6x (see
[research and
analysis](../research/research-rust-performance.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)).
