# Rust production cutover

[← back to README](../README.md)

Workplan for making [the Rust array core](rust-array-core.md) (`perceptron_array`) the primary
array backend actual training/demo code uses, per
[vectorization](vectorization.md#decision)'s own recorded decision - while keeping
[the numpy-backed vectorized classes](vectorized-array-classes.md) permanently available as the
comparison point, for both wall-clock performance and, to the extent the RNG mismatch allows,
trained results. Nothing in `perceptron/` changes as a result of this document itself - it's the
plan, not the implementation.

## the decisive finding this plan has to answer first

Before any wiring work, a throwaway benchmark measured the *current* Rust core - real
`perceptron_array.Array`, called from Python exactly the way a line-for-line port of
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
(see [research and analysis](research-and-analysis.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact)
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
closer than this plan originally assumed. Phase 0 remains a required, measured gate before any
class is built: finish fusing (0b), re-measure, and only proceed to building
`RustArrayMultiClassBackpropClassifierNetwork` if the result actually beats numpy at realistic
sizes. If it doesn't, the honest outcome is the same kind of measured null this codebase already
has several of (momentum, L2, Xavier/Glorot) - correctness-validated, not adopted, recorded as
such rather than forced through.

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
`perceptron/model/array_layer.py`'s own `ArrayLayer` methods, the actual production reference
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
[research and analysis](research-and-analysis.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact)
for the full numbers and how they were produced. What this means for phase 1 - build it for the
per-example path only, build it anyway as a correctness-first standalone (the same treatment
momentum/L2/Xavier-Glorot got), or not build it - is recorded as an open decision, not resolved by
this document alone.

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

### if phase 0 doesn't close the gap

A real possible outcome, not to be argued away: if a reordered loop plus fused per-layer calls
still don't beat numpy at realistic batch sizes, that's the answer, recorded honestly in
[research and analysis](research-and-analysis.md) the same way momentum/L2/Xavier-Glorot's null
results are - "correctness-validated, not adopted for performance reasons" - rather than shipped
as a production regression to satisfy this plan's original premise. A follow-on decision (SIMD
intrinsics, a blocked/tiled matmul, batching every operation across the whole training set instead
of per-layer) would be a separate, later piece of work, not assumed as part of this one.

## phase 1: `RustArrayLayer` / `RustArrayMultiClassBackpropClassifierNetwork`

Only started once phase 0 clears its gate. Mirrors `ArrayLayer`/
`VectorizedMultiClassBackpropClassifierNetwork`'s design exactly (same method names, same
external contract: `learn`, `learn_batch`, `randomize`/`randomized`, `classify_state`,
`predict_probabilities`, `snapshot`/`restore`, `save`/`load`), but every method body is a single
call into one of phase 0's fused Rust functions instead of a composition of `Array` operators.
Reuses `perceptron/train.py`'s `train_linear_classifier_network`/`train_backprop_network_mini_batch`
completely unchanged - both already work via duck typing against exactly this contract, the same
way every prior sibling network (including `VectorizedMultiClassBackpropClassifierNetwork`
itself) plugs in.

- `randomize()` calls `perceptron_array.uniform` - the RNG-non-reproducibility caveat
  ([the Rust core](rust-array-core.md#the-rng-exception)) applies directly here and shapes the
  parity strategy below.
- `save()`/`load()`: own JSON envelope via `Array.tolist()`/`Array(nested_list)`, the same
  pattern `VectorizedMultiClassBackpropClassifierNetwork`/`ConvMultiClassBackpropClassifierNetwork`
  already use for a class `save_model_json` can't serve.

## phase 2: numerical parity, given the RNG mismatch can't be closed

Two tiers, matching what's actually checkable:

- **Tier 1 - exact, always checkable.** Inject identical fixed weights/inputs (never
  `randomize()`) into both `VectorizedMultiClassBackpropClassifierNetwork` (numpy) and
  `RustArrayMultiClassBackpropClassifierNetwork` (Rust), and require the two to produce
  bit-identical or float64-noise-close forward/backward/gradient/apply results at both
  `batch_size=1` and `batch_size>1`. This validates the *math* independent of the RNG problem,
  and is the required regression gate before any accuracy or performance claim from either
  network is trusted - the same discipline every prior numerical-parity gate in this codebase
  uses.
- **Tier 2 - statistical, not exact.** Full training runs from each class's own `randomize()`,
  same dataset/hyperparameters, multiple independent seeds per class, compared by trajectory
  shape and final-accuracy distribution (mean/stdev across seeds) rather than seed-for-seed
  weight identity - the honest substitute for "same seed → same weights," which the RNG mismatch
  makes structurally impossible (see [the Rust core](rust-array-core.md#the-rng-exception)).
  "Identical accuracy trajectory" is not an achievable or meaningful claim here; "statistically
  indistinguishable accuracy distribution, at the wall-clock cost measured in phase 0/3" is.

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
- Once real numbers exist, record them in [research and analysis](research-and-analysis.md), the
  same place every other measured comparison in this codebase lives - not the extrapolated
  figures this plan itself contains.

## phase 4: the actual cutover

Only `demo_vectorized_uci_digit_recognition.py` and `demo_vectorized_mnist_recognition.py`
currently instantiate the array-based classes (checked directly - nothing else in `perceptron/`
does), so "production" concretely means: whichever of those two demos' roles becomes the primary,
recommended path switches its default network to `RustArrayMultiClassBackpropClassifierNetwork`,
with the numpy-backed class kept alongside as the explicit, permanent comparison point per the
user's own requirement - not removed, not deprecated. `structure.md`'s "vectorized array-based
classes" section gets rewritten once this lands, to describe the Rust class as the primary path
and numpy as the comparison, replacing this plan document's own role the same way every prior
built-and-shipped plan in this codebase has been folded into `structure.md`.

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
   [research and analysis](research-and-analysis.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact).
2. **Done.** Fused forward functions (`layer_forward`, `layer_forward_batch`) + parity tests.
3. **Done.** Fused backward/gradient functions (`layer_output_delta`, `layer_hidden_delta`,
   `layer_hidden_delta_batch`, `layer_accumulate_gradient`, `layer_accumulate_gradient_batch`,
   `layer_apply_accumulated_gradient`) + parity tests - delivered together with stage 2 rather
   than as a separate PR, since both were built in the same pass once the debug-build fix (stage
   1) reset the whole gate's premise.
4. **Done.** Go/no-go benchmark re-run against numpy at realistic batch sizes; recorded in "0b.
   fuse each layer operation into one Rust call" above and
   [research and analysis](research-and-analysis.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact) -
   a split result (faster at `batch_size=1`, slower and widening from `batch_size=32` up), not a
   clean go or no-go, left as an open decision rather than forced into either bucket.
5. *(pending the phase-1 decision above)* `RustArrayLayer`/
   `RustArrayMultiClassBackpropClassifierNetwork` + tier-1 exact parity tests.
6. Tier-2 statistical parity + accuracy validation (UCI digits first, then real MNIST).
7. The benchmark demo(s) (phase 3).
8. `structure.md`/`vectorization.md` updated to describe the shipped result.
9. *(only if adopted)* Retarget the two vectorized demos' primary path to the Rust-backed class.
