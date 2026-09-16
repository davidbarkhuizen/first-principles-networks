# an array-based (Rust-matmul-backed) Adam sibling

[← back to README](../README.md)

**Status: all five stages done.** Written up front as a design/measurement plan before any of it
existed, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [the Rust
production cutover plan](rust-production-cutover.md) for precedent) - updated here with
stage-by-stage status notes as it was executed.

## why this, and why now

`AdamBackpropClassifierNetwork` (see [Adam optimizer](adam-optimizer.md)) is still built on the
per-node `BackpropNode` object graph - one Python object, one method call, per weight, per step.
`VectorizedMultiClassBackpropClassifierNetwork`'s `ArrayLayer` (numpy) and its
`RustArrayMultiClassBackpropClassifierNetwork` sibling's `RustArrayLayer` (Rust-matmul-backed,
AVX2-accelerated - see [research and
analysis](research-rust-performance.md)) already do real batched forward/backward as whole-array
matrix operations, proven and measured faster than numpy on real training runs (~3.6x at UCI
digits scale, ~2.6x at real MNIST scale). [Adam's stage 5
result](research-adam-optimizer.md#adam-at-real-mnist-ensemble-scale-the-proxy-result-holds-stage-5-of-the-adam-optimizer-workplan)
found Adam stays robust across `batch_size` where sigmoid collapses, but explicitly noted this
isn't yet a wall-clock win on its own - mini-batching alone only changes *when* the weight write
happens, not whether forward/backward gets vectorized across the batch. This is the missing piece
that would make large batches genuinely cheap in wall-clock terms too, not just accuracy-robust.

## scope

**Numpy-backed first** (`AdamArrayLayer`, an `ArrayLayer` sibling, plus
`AdamVectorizedMultiClassBackpropClassifierNetwork`), stage 2 - mirroring how every existing
array-based capability here was built numpy-first, then ported to Rust as its own later,
independently-measured effort (see [the Rust production cutover
plan](rust-production-cutover.md)'s own phased history). The Rust-backed
`AdamRustArrayLayer`/`AdamRustArrayMultiClassBackpropClassifierNetwork` counterpart, stage 5, is
now also built - see "the Rust-matmul-backed counterpart" below. Scoped to
`MultiClassBackpropClassifierNetwork`'s array-based line, not `BackpropClassifierNetwork`'s -
`VectorizedMultiClassBackpropClassifierNetwork`/`RustArrayMultiClassBackpropClassifierNetwork` are
themselves multi-class-only, so there's no single-output array-based sibling to extend Adam onto
without first building one (out of scope here, still).

## design: mirrors `RustArrayMultiClassBackpropClassifierNetwork`'s own precedent - a new class, not a swap point

`VectorizedMultiClassBackpropClassifierNetwork.__init__` hardcodes `ArrayLayer(size,
previous_size)` construction - there's no `layer_cls`-style extension point at this level the way
`BackpropNetworkBase`'s `hidden_layer_cls`/`output_layer_cls` gives the per-node hierarchy.
`RustArrayMultiClassBackpropClassifierNetwork` itself is not a subclass of
`VectorizedMultiClassBackpropClassifierNetwork` with a swapped layer type - it's a wholly separate
class duplicating the same external contract (`learn`/`learn_batch`/`classify_state`/
`predict_probabilities`/`randomize`/`randomized`/`snapshot`/`restore`/`save`/`load`) against a
different layer implementation. `AdamVectorizedMultiClassBackpropClassifierNetwork` follows that
same precedent - a new, standalone class, not a modification to the existing one - consistent with
every other sibling in this codebase being purely additive.

`AdamArrayLayer(ArrayLayer)` overrides only `apply_accumulated_gradient`, mirroring
`make_adam_node_cls`'s exact per-parameter formula but as whole-array numpy ops instead of a
per-weight Python loop:

```python
class AdamArrayLayer(ArrayLayer):
    def __init__(self, size: int, input_size: int, beta1: float, beta2: float, epsilon: float) -> None:
        super().__init__(size, input_size)
        self._beta1, self._beta2, self._epsilon = beta1, beta2, epsilon
        self._m_W = np.zeros((size, input_size))
        self._v_W = np.zeros((size, input_size))
        self._m_b = np.zeros(size)
        self._v_b = np.zeros(size)
        self._t = 0

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self._t += 1
        bias_correction1 = 1 - self._beta1 ** self._t
        bias_correction2 = 1 - self._beta2 ** self._t

        g_W = self._grad_W / batch_size
        self._m_W = self._beta1 * self._m_W + (1 - self._beta1) * g_W
        self._v_W = self._beta2 * self._v_W + (1 - self._beta2) * g_W * g_W
        m_hat_W = self._m_W / bias_correction1
        v_hat_W = self._v_W / bias_correction2
        self.W -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self._epsilon)

        # same formula, bias term (g_b/m_b/v_b), omitted here for brevity - see adam_layer.py's
        # own bias handling for the exact parallel structure to mirror
        ...

        self._reset_gradient_accum()
```

`beta1`/`beta2`/`epsilon` default to Kingma & Ba's own published values, matching
`AdamBackpropClassifierNetwork`'s own resolved posture (see [Adam
optimizer](adam-optimizer.md#hyperparameters)) - not reopened as a question here.

**A real risk, not to assume away**: `snapshot()`/`restore()` on the base `ArrayLayer`/
`VectorizedMultiClassBackpropClassifierNetwork` only save/restore `W`/`b` - `m`/`v`/`t` are
Adam-specific per-parameter state that a resumed-training scenario would need too, unlike the
one-shot train-then-evaluate use every measurement in this codebase has used so far (where
mid-training optimizer state is simply discarded, the same posture
`ensemble_train.py::_collect_ensemble_results`'s own docstring already notes is harmless for
`W`/`b`-only snapshots).

**Resolved during implementation**: `AdamVectorizedMultiClassBackpropClassifierNetwork` reuses the
base `snapshot()`/`restore()`/`save()`/`load()` contract unchanged (`W`/`b` only, `m`/`v`/`t`
discarded across a round trip) - not a new gap this class introduces, but the same posture
`AdamBackpropClassifierNetwork`'s own per-node counterpart already has:
`BackpropNetworkBase.snapshot`/`restore` likewise only ever captured weights/bias, never a
momentum/Adam node's own velocity/`m`/`v`/`t`, and no measurement in this codebase has yet used a
resumed-training scenario that would need otherwise.

## correctness validation

Same two-tier convention as every array-based sibling, *plus* a parity check against the existing
per-node reference (the established pattern `test_array_layer.py`'s own
`..._matches_backprop_node_across_a_random_sweep`-named tests already use for plain `ArrayLayer`):

- `test_adam_array_layer.py`: `AdamArrayLayer.apply_accumulated_gradient` checked directly against
  `AdamBackpropNode.apply_accumulated_gradient` (via `make_adam_layer_cls`) - same starting
  weights/gradients/hyperparameters, both paths run for several steps (10 single-example steps,
  and 5 batches of 6), `pytest.approx`/`np.allclose`-compared after each one (not just the end
  state) - the same "matches after every step, not just at the end" discipline
  `test_vectorized_multiclass_backprop_model.py`'s own `learn`/`learn_batch` tests use.
- `test_adam_vectorized_multiclass_backprop_model.py`: whole-network parity against a genuine
  per-node reference, not a hand-derived fixture - the single-output vs. multi-class mismatch
  flagged below turned out not to block this: `MultiClassBackpropClassifierNetwork` already
  extends `BackpropNetworkBase`, whose `hidden_layer_cls`/`output_layer_cls` extension points are
  exactly what `AdamBackpropClassifierNetwork` itself hooks for the single-output case, so a
  test-only `tests/helpers.py::AdamMultiClassBackpropClassifierNetwork` subclass (same
  construction, multi-class sized) gives a real per-node Adam multi-class reference to compare
  against, via `tests/helpers.py::matching_adam_array_backprop_networks` - the Adam-sibling
  analogue of `matching_array_backprop_networks`'s own weight-injection pattern. Hoisted into
  `tests/helpers.py` (rather than kept local to this one test file, as originally written in stage
  2) once stage 5's `test_adam_rust_array_multiclass_backprop_model.py` needed the identical
  reference and helper.
- `test_adam_rust_array_layer.py`/`test_adam_rust_array_multiclass_backprop_model.py` (stage 5):
  the same two-tier treatment, against `AdamRustArrayLayer`/
  `AdamRustArrayMultiClassBackpropClassifierNetwork` instead of the numpy siblings - see "the
  Rust-matmul-backed counterpart" below.
- `rust/indrajala_ml_array/tests/test_adam_fused_layer_ops.py` (stage 5): the Rust crate's own
  fused `layer_adam_apply_accumulated_gradient` checked directly against `AdamArrayLayer` (the
  numpy production reference), the same treatment `test_fused_layer_ops.py` already gives every
  non-Adam fused op - at `t=1` and across several accumulated steps so `m`/`v`/`t` actually
  exercise their accumulation logic, plus the usual shape/argument-validation tests.

## the Rust-matmul-backed counterpart (stage 5)

`AdamRustArrayLayer(RustArrayLayer)` mirrors `AdamArrayLayer(ArrayLayer)`'s relationship to its
own base class exactly, one level over: same `m`/`v`/`t` state (held as `indrajala_ml_array.Array`,
not numpy), same override-only-`apply_accumulated_gradient` shape, but the update itself is one
fused Rust call (`layer_adam_apply_accumulated_gradient`, `fused.rs`) instead of a numpy
expression - the same "fuse the whole method into one Rust call" treatment
[the Rust production cutover plan](rust-production-cutover.md#0b-fuse-each-layer-operation-into-one-rust-call)
already gave every plain-SGD `ArrayLayer` method. `t` is passed into the fused call already
incremented (`self._t += 1` happens in Python first, mirroring `AdamArrayLayer`'s own order),
since the bias-correction terms need the post-increment value and the Rust function has no reason
to own that counter itself. `AdamRustArrayMultiClassBackpropClassifierNetwork` mirrors
`AdamVectorizedMultiClassBackpropClassifierNetwork`'s external contract exactly, against
`AdamRustArrayLayer` instead - same relationship `RustArrayMultiClassBackpropClassifierNetwork`
has to `VectorizedMultiClassBackpropClassifierNetwork`, applied one level further. No new design
questions here - the design was already fully determined by mirroring two existing patterns
(`AdamArrayLayer` and `RustArrayLayer`) at once, so this stage skipped a separate design-only PR
and went straight to implementation + tests.

## measurement plan and result (stage 3)

No new algorithmic question here - this stage is an architectural port (same algorithm proven
correct in [Adam optimizer](adam-optimizer.md), different execution substrate), not a new
hypothesis to test. Two things this codebase's own rigor standard requires checking rather than
assuming, though: does the vectorized port actually turn into a wall-clock win (not just a
theoretical one), and does the per-node reference's batch-size-robustness result (fixed
`learning_rate` barely degrades Adam across `batch_size`, where sigmoid collapses -
[research and analysis](research-adam-optimizer.md#adam-under-batch-size-a-much-bigger-cleaner-win-stage-4-of-the-adam-optimizer-workplan))
still hold once the algorithm runs through a genuinely different code path (batched matmul,
different floating-point summation order).

**Part A - wall-clock, one mini-batch training step** (`forward_batch` + `compute_output_delta_batch`
+ `accumulate_gradient_batch` + `apply_accumulated_gradient`, `dimension=784, hidden=10` - the same
shape/methodology [the Rust production cutover plan](rust-production-cutover.md#0b-fuse-each-layer-operation-into-one-rust-call)'s
own go/no-go benchmark used for the plain (non-Adam) fused ops), three backends (per-node via
`make_adam_layer_cls`, numpy `AdamArrayLayer`, Rust `AdamRustArrayLayer`), median of 20-50 timed
reps per config:

| batch size | per-node (us/example) | numpy (us/example) | Rust (us/example) | per-node/numpy | per-node/Rust | Rust/numpy |
|---|---|---|---|---|---|---|
| 1 | 8542.49 | 121.91 | 84.09 | 70.1x | 101.6x | 0.69x - **Rust faster** |
| 8 | 3693.14 | 16.62 | 16.61 | 222.2x | 222.3x | 1.00x - parity |
| 32 | 2800.73 | 5.63 | 9.99 | 497.5x | 280.4x | 1.78x slower |
| 128 | 2713.91 | 3.94 | 8.08 | 688.8x | 335.9x | 2.05x slower |
| 512 | 2715.24 | 3.42 | 12.05 | 793.9x | 225.3x | 3.52x slower |

**Yes, decisively: both array-based backends are 70x-794x faster per example than the per-node
path across every batch size tested**, and the win *grows* with batch size (the per-node path's
per-example cost stays flat at ~2700-8500us regardless of batch - no vectorization to amortize -
while numpy's/Rust's per-example cost keeps shrinking as batch grows). This directly confirms
[Adam's stage 5 result](research-adam-optimizer.md#adam-at-real-mnist-ensemble-scale-the-proxy-result-holds-stage-5-of-the-adam-optimizer-workplan)'s
own deferred question: mini-batching's *accuracy* robustness now comes with a *wall-clock* win too,
once matmul is real, not just "the same number of per-example Python calls happening in a different
order."

The Rust/numpy split (parity at `batch_size<=8`, numpy pulling ahead by a widening margin from
`batch_size=32` up) looks at first like a new Adam-specific regression, but a same-run plain-SGD
control (`ArrayLayer`/`RustArrayLayer`, identical shape/methodology) rules that out: the control's
own Rust/numpy ratio (0.45x at `batch_size=1`, 4.10x-4.72x by `batch_size=128-512`) is *at least as
wide*, often wider. This is [the already-documented, already-accepted naive-matmul-vs-BLAS
gap](rust-production-cutover.md#the-matmul-gap-that-remains) reappearing at this specific
shape/measurement run, not something `layer_adam_apply_accumulated_gradient` introduced - and per
[the 2026-09-16 clarification](rust-production-cutover.md#phase-0-fix-the-two-measured-bottlenecks-in-the-rust-core-itself-required-gate),
adoption of the Rust backend was already unconditional on this gap, so it isn't a reason to
reconsider `AdamRustArrayLayer` either.

**Part B - accuracy-robustness across `batch_size`, array/Rust-backed network**
(`AdamRustArrayMultiClassBackpropClassifierNetwork`, `[16]` hidden, fixed `learning_rate=0.01`, 10
seeds, 5 epochs, same 320-example real-MNIST digit-3 proxy construction/scale
[stage 4 of the Adam optimizer workplan](research-adam-optimizer.md#adam-under-batch-size-a-much-bigger-cleaner-win-stage-4-of-the-adam-optimizer-workplan)
used for the per-node result):

| batch_size | accuracy |
|---|---|
| 1 | 86.62% ± 1.87% |
| 8 | 85.88% ± 2.05% |
| 32 | 88.50% ± 1.29% |
| 128 | 87.38% ± 0.40% |

**The robustness result holds through the array/Rust port**: no collapse, no meaningful trend
across two orders of magnitude of `batch_size` (86.62%-88.50%, well within seed-to-seed noise of
each other), stdev staying low throughout (0.40%-2.05%) rather than exploding the way sigmoid's did
in the per-node sweep (±3.87% to ±13.33%). Not bit-comparable to the per-node result
(90.50%-86.75%) - different network entirely (`[16]`-hidden multi-class array network with
fan-in-aware init and its own RNG stream vs. a single-output per-node network), the same
RNG-mismatch caveat [the Rust core](rust-array-core.md#the-rng-exception) already documents - but
qualitatively the same flat-across-batch-size shape, confirming the algorithmic property survived
the architectural port rather than being an artifact of the specific per-node implementation.

**Measured, not assumed - script not committed** (ephemeral, following this codebase's own
practice for one-off sweep scripts - see [research and
analysis](research-adam-optimizer.md#adam-under-batch-size-a-much-bigger-cleaner-win-stage-4-of-the-adam-optimizer-workplan)'s
own "hand-rolled script" precedent): `part_a`/`part_b` in a throwaway benchmark script, run against
this repo's `.venv` with the real MNIST data already fetched locally.

## risks and open questions

- **snapshot/restore scope for `m`/`v`/`t`** - resolved; see "design" above.
- **Whether to also build the single-output (`BackpropClassifierNetwork`-equivalent) array-based
  line**, since `AdamBackpropClassifierNetwork` itself is single-output but
  `VectorizedMultiClassBackpropClassifierNetwork`/`RustArrayMultiClassBackpropClassifierNetwork`
  are both multi-class-only - flagged, not resolved; may narrow the parity-check design above
  depending on which way this goes.
- **The Rust-backed follow-on** (`AdamRustArrayLayer`) - resolved; see "the Rust-matmul-backed
  counterpart" above (stage 5, done).

## delivery stages (each its own PR, per this repo's practice)

1. ✅ This design document.
2. ✅ `AdamArrayLayer` (`indrajala_ml/model/adam_array_layer.py`) +
   `AdamVectorizedMultiClassBackpropClassifierNetwork`
   (`indrajala_ml/model/adam_vectorized_multiclass_backprop_classifier_network.py`) + the
   parity-check tests above (`test_adam_array_layer.py`, `test_adam_vectorized_multiclass_backprop_model.py`)
   - the actual capability, buildable and mergeable independent of any later measurement. Full
   suite passes (332 tests in `tests/`, up 14 from before this stage; the separate
   `rust/indrajala_ml_array/tests` and `test_rust_array_multiclass_backprop_model.py` are
   unaffected - out of this stage's numpy-only scope per "scope" above).
3. ✅ The wall-clock/robustness follow-on measurement - see "measurement plan and result" above:
   both array-based backends 70x-794x faster per example than the per-node path, widening with
   batch size; batch-size accuracy-robustness confirmed to survive the array/Rust port.
4. ✅ Docs closeout: `structure.md`'s possible-next-steps entry updated to reflect the actual
   result (see below).
5. ✅ `AdamRustArrayLayer` (`indrajala_ml/model/adam_rust_array_layer.py`) +
   `AdamRustArrayMultiClassBackpropClassifierNetwork`
   (`indrajala_ml/model/adam_rust_array_multiclass_backprop_classifier_network.py`) + the fused
   Rust op (`layer_adam_apply_accumulated_gradient`, `fused.rs`) + parity-check tests at every
   tier (`rust/indrajala_ml_array/tests/test_adam_fused_layer_ops.py`,
   `test_adam_rust_array_layer.py`, `test_adam_rust_array_multiclass_backprop_model.py`) - the
   Rust-backed counterpart flagged as out of scope in stage 2, now built; see "the Rust-matmul-
   backed counterpart" above. Independent of stages 3/4 (a separate execution substrate for the
   same already-proven-correct algorithm, not a further measurement of it) - built without waiting
   on either. Full suite passes (355 tests in `tests/` - 14 of them this stage's own new
   `test_adam_rust_array_layer.py`/`test_adam_rust_array_multiclass_backprop_model.py`, the rest of
   the growth since stage 2's 332 from unrelated work done in between; `tests/helpers.py`'s shared
   `AdamMultiClassBackpropClassifierNetwork`/`matching_adam_array_backprop_networks` also replaced
   stage 2's test-file-local duplicate of the same reference). 618 tests pass in
   `rust/indrajala_ml_array/tests` overall (up from 525 - see [the Rust production cutover
   plan](rust-production-cutover.md) for that count's own history), including this stage's own new
   `test_adam_fused_layer_ops.py`.
