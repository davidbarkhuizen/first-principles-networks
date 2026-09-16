# an array-based (Rust-matmul-backed) Adam sibling

[← back to README](../README.md)

**Status: stage 2 (the numpy capability) and stage 5 (the Rust-matmul-backed counterpart) done;
stage 3 (the wall-clock/robustness follow-on measurement) and stage 4 (docs closeout) not yet
run.** Written up front as a design/measurement plan before any of it existed, per this repo's own
practice (see [Adam optimizer](adam-optimizer.md), [the Rust production cutover
plan](rust-production-cutover.md) for precedent) - updated here with stage-by-stage status notes
as it's executed.

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

## measurement plan

No new algorithmic question here - this stage is an architectural port (same algorithm proven
correct in [Adam optimizer](adam-optimizer.md), different execution substrate), not a new
hypothesis to test. Once correctness is established, the natural follow-on measurement - does
Adam's batch-size robustness translate into an actual wall-clock win once matmul is real, the
question stage 5 explicitly deferred to this capability - is its own next stage, not assumed or
folded into this one.

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
3. The wall-clock/robustness follow-on measurement flagged above (conditional on stage 2 landing
   cleanly and a concrete scenario worth spending a real training run on).
4. Docs closeout: `structure.md`'s possible-next-steps entry updated to reflect the actual result.
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
