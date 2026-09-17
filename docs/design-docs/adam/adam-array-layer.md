# an array-based (Rust-matmul-backed) Adam sibling

[← back to README](../../../README.md)

Design reference for `AdamArrayLayer`/`AdamVectorizedMultiClassBackpropClassifierNetwork` (numpy)
and `AdamRustArrayLayer`/`AdamRustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed) -
array siblings of `AdamBackpropClassifierNetwork` (see [structure](../../project/structure.md#backprop-siblings)).
Built and measured; see [research and
analysis](../../research/research-adam-optimizer.md#an-array-based-rust-matmul-backed-adam-sibling-the-wall-clock-win-the-accuracy-result-was-missing)
for the wall-clock/robustness measurement and decision.

## design: `hidden_layer_cls`/`output_layer_cls`-shaped over `ArrayNetworkBase`

**2026-09-17 update:** this section originally justified `AdamVectorizedMultiClassBackpropClassifierNetwork`
as "a new, wholly separate class, not a subclass swapping a `layer_cls` extension point" - because
at the time, `VectorizedMultiClassBackpropClassifierNetwork.__init__` hardcoded `ArrayLayer`
construction with no such extension point, and the first array sibling
(`RustArrayMultiClassBackpropClassifierNetwork`, swapping backends rather than a hyperparameter)
had already been built that way, so every later sibling followed the same precedent rather than
retrofitting one. A follow-up DRY audit found that precedent-following, not a reasoned decision -
the per-node family already had exactly this extension point
(`BackpropNetworkBase.hidden_layer_cls`/`output_layer_cls`) - and added the array-level
equivalent: `ArrayNetworkBase`/`RustArrayNetworkBase` now expose `hidden_layer_cls`/
`output_layer_cls`, and `VectorizedMultiClassBackpropClassifierNetwork`/
`RustArrayMultiClassBackpropClassifierNetwork` are thin "shape" subclasses of them (multiclass
argmax/one-hot-target/class_count handling), the same way `MultiClassBackpropClassifierNetwork`
sits over `BackpropNetworkBase`.

`AdamVectorizedMultiClassBackpropClassifierNetwork`/`AdamRustArrayMultiClassBackpropClassifierNetwork`
are now thin subclasses of those shape classes: `__init__` sets `self.hidden_layer_cls =
self.output_layer_cls` to a closure binding `beta1`/`beta2`/`epsilon` into `AdamArrayLayer`/
`AdamRustArrayLayer` before calling `super().__init__(...)` - the same
"instance-attribute-closure-before-`super().__init__()`" pattern `AdamBackpropClassifierNetwork`
already used one layer down over `BackpropNetworkBase`. `save`/`load`'s hyperparameter round-trip
goes through `_extra_state`/`_extra_init_kwargs` hooks on the base rather than a hand-written
envelope. See `indrajala_ml/model/adam_vectorized_multiclass_backprop_classifier_network.py`
directly for the current ~55-line result (down from the ~170-line duplicate this section
originally described).

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
`AdamBackpropClassifierNetwork`'s own resolved posture (see [structure](../../project/structure.md#backprop-siblings)).

`snapshot()`/`restore()` on the base `ArrayLayer`/`VectorizedMultiClassBackpropClassifierNetwork`
only save/restore `W`/`b` - `m`/`v`/`t` are Adam-specific per-parameter state that a
resumed-training scenario would need too, unlike the one-shot train-then-evaluate use every
measurement in this codebase has used so far (where mid-training optimizer state is simply
discarded, the same posture `ensemble_train.py::_collect_ensemble_results`'s own docstring already
notes is harmless for `W`/`b`-only snapshots).
`AdamVectorizedMultiClassBackpropClassifierNetwork` reuses the base
`snapshot()`/`restore()`/`save()`/`load()` contract unchanged (`W`/`b` only, `m`/`v`/`t` discarded
across a round trip) - not a new gap this class introduces, but the same posture
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
  per-node reference, not a hand-derived fixture: `MultiClassBackpropClassifierNetwork` already
  extends `BackpropNetworkBase`, whose `hidden_layer_cls`/`output_layer_cls` extension points are
  exactly what `AdamBackpropClassifierNetwork` itself hooks for the single-output case, so a
  test-only `tests/helpers.py::AdamMultiClassBackpropClassifierNetwork` subclass (same
  construction, multi-class sized) gives a real per-node Adam multi-class reference to compare
  against, via `tests/helpers.py::matching_adam_array_backprop_networks` - the Adam-sibling
  analogue of `matching_array_backprop_networks`'s own weight-injection pattern.
- `test_adam_rust_array_layer.py`/`test_adam_rust_array_multiclass_backprop_model.py`: the same
  two-tier treatment, against `AdamRustArrayLayer`/`AdamRustArrayMultiClassBackpropClassifierNetwork`
  instead of the numpy siblings - see "the Rust-matmul-backed counterpart" below.
- `rust/indrajala_ml_array/tests/test_adam_fused_layer_ops.py`: the Rust crate's own fused
  `layer_adam_apply_accumulated_gradient` checked directly against `AdamArrayLayer` (the numpy
  production reference), the same treatment `test_fused_layer_ops.py` already gives every
  non-Adam fused op - at `t=1` and across several accumulated steps so `m`/`v`/`t` actually
  exercise their accumulation logic, plus the usual shape/argument-validation tests.

## the Rust-matmul-backed counterpart

`AdamRustArrayLayer(RustArrayLayer)` mirrors `AdamArrayLayer(ArrayLayer)`'s relationship to its
own base class exactly, one level over: same `m`/`v`/`t` state (held as `indrajala_ml_array.Array`,
not numpy), same override-only-`apply_accumulated_gradient` shape, but the update itself is one
fused Rust call (`layer_adam_apply_accumulated_gradient`, `fused.rs`) instead of a numpy
expression - the same "fuse the whole method into one Rust call" treatment [the Rust production
cutover plan](../../architecture/rust-production-cutover.md#0b-fuse-each-layer-operation-into-one-rust-call) already
gave every plain-SGD `ArrayLayer` method. `t` is passed into the fused call already incremented
(`self._t += 1` happens in Python first, mirroring `AdamArrayLayer`'s own order), since the
bias-correction terms need the post-increment value and the Rust function has no reason to own
that counter itself. `AdamRustArrayMultiClassBackpropClassifierNetwork` mirrors
`AdamVectorizedMultiClassBackpropClassifierNetwork`'s external contract exactly, against
`AdamRustArrayLayer` instead - same relationship `RustArrayMultiClassBackpropClassifierNetwork`
has to `VectorizedMultiClassBackpropClassifierNetwork`, applied one level further.
