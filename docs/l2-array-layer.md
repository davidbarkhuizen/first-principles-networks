# an array-based (Rust-matmul-backed) L2 regularization sibling

[← back to README](../README.md)

**Status: proposed, not started.** Written up front as a design/measurement plan before any of
it exists, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based
Adam sibling](adam-array-layer.md) for precedent).

## why this, and why now

`L2RegularizedBackpropClassifierNetwork` (`l2_regularization_layer.py`'s `make_l2_node_cls`/
`make_l2_layer_cls`) is still per-node-only. Its own measurement ([L2 weight
regularization](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it))
found it closes the train-test accuracy gap but never improves held-out accuracy above the
unregularized baseline, on a 400-example real-MNIST digit-3 proxy - plausibly because a single
16-node hidden layer on 320 training examples doesn't overfit severely enough for a
weight-magnitude penalty to have room to help. [Dropout](dropout.md)'s own workplan flagged the
identical "not enough overfitting headroom" risk and proposed a conditional, more-overfitting-prone
follow-up (larger hidden layer, more epochs) that was never run - not because it wasn't worth
running, but because the per-node path made even the *original*, smaller sweep too slow to afford
twice (this session's own dropout sweep, on the same setup, was abandoned at an estimated 40+
minutes).

Per the updated mandate behind this round of workplans: vectorizing L2 is the cheap way to
actually run that escalation, not an expensive follow-on gated on the null result being
overturned first. This workplan proposes building the array/Rust counterpart specifically so the
overfitting-headroom question - for L2 and, once [dropout's own array port](dropout-array-layer.md)
exists, for both regularizers side by side on equal footing - can finally be tested at a scale
that could actually show a difference.

## scope

`L2ArrayLayer` (numpy) and `L2RustArrayLayer` (Rust-matmul-backed), the same two-stage
numpy-then-Rust precedent [an array-based Adam sibling](adam-array-layer.md) established. Scoped
to `MultiClassBackpropClassifierNetwork`'s array line
(`L2VectorizedMultiClassBackpropClassifierNetwork`/
`L2RustArrayMultiClassBackpropClassifierNetwork`) for the same still-unresolved reason [an
array-based Adam sibling](adam-array-layer.md#risks-and-open-questions) and [the momentum array
workplan](momentum-array-layer.md) both give: no single-output array-based line exists to extend
instead. `L2RegularizedBackpropClassifierNetwork` sets both `hidden_layer_cls` and
`output_layer_cls`; this workplan mirrors that (both `layers` and `output_layer` built from
`L2ArrayLayer`).

## design: the simplest port in this whole round - no new per-layer state at all

`L2ArrayLayer(ArrayLayer)` overrides only `apply_accumulated_gradient`, and unlike momentum or
Adam, needs no persistent per-parameter state between steps at all - `l2_lambda` is a fixed
scalar, and the penalty term is a pure function of the *current* weight, not any running history:

```python
class L2ArrayLayer(ArrayLayer):
    def __init__(self, size: int, input_size: int, l2_lambda: float) -> None:
        super().__init__(size, input_size)
        self._l2_lambda = l2_lambda

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W -= learning_rate * (self._grad_W / batch_size + self._l2_lambda * self.W)
        self.b -= learning_rate * self._grad_b / batch_size  # bias unregularized, matching
        self._reset_gradient_accum()                          # make_l2_node_cls's own comment
```

Directly mirrors `make_l2_node_cls`'s per-node formula (`w -= lr*(accum/batch_size +
l2_lambda*w)`, bias untouched) - one array op per parameter tensor instead of a per-weight Python
loop, and no snapshot/restore question at all (no new state beyond `W`/`b`, which the base class
already captures) - a strictly simpler port than momentum's or Adam's own.

`L2RustArrayLayer(RustArrayLayer)` mirrors this one level over, the same fused-Rust-call
treatment [an array-based Adam sibling](adam-array-layer.md#the-rust-matmul-backed-counterpart-stage-5)
gave `AdamRustArrayLayer` (`layer_l2_apply_accumulated_gradient`, `fused.rs`).

## correctness validation

Same two-tier convention, plus the parity-against-per-node-reference tier:

- `test_l2_array_layer.py`: `L2ArrayLayer.apply_accumulated_gradient` checked directly against
  `L2RegularizedBackpropNode.apply_accumulated_gradient` (via `make_l2_layer_cls`) - same starting
  weights/gradients/`l2_lambda`, several steps, `np.allclose`-compared after each. Fully
  deterministic on both sides, a genuine bit-close parity check.
- `test_l2_vectorized_multiclass_backprop_model.py`: whole-network parity against a test-only
  `L2MultiClassBackpropClassifierNetwork` reference (`hidden_layer_cls = output_layer_cls =
  make_l2_layer_cls(l2_lambda)`), the same `tests/helpers.py`-hoisted weight-injection pattern
  used throughout this round.
- `test_l2_rust_array_layer.py`/`test_l2_rust_array_multiclass_backprop_model.py` +
  `rust/indrajala_ml_array/tests/test_l2_fused_layer_ops.py`: the same two tiers against the
  Rust-backed classes.

## measurement plan

- **Wall-clock**: the same fused-layer-operation benchmark used throughout this round
  (`dimension=784, hidden=10`, per-node/numpy/Rust, several batch sizes) - expect a result in the
  same 70x-800x-class range as Adam's/momentum's own, since L2's extra term is one elementwise
  multiply-subtract on top of the same matmul-dominated cost.
- **The overfitting-headroom escalation, now affordable**: re-run [L2's own original
  setup](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it)
  (400-example real-MNIST digit-3 proxy, 80/20 split, fan-in-aware init, `[16]` hidden,
  `learning_rate=0.5`) at many more seeds than the per-node path afforded, *and* the
  more-overfitting-prone escalation dropout's own workplan proposed but never ran - a
  substantially larger hidden layer relative to the fixed 320 training examples, and/or
  substantially more epochs, deliberately chosen to induce visible overfitting first (verified by
  checking the train-test gap actually widens before re-testing whether any `l2_lambda` closes it
  further than the unregularized baseline already does). Report honestly whichever way it lands.

## risks and open questions

- **A reconfirmed null is a legitimate outcome** - same framing as [the momentum array
  workplan](momentum-array-layer.md#risks-and-open-questions): this workplan's value is a cheap,
  better-powered re-check, not a guaranteed new capability.
- **The single-output array-based line gap** - inherited, unresolved, same as every sibling in
  this round.
- **Head-to-head comparability with dropout** - a fair L2-vs-dropout comparison at the escalated,
  overfitting-prone setting needs [dropout's own array port](dropout-array-layer.md) to exist
  too, on the identical setup; this workplan doesn't block on that, but the most valuable version
  of its own measurement (a genuine two-regularizer comparison, not two isolated single-regularizer
  checks) does.

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. `L2ArrayLayer` + `L2VectorizedMultiClassBackpropClassifierNetwork` + the parity-check tests
   above (numpy only).
3. The wall-clock/overfitting-headroom measurement described above.
4. Docs closeout: `structure.md`'s possible-next-steps entry updated to reflect the actual result.
5. `L2RustArrayLayer` + `L2RustArrayMultiClassBackpropClassifierNetwork` + the fused Rust op +
   parity-check tests at every tier - independent of stage 3's measurement.
