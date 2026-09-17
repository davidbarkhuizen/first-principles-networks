# an array-based (Rust-matmul-backed) momentum sibling

[← back to README](../README.md)

Design reference for `MomentumArrayLayer`/`MomentumVectorizedMultiClassBackpropClassifierNetwork`
(numpy) and `MomentumRustArrayLayer`/`MomentumRustArrayMultiClassBackpropClassifierNetwork`
(Rust-matmul-backed) - array siblings of `MomentumBackpropClassifierNetwork` (see
[structure](structure.md#backprop-siblings)). Built and measured; see [research and
analysis](research-backprop-siblings.md#an-array-based-momentum-sibling-unchanged-on-stronger-evidence)
for the wall-clock/accuracy measurement and decision.

## design: an override-only `apply_accumulated_gradient`, mirroring Adam's own precedent exactly

`MomentumArrayLayer(ArrayLayer)` overrides only `apply_accumulated_gradient`, the same shape
`AdamArrayLayer` used, but simpler - one previous-delta array per parameter tensor, no bias
correction, no second moment:

```python
class MomentumArrayLayer(ArrayLayer):
    def __init__(self, size: int, input_size: int, momentum: float) -> None:
        super().__init__(size, input_size)
        self._momentum = momentum
        self._prev_delta_W = np.zeros((size, input_size))
        self._prev_delta_b = np.zeros(size)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        delta_W = learning_rate * self._grad_W / batch_size + self._momentum * self._prev_delta_W
        delta_b = learning_rate * self._grad_b / batch_size + self._momentum * self._prev_delta_b
        self.W -= delta_W
        self.b -= delta_b
        self._prev_delta_W, self._prev_delta_b = delta_W, delta_b
        self._reset_gradient_accum()
```

Directly mirrors `make_momentum_node_cls`'s per-node formula (`Δw(n) = η·δ·a + α·Δw(n-1)`), one
array op per parameter tensor instead of a per-weight Python loop. `momentum` is a required
constructor argument, no default - the same posture `make_momentum_node_cls` itself takes (this
codebase's own measurements never found a value worth recommending).

Like Adam's own snapshot/restore note: `_prev_delta_W`/`_prev_delta_b` are momentum-specific
per-parameter state a resumed-training scenario would need, but every measurement in this
codebase uses one-shot train-then-evaluate, so `snapshot()`/`restore()` stay `W`/`b`-only,
unchanged from the base class - the same posture the array-based Adam sibling already resolved
this way, not a new gap.

`MomentumRustArrayLayer(RustArrayLayer)` mirrors `AdamRustArrayLayer`'s own relationship to
`AdamArrayLayer` one level over - the same update expressed as one fused Rust call
(`layer_momentum_apply_accumulated_gradient`, `fused.rs`) instead of a numpy expression, per [the
Rust production cutover plan](rust-production-cutover.md#0b-fuse-each-layer-operation-into-one-rust-call)'s
"fuse the whole method into one Rust call" treatment.

## correctness validation

Same two-tier convention as every array-based sibling, plus the parity-against-per-node-reference
tier [an array-based Adam sibling](adam-array-layer.md#correctness-validation) established:

- `test_momentum_array_layer.py`: `MomentumArrayLayer.apply_accumulated_gradient` checked
  directly against `MomentumBackpropNode.apply_accumulated_gradient` (via
  `make_momentum_layer_cls`) - same starting weights/gradients/`momentum` coefficient, several
  steps (single-example and batched), `np.allclose`-compared after every step. Fully
  deterministic on both sides (no RNG involved in the update rule itself), so this is a genuine
  bit-close parity check, not a qualitative-shape one the way dropout's own port needed.
- `test_momentum_vectorized_multiclass_backprop_model.py`: whole-network parity against a
  test-only `MomentumMultiClassBackpropClassifierNetwork` reference (`hidden_layer_cls =
  output_layer_cls = make_momentum_layer_cls(momentum)`), the same `tests/helpers.py`-hoisted
  weight-injection pattern `matching_adam_array_backprop_networks` established.
- `test_momentum_rust_array_layer.py`/`test_momentum_rust_array_multiclass_backprop_model.py` +
  `rust/indrajala_ml_array/tests/test_momentum_fused_layer_ops.py`: the same two tiers against
  the Rust-backed classes, mirroring [an array-based Adam
  sibling](adam-array-layer.md#the-rust-matmul-backed-counterpart)'s own stage 5.
