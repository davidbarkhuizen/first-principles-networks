# an array-based (Rust-matmul-backed) L2 regularization sibling

[← back to README](../../../README.md)

Design reference for `L2ArrayLayer`/`L2VectorizedMultiClassBackpropClassifierNetwork` (numpy) and
`L2RustArrayLayer`/`L2RustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed) - array
siblings of `L2RegularizedBackpropClassifierNetwork` (see
[structure](../../project/structure.md#backprop-siblings)). Built and measured; see [research and
analysis](../../research/research-backprop-siblings.md#an-array-based-l2-sibling-the-reconfirmed-null-holds-at-far-higher-power)
for the wall-clock/overfitting-headroom measurement and decision.

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

`L2RustArrayLayer(RustArrayLayer)` mirrors this one level over, the same fused-Rust-call treatment
the array-based Adam sibling gave `AdamRustArrayLayer` (`layer_l2_apply_accumulated_gradient`,
`fused.rs`).

## correctness validation

Same two-tier convention, plus the parity-against-per-node-reference tier:

- `test_l2_array_layer.py`: `L2ArrayLayer.apply_accumulated_gradient` checked directly against
  `L2RegularizedBackpropNode.apply_accumulated_gradient` (via `make_l2_layer_cls`) - same starting
  weights/gradients/`l2_lambda`, several steps, `np.allclose`-compared after each. Fully
  deterministic on both sides, a genuine bit-close parity check.
- `test_l2_vectorized_multiclass_backprop_model.py`: whole-network parity against a test-only
  `L2MultiClassBackpropClassifierNetwork` reference (`hidden_layer_cls = output_layer_cls =
  make_l2_layer_cls(l2_lambda)`), the same `tests/helpers.py`-hoisted weight-injection pattern
  used throughout this codebase's array-porting history.
- `test_l2_rust_array_layer.py`/`test_l2_rust_array_multiclass_backprop_model.py` +
  `rust/indrajala_ml_array/tests/test_l2_fused_layer_ops.py`: the same two tiers against the
  Rust-backed classes.
