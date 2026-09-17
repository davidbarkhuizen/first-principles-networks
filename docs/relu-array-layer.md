# an array-based (Rust-matmul-backed) ReLU sibling

[← back to README](../README.md)

Design reference for `ReLUArrayLayer`/`ReLUVectorizedMultiClassBackpropClassifierNetwork` (numpy)
and `ReLURustArrayLayer`/`ReLURustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed) -
array siblings of `ReLUBackpropClassifierNetwork` (see [structure](structure.md#backprop-siblings)).
Built and measured; see [research and
analysis](research-backprop-siblings.md#an-array-based-relu-sibling-a-genuine-scale-dependent-finding)
for the wall-clock/accuracy measurement and decision.

## design: forward and `compute_hidden_delta` both change; no persistent state

Unlike momentum/L2/Adam (which only touch `apply_accumulated_gradient`), ReLU changes the
*activation* itself, so `ReLUArrayLayer` overrides `forward`/`forward_batch` and
`compute_hidden_delta`/`compute_hidden_delta_batch` - `apply_accumulated_gradient` is inherited
unchanged from `ArrayLayer`:

```python
class ReLUArrayLayer(ArrayLayer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        self.a = np.maximum(0.0, self.z)          # relu_layer.py's relu_activation, vectorized
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        self.Z = X @ self.W.T + self.b
        self.A = np.maximum(0.0, self.Z)
        return self.A

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.W.T @ next_layer.delta
        self.delta = downstream * (self.a > 0.0)   # relu_hidden_delta's derivative: 1 where
                                                     # z>0 (equivalently a>0), 0 otherwise - no
                                                     # a*(1-a) damping term at all
    def compute_hidden_delta_batch(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.delta_batch @ next_layer.W
        self.delta_batch = downstream * (self.A > 0.0)
```

`compute_output_delta`/`compute_output_delta_batch` are inherited unchanged too, but must never
actually be called on a `ReLUArrayLayer` instance in practice (hidden-only convention) - mirroring
`ReLUNode.compute_output_delta`'s own `raise NotImplementedError` guard, this class should raise
the same way rather than silently computing a meaningless sigmoid-shaped output delta on an
unbounded ReLU activation.

This is the first array port in this codebase's history that touches the array-based forward pass
itself, not just the weight-update rule - `np.where`/`np.maximum` are genuinely new operations
against [the numpy interface subset](numpy-interface-subset.md#explicitly-not-required)'s own
documented scope, which explicitly named `np.maximum`/`np.where` as deferred exactly to "a future
follow-on covering a specific variant (say, a vectorized ReLU sibling)" - this is that follow-on,
and the Rust core's own operation set needed the equivalent primitive added
(`array_relu`/`array_relu_mask`-shaped, mirroring the numpy calls above) before
`ReLURustArrayLayer` could be built - a real, small extension to [the Rust array
core](rust-array-core.md)'s scope, not assumed to already exist there.

## correctness validation

Fully deterministic on both sides (no RNG involved anywhere in ReLU's own math) - a genuine
bit-close parity check, the cleanest of every array-ported sibling in this codebase:

- `test_relu_array_layer.py`: `forward`/`forward_batch`/`compute_hidden_delta`/
  `compute_hidden_delta_batch` checked directly against `ReLUNode`/`ReLULayer`'s own per-node
  formulas across a random sweep of weights/inputs - including the `z == 0` boundary case
  explicitly (both this class's `> 0.0` mask and `relu_hidden_delta`'s own either-branch
  convention agree it's zero-derivative there, measure-zero in practice but worth one explicit
  test rather than leaving it implicit).
- `test_relu_vectorized_multiclass_backprop_model.py`: whole-network parity against a test-only
  `ReLUMultiClassBackpropClassifierNetwork` reference (`hidden_layer_cls = ReLULayer`), the same
  `tests/helpers.py`-hoisted weight-injection pattern used throughout this codebase's
  array-porting history.
- `test_relu_rust_array_layer.py`/`test_relu_rust_array_multiclass_backprop_model.py` +
  `rust/indrajala_ml_array/tests/test_array_relu.py`: the Rust core's new `array_relu` primitive
  parity-tested against numpy directly (the same treatment every existing Rust array operation
  gets - [the Rust array core](rust-array-core.md)), then the same two model-level tiers against
  the Rust-backed classes.
