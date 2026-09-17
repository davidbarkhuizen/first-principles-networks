# an array-based (Rust-matmul-backed) softmax sibling

[← back to README](../../../README.md)

Design reference for `SoftmaxArrayLayer`/`SoftmaxVectorizedMultiClassBackpropClassifierNetwork`
(numpy) and `SoftmaxRustArrayLayer`/`SoftmaxRustArrayMultiClassBackpropClassifierNetwork`
(Rust-matmul-backed) - array siblings of `SoftmaxMultiClassBackpropClassifierNetwork` (see
[structure](../../project/structure.md#multi-class)). Built and measured; see [research and
analysis](../../research/research-multiclass-and-loss.md#an-array-based-softmax-sibling-the-retune-the-per-node-investigation-declined-to-spend-on)
for the wall-clock/reproduction/retune measurement and decision.

## design: `output_layer_cls`-shaped at the array level - `forward` and `compute_output_delta` change

`SoftmaxArrayLayer(ArrayLayer)` overrides `forward`/`forward_batch` (joint normalization across
the whole output vector, not an independent per-row sigmoid) and `compute_output_delta`/
`compute_output_delta_batch` (the same `activation - target` simplification `SoftmaxOutputNode`
already uses, no `a*(1-a)` factor):

```python
class SoftmaxArrayLayer(ArrayLayer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        shifted = self.z - np.max(self.z)          # same numerically-stable shift
        exp_values = np.exp(shifted)                # SoftmaxOutputLayer.forward already uses
        self.a = exp_values / exp_values.sum()
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        self.Z = X @ self.W.T + self.b
        shifted = self.Z - self.Z.max(axis=1, keepdims=True)   # row-wise max, one row per example
        exp_values = np.exp(shifted)
        self.A = exp_values / exp_values.sum(axis=1, keepdims=True)
        return self.A

    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = self.a - reference             # SoftmaxOutputNode's own simplification

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = self.A - reference_batch
```

Softmax's cross-node coupling is real but shallow here too, exactly as the per-node version found
("softmax's cross-node coupling ... only affects the forward pass" -
[structure](../../project/structure.md#multi-class)): `compute_hidden_delta` for whatever layer feeds this one
is untouched (it only ever reads `next_layer.W`/`next_layer.delta`, never `next_layer.a`
directly), so no other class in `VectorizedMultiClassBackpropClassifierNetwork`'s chain needs any
change - the same "single class-attribute override, zero backward-pass-plumbing changes" shape the
per-node version already achieved.

The row-wise batched max/sum (`axis=1, keepdims=True`) is a genuinely new operation against [the
numpy interface subset](../../architecture/numpy-interface-subset.md#explicitly-not-required)'s documented scope
("general axis-parameterized reductions ... belong to variants ... this subset ... defers");
`SoftmaxRustArrayLayer` needed the Rust core's matmul-and-reduction primitives extended with a
row-wise max/sum-and-normalize op before it could be built - the same category of real, small
scope extension the array-based ReLU sibling needed for its own primitive.

## correctness validation

Fully deterministic given the same starting weights (no RNG in softmax's own math):

- `test_softmax_array_layer.py`: `forward`/`forward_batch`/`compute_output_delta`/
  `compute_output_delta_batch` checked directly against `SoftmaxOutputNode`/`SoftmaxOutputLayer`'s
  own per-node formulas across a random sweep of weights/inputs, including a
  numerically-adversarial case (large-magnitude `z` values) to confirm the max-shift trick matches
  the per-node reference's own overflow-safe behavior, not just the well-conditioned case.
- `test_softmax_vectorized_multiclass_backprop_model.py`: whole-network parity against
  `SoftmaxMultiClassBackpropClassifierNetwork` itself directly (a genuine per-node reference
  already exists here, unlike momentum/L2/ReLU which needed a test-only helper subclass) - via a
  `matching_array_backprop_networks`-style weight-injection helper.
- `test_softmax_rust_array_layer.py`/`test_softmax_rust_array_multiclass_backprop_model.py` +
  `rust/indrajala_ml_array/tests/test_array_softmax.py`: the Rust core's new row-wise
  softmax-normalization primitive parity-tested against numpy directly, then the same two
  model-level tiers against the Rust-backed classes.
