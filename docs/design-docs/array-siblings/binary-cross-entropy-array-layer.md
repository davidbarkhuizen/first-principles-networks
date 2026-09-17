# an array-based (Rust-matmul-backed) binary cross-entropy sibling

[← back to README](../../../README.md)

Design reference for `CrossEntropyArrayLayer`/`CrossEntropyArrayBackpropClassifierNetwork`/
`CrossEntropyVectorizedMultiClassBackpropClassifierNetwork` (numpy) and
`CrossEntropyRustArrayLayer`/`CrossEntropyRustArrayBackpropClassifierNetwork`/
`CrossEntropyRustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed) - array siblings
of `BinaryCrossEntropyBackpropClassifierNetwork` (see
[structure](../../project/structure.md#possible-next-steps)). Built and measured; see [research
and
analysis](../../research/research-multiclass-and-loss.md#an-array-based-binary-cross-entropy-sibling-the-retune-the-per-node-investigation-deferred)
for the real-MNIST retune measurement and decision.

## design: `output_layer_cls`-shaped at the array level - only `compute_output_delta` changes

`CrossEntropyArrayLayer(ArrayLayer)` overrides only `compute_output_delta`/
`compute_output_delta_batch` - binary cross-entropy's delta simplifies to `activation - target`,
with no `a*(1-a)` damping term, the same simplification `SoftmaxArrayLayer`'s own delta uses for
the multi-class case:

```python
class CrossEntropyArrayLayer(ArrayLayer):
    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = self.a - reference

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = self.A - reference_batch
```

`forward`/`forward_batch`/`compute_hidden_delta`/`compute_hidden_delta_batch`/
`apply_accumulated_gradient` are all inherited unchanged from `ArrayLayer` - unlike
`SoftmaxArrayLayer`, a single sigmoid-activated output needs nothing from any sibling node, the
same point `CrossEntropyOutputNode`'s own docstring makes.

Two hosts built on top of this layer - **2026-09-17 update:** both are now one-line
`output_layer_cls` overrides of the relevant "shape" base class
(`ArrayBackpropClassifierNetwork`/`RustArrayBackpropClassifierNetwork` for the single-output pair,
`VectorizedMultiClassBackpropClassifierNetwork`/`RustArrayMultiClassBackpropClassifierNetwork` for
the multiclass pair) rather than the "wholly separate class, not a subclass swapping a
`layer_cls` extension point" duplication this section originally described - see
`docs/design-docs/adam/adam-array-layer.md`'s own 2026-09-17 update for the full rationale behind that
change:

- **`CrossEntropyArrayBackpropClassifierNetwork`/`CrossEntropyRustArrayBackpropClassifierNetwork`**
  - the literal single-output array counterpart of `BinaryCrossEntropyBackpropClassifierNetwork`,
    structurally identical to `ArrayBackpropClassifierNetwork`/`RustArrayBackpropClassifierNetwork`
    with a `CrossEntropyArrayLayer`/`CrossEntropyRustArrayLayer` output instead of a plain one.
    Named without the `Vectorized` prefix, matching that pair's own convention - the prefix is
    reserved for the `class_count`-wide multiclass line.
- **`CrossEntropyVectorizedMultiClassBackpropClassifierNetwork`/`CrossEntropyRustArrayMultiClassBackpropClassifierNetwork`**
  - `CrossEntropyArrayLayer` also works unchanged as a `class_count`-sized output on the existing
    multiclass array line (an independent per-node cross-entropy delta at each output - a
    one-vs-rest-with-cross-entropy-loss variant, distinct from softmax's jointly-normalized one),
    mirroring `SoftmaxVectorizedMultiClassBackpropClassifierNetwork`'s own `output_layer_cls`
    override even though it isn't the literal counterpart of the single-output class above.

**The Rust stage needed no new Rust primitive** - checked directly against the Rust source, not
assumed: `SoftmaxArrayLayer.compute_output_delta`'s own formula (`self.a - reference`) is
algebraically identical to what cross-entropy needs, and its existing Rust-fused counterpart,
`pa.layer_softmax_output_delta` (`fused.rs`), is already shape-agnostic (`require_same_shape` plus
an elementwise subtract, no softmax-specific math) - `CrossEntropyRustArrayLayer` calls it
directly.

## correctness validation

Fully deterministic given the same starting weights (no RNG in cross-entropy's own delta
formula):

- `test_cross_entropy_array_layer.py`/`test_cross_entropy_rust_array_layer.py`:
  `compute_output_delta`/`compute_output_delta_batch` checked directly against
  `CrossEntropyOutputNode.compute_output_delta` across a random sweep of activations/targets.
- `test_cross_entropy_array_backprop_model.py`/`test_cross_entropy_rust_array_backprop_model.py`:
  whole-network parity against `BinaryCrossEntropyBackpropClassifierNetwork` itself directly (a
  genuine per-node reference already exists here) via `tests/helpers.py`'s
  `matching_cross_entropy_array_backprop_networks` helper.
- `test_cross_entropy_vectorized_multiclass_backprop_model.py`/
  `test_cross_entropy_rust_array_multiclass_backprop_model.py`: the multiclass-line variant,
  parity-checked against a test-only per-node reference
  (`CrossEntropyMultiClassBackpropClassifierNetwork`, `tests/helpers.py`), the same two-tier
  convention every array-ported sibling in this codebase uses.
