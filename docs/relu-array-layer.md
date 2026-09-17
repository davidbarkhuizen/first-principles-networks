# an array-based (Rust-matmul-backed) ReLU sibling

[← back to README](../README.md)

**Status: proposed, not started.** Written up front as a design/measurement plan before any of
it exists, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based
Adam sibling](adam-array-layer.md) for precedent).

## why this, and why now

`ReLUBackpropClassifierNetwork` (`relu_layer.py`'s `ReLUNode`/`ReLULayer`) is still per-node-only,
despite being one of only two siblings ([structure](structure.md#backprop-siblings)) with a
*genuine, adopted* measured win: retuned to its own learning rate, ReLU exceeds the sigmoid
baseline (99.27% vs 97.80% mean on a fixed XOR scenario - [ReLU hidden-layer
activation](research-backprop-siblings.md#relu-hidden-layer-activation-a-clean-win-once-retuned)).
That toy-scale result has never been checked at UCI-digits or real-MNIST scale, where sigmoid
saturation is the exact, already-measured pathology fan-in-aware init was built to fix
([FanInAware...](structure.md#backprop-siblings) - 83.5% of hidden activations saturated at
initialization at MNIST's 784-dimension fan-in) - ReLU's non-saturating positive side is a
directly relevant candidate there, and today the only way to check is the 70-800x-slower per-node
path.

## scope

`ReLUArrayLayer` (numpy) and `ReLURustArrayLayer` (Rust-matmul-backed), the same two-stage
precedent [an array-based Adam sibling](adam-array-layer.md) established. Hidden-layer-only,
matching `ReLUNode`'s own convention ([structure](structure.md#backprop-siblings): "the output
layer is untouched, since ReLU is a hidden-layer-only convention") - `output_layer` stays a plain
`ArrayLayer` (sigmoid). Scoped to `MultiClassBackpropClassifierNetwork`'s array line
(`ReLUVectorizedMultiClassBackpropClassifierNetwork`/`ReLURustArrayMultiClassBackpropClassifierNetwork`),
the same still-open single-output-line gap every sibling in this round inherits from [an
array-based Adam sibling](adam-array-layer.md#risks-and-open-questions).

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

This is the first array port in this repo's history that touches the array-based forward pass
itself, not just the weight-update rule - `np.where`/`np.maximum` are genuinely new operations
against [the numpy interface subset](numpy-interface-subset.md#explicitly-not-required)'s own
documented scope, which explicitly named `np.maximum`/`np.where` as deferred exactly to "a future
follow-on covering a specific variant (say, a vectorized ReLU sibling)" - this workplan is that
follow-on, and the Rust core's own operation set needs the equivalent primitive added
(`array_relu`/`array_relu_mask`-shaped, mirroring the numpy calls above) before
`ReLURustArrayLayer` can be built - a real, small extension to [the Rust array
core](rust-array-core.md)'s scope, not assumed to already exist there.

## correctness validation

Fully deterministic on both sides (no RNG involved anywhere in ReLU's own math) - a genuine
bit-close parity check, the cleanest of this whole round:

- `test_relu_array_layer.py`: `forward`/`forward_batch`/`compute_hidden_delta`/
  `compute_hidden_delta_batch` checked directly against `ReLUNode`/`ReLULayer`'s own per-node
  formulas across a random sweep of weights/inputs - including the `z == 0` boundary case
  explicitly (both this class's `> 0.0` mask and `relu_hidden_delta`'s own either-branch
  convention agree it's zero-derivative there, measure-zero in practice but worth one explicit
  test rather than leaving it implicit).
- `test_relu_vectorized_multiclass_backprop_model.py`: whole-network parity against a test-only
  `ReLUMultiClassBackpropClassifierNetwork` reference (`hidden_layer_cls = ReLULayer`), the same
  `tests/helpers.py`-hoisted weight-injection pattern used throughout this round.
- `test_relu_rust_array_layer.py`/`test_relu_rust_array_multiclass_backprop_model.py` +
  `rust/indrajala_ml_array/tests/test_array_relu.py`: the Rust core's new `array_relu` primitive
  parity-tested against numpy directly (the same treatment every existing Rust array operation
  gets - [the Rust array core](rust-array-core.md)), then the same two model-level tiers against
  the Rust-backed classes.

## measurement plan

- **Wall-clock**: the standard fused-layer benchmark (`dimension=784, hidden=10`,
  per-node/numpy/Rust, several batch sizes) - expect a similar large win to every other sibling in
  this round, since ReLU's own extra cost (one `max`/mask op) is cheap relative to the
  matmul-dominated baseline.
- **Accuracy, at the scale the per-node result never checked**: reproduce [ReLU's own tuned-XOR
  result](research-backprop-siblings.md#relu-hidden-layer-activation-a-clean-win-once-retuned)'s
  learning-rate retune methodology, but on UCI digits and real MNIST (the scale sigmoid saturation
  was actually measured to hurt at) - does ReLU's toy-scale win transfer, and does it still need
  its own retuned learning rate the way [binary
  cross-entropy](binary-cross-entropy-array-layer.md) and [softmax](softmax-array-layer.md) each
  needed theirs, once their own array/Rust ports are measured at this scale too?

## risks and open questions

- **The Rust core's operation set needs a real, if small, extension** (`np.maximum`/masking) -
  flagged above under "design," not assumed to be a drop-in fused call the way every prior
  sibling in this codebase's array-porting history has been.
- **Dead-ReLU risk at real-MNIST scale, not just XOR** - the per-node investigation checked and
  ruled out dead units at XOR scale ([ReLU hidden-layer
  activation](research-backprop-siblings.md#relu-hidden-layer-activation-a-clean-win-once-retuned));
  fan-in-aware init's own smaller weight range at high fan-in could interact differently with
  ReLU's zero-derivative-below-zero region at MNIST's 784-dimension scale - worth checking
  directly rather than assuming the XOR-scale finding transfers.
- **The single-output array-based line gap** - inherited, unresolved, same as every sibling in
  this round (immaterial here since ReLU is hidden-layer-only and the multiclass array line
  already has hidden layers to swap).

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. The Rust core's `array_relu` primitive + its own parity tests (independent of any Python
   model-class work, per [the Rust array core](rust-array-core.md)'s own layering).
3. `ReLUArrayLayer` + `ReLUVectorizedMultiClassBackpropClassifierNetwork` + the parity-check tests
   above (numpy only).
4. The wall-clock/accuracy measurement described above.
5. Docs closeout.
6. `ReLURustArrayLayer` + `ReLURustArrayMultiClassBackpropClassifierNetwork` + parity-check tests
   at every tier, built on stage 2's primitive.
