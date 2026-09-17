# an array-based (Rust-matmul-backed) softmax sibling

[← back to README](../README.md)

**Status: proposed, not started.** Written up front as a design/measurement plan before any of
it exists, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based
Adam sibling](adam-array-layer.md) for precedent).

## why this, and why now

`SoftmaxMultiClassBackpropClassifierNetwork` (`softmax_output_layer.py`'s `SoftmaxOutputNode`/
`SoftmaxOutputLayer`) is still per-node-only, and it's the canonical treatment for a mutually
exclusive multi-class target like digit classification - the one-vs-rest default it corrects was
an unexamined incidental choice, not a considered tradeoff
([softmax/cross-entropy re-alignment](research-multiclass-and-loss.md#softmaxcross-entropy-re-alignment)).
Its own measured record is scale-dependent, and worth stating plainly rather than oversold: a real
win at UCI-digits scale (96.94% vs. 95.54% test accuracy, full training accuracy in about half the
epochs -
[softmax/cross-entropy re-alignment](research-multiclass-and-loss.md#softmaxcross-entropy-re-alignment)),
but a loss at real-MNIST scale under the quadratic-tuned learning rate, never retuned for softmax
(89.12% vs. 92.75% -
[softmax on real full-scale MNIST](research-multiclass-and-loss.md#softmax-on-real-full-scale-mnist)).
That entry explicitly declined to retune further, citing cost: "not retuned further (the same
cost/value tradeoff the binary cross-entropy investigation already declined to spend on)." That
cost was a real-MNIST training run at the per-node path's own wall-clock cost (~30+ minutes per
run). A vectorized/Rust-backed softmax sibling removes exactly that excuse - a learning-rate sweep
that was too expensive to run once becomes cheap to run many times.

## scope

`SoftmaxArrayLayer` (numpy) and `SoftmaxRustArrayLayer` (Rust-matmul-backed), output-layer-only -
unlike every other sibling in this round, softmax is naturally multiclass-only already (it needs
`class_count >= 2` to normalize over, exactly `SoftmaxOutputLayer`'s own `assert size >= 2`), so
there's no single-output-line scoping gap to inherit here: `SoftmaxVectorizedMultiClassBackpropClassifierNetwork`/
`SoftmaxRustArrayMultiClassBackpropClassifierNetwork` are natural, complete counterparts, not a
narrowed subset the way momentum/L2/ReLU's workplans are.

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
[structure](structure.md#multi-class)): `compute_hidden_delta` for whatever layer feeds this one
is untouched (it only ever reads `next_layer.W`/`next_layer.delta`, never `next_layer.a`
directly), so no other class in `VectorizedMultiClassBackpropClassifierNetwork`'s chain needs any
change - the same "single class-attribute override, zero backward-pass-plumbing changes" shape the
per-node version already achieved.

The row-wise batched max/sum (`axis=1, keepdims=True`) is a genuinely new operation against [the
numpy interface subset](numpy-interface-subset.md#explicitly-not-required)'s documented scope
("general axis-parameterized reductions ... belong to variants ... this subset ... defers");
`SoftmaxRustArrayLayer` needs the Rust core's matmul-and-reduction primitives extended with a
row-wise max/sum-and-normalize op before it can be built - flagged the same way [the ReLU array
workplan](relu-array-layer.md#design-forward-and-compute_hidden_delta-both-change-no-persistent-state)
flags its own operation-set extension.

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

## measurement plan

- **Wall-clock**: the standard fused-layer benchmark, output-layer shape (`dimension=784,
  class_count=10`) - expect a large win in line with every other sibling in this round.
- **Reproduce both existing results, honestly, at array/Rust speed**: the UCI-digits win and the
  real-MNIST loss, both exactly as measured on the per-node path, to confirm the port preserves
  the same qualitative (not necessarily bit-identical - different RNG stream for `randomize()`,
  same [RNG-exception](rust-array-core.md#the-rng-exception) caveat every array port carries)
  shape before drawing any new conclusion from it.
- **The retune the per-node investigation explicitly declined to spend on**: now that a real-MNIST
  training run is cheap, sweep `learning_rate` down for softmax alone on the identical real-MNIST
  setup [softmax on real full-scale MNIST](research-multiclass-and-loss.md#softmax-on-real-full-scale-mnist)
  used, the same methodology [binary cross-entropy's own
  retune](research-multiclass-and-loss.md#binary-cross-entropy-for-backpropclassifiernetwork) and
  [ReLU's own retune](research-backprop-siblings.md#relu-hidden-layer-activation-a-clean-win-once-retuned)
  already used at smaller scale. This directly closes the flagged, previously-too-expensive gap.

## risks and open questions

- **The Rust core's operation set needs extending** (row-wise max/sum-and-normalize) - same
  category of real, small scope extension [the ReLU array workplan](relu-array-layer.md#risks-and-open-questions)
  flags for its own primitive.
- **The retune result could go either way** - flagged per the updated mandate behind this round:
  worth running and reporting honestly, not a foregone conclusion that retuning will make softmax
  win at MNIST scale.
- **Sequential-only training persists in the array world too** - the per-node investigation noted
  softmax's joint output "can't be split across processes the way the ensemble's ten binary
  classifiers can"; the array/Rust port doesn't change that structurally (still one network, one
  process), it only makes each sequential step much cheaper.

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. The Rust core's row-wise softmax-normalization primitive + its own parity tests.
3. `SoftmaxArrayLayer` + `SoftmaxVectorizedMultiClassBackpropClassifierNetwork` + the
   parity-check tests above (numpy only).
4. The wall-clock/reproduction/retune measurement described above.
5. Docs closeout.
6. `SoftmaxRustArrayLayer` + `SoftmaxRustArrayMultiClassBackpropClassifierNetwork` + parity-check
   tests at every tier, built on stage 2's primitive.
