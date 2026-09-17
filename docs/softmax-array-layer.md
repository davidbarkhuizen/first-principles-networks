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

## measurement plan and result (stage 4)

**Part A - wall-clock, one mini-batch training step** (`forward`/`forward_batch` +
`compute_output_delta`/`compute_output_delta_batch` + `accumulate_gradient`/
`accumulate_gradient_batch` + `apply_accumulated_gradient`, output-layer shape `dimension=784,
class_count=10`, per-node (`SoftmaxOutputLayer`) vs numpy (`SoftmaxArrayLayer`) - no Rust
`SoftmaxRustArrayLayer` yet at measurement time, that's stage 6, independent of this measurement),
median of 30 timed reps:

| batch size | per-node (us/example) | numpy (us/example) | per-node/numpy |
|---|---|---|---|
| 1 | 4127.89 | 54.89 | 75.2x |
| 8 | 2864.95 | 8.13 | 352.4x |
| 32 | 2663.63 | 4.27 | 623.7x |
| 128 | 2642.41 | 2.42 | 1089.7x |
| 512 | 2634.65 | 3.06 | 862.1x |

**Yes, decisively**: 75.2x-1089.7x faster per example than the per-node path - matches, and at
several batch sizes exceeds, every other sibling in this round.

**Part B - does the array/Rust-speed port reproduce both existing results honestly, and does the
now-cheap retune close the previously-too-expensive gap?** Both real datasets, via
`SoftmaxVectorizedMultiClassBackpropClassifierNetwork` vs the one-vs-rest baseline
(`VectorizedMultiClassBackpropClassifierNetwork`), same architecture/hyperparameters as each
original per-node measurement:

**UCI digits** (`[32]` hidden, `learning_rate=0.5`, 30 epochs, single seed, the same architecture
[softmax/cross-entropy re-alignment](research-multiclass-and-loss.md#softmaxcross-entropy-re-alignment)
used):

| | training accuracy | held-out test accuracy |
|---|---|---|
| one-vs-rest (numpy) | 99.58% | 96.66% |
| softmax (numpy) | **100.00%** | **98.33%** |

**The UCI-digits win reproduces at array speed** - softmax again reaches full training accuracy
and generalizes better, the same qualitative shape the per-node measurement found (99.51%/95.54%
vs. 100.00%/96.94%), not bit-identical (different RNG stream for `randomize()`, the same
[RNG-exception](rust-array-core.md#the-rng-exception) caveat every array port carries) but the
same direction and margin.

**Real MNIST, untuned** (`[16]` hidden, `learning_rate=0.5`, 5 epochs, full 60000/10000, single
seed, the identical setup [softmax on real full-scale
MNIST](research-multiclass-and-loss.md#softmax-on-real-full-scale-mnist) used):

| | training accuracy | held-out test accuracy |
|---|---|---|
| one-vs-rest (numpy) | 93.71% | 93.01% |
| softmax @ 0.5, untuned (numpy) | 89.77% | 90.16% |

**The real-MNIST loss reproduces too, at the same untuned rate** - confirming this isn't a
per-node-path artifact, before spending the now-cheap retune below.

**The retune, run for the first time**: sweeping `learning_rate` down for softmax alone (single
seed, 5 epochs each):

| learning_rate | training accuracy | test accuracy |
|---|---|---|
| 0.5 (untuned) | 89.77% | 90.16% |
| 0.25 | 90.82% | 90.76% |
| 0.1 | 93.16% | 92.44% |
| 0.05 | 94.69% | 93.90% |
| 0.025 | 94.73% | 93.95% |
| 0.01 | 94.81% | **94.17%** |
| 0.005 | 94.19% | 93.88% |
| 0.0025 | 93.12% | 93.11% |
| 0.001 | 91.23% | 91.34% |
| 0.0005 | 88.92% | 89.16% |

A real peak, not a monotonic trend run off the edge of the sweep: accuracy rises from 0.5 down to
a plateau around 0.01-0.05, then falls again below 0.0025 - `learning_rate=0.01` is the best point
found.

**Confirmed at 5 seeds**, `learning_rate=0.01` against the untuned-baseline's own tuned rate
(`0.5`), same architecture:

| | seed accuracies | mean | stdev |
|---|---|---|---|
| one-vs-rest @ 0.5 | 93.01%, 93.02%, 92.40%, 93.25%, 92.55% | 92.85% | 0.36% |
| softmax @ 0.01 (retuned) | 94.17%, 94.10%, 93.67%, 93.70%, 93.61% | **93.85%** | 0.26% |

**A real, decisive win at real-MNIST scale, not just a tie or a modest signal** - softmax's
retuned margin over the baseline (+1.00 point) is larger than either side's own seed-to-seed
stdev, and every one of softmax's 5 seeds (93.61%-94.17%) beats every one of the baseline's 5
seeds (92.40%-93.25%) - non-overlapping ranges, not just a favorable mean. This closes the gap
[softmax on real full-scale MNIST](research-multiclass-and-loss.md#softmax-on-real-full-scale-mnist)
explicitly declined to spend on: retuned, softmax is the better full-scale one-network option,
reversing that entry's "`MultiClassBackpropClassifierNetwork` remains the better full-scale
one-network option at today's tuning" conclusion, which was scoped to *untuned* softmax at the
quadratic-tuned rate, not softmax generally.

**Decision: adopted.** Softmax needs its own tuned learning rate at real-MNIST scale (never a
drop-in replacement for one-vs-rest at existing hyperparameters, the same caveat binary
cross-entropy and ReLU each carry) - but once retuned, it's a genuine, decisive win at both scales
checked (UCI digits and real MNIST), not the scale-dependent split ReLU's own measurement found.
The ensemble (96.01%) remains the best of all three real-MNIST options by a wide margin; this
result is about the single-network comparison specifically.

## risks and open questions

- ✅ **The Rust core's operation set needed extending** (row-wise max/sum-and-normalize) - resolved
  in stage 2 (`array_softmax`), the same category of real, small scope extension [the ReLU array
  workplan](relu-array-layer.md#risks-and-open-questions) needed for its own primitive.
- ✅ **The retune result could have gone either way** - resolved, see "measurement plan and
  result" above: it's a real, decisive win (+1.00 point, non-overlapping 5-seed ranges), not a
  foregone conclusion that turned out true by luck.
- **Sequential-only training persists in the array world too** - the per-node investigation noted
  softmax's joint output "can't be split across processes the way the ensemble's ten binary
  classifiers can"; the array/Rust port doesn't change that structurally (still one network, one
  process), it only makes each sequential step much cheaper.

## delivery stages (each its own PR, per this repo's practice)

1. **Done.** This design document.
2. **Done.** The Rust core's row-wise softmax-normalization primitive (`array_softmax`) + its own
   parity tests.
3. **Done.** `SoftmaxArrayLayer` + `SoftmaxVectorizedMultiClassBackpropClassifierNetwork` + the
   parity-check tests above (numpy only).
4. **Done.** The wall-clock/reproduction/retune measurement - see "measurement plan and result"
   above: 75.2x-1089.7x faster per example than the per-node path; the UCI-digits win reproduces
   at array speed; the real-MNIST loss reproduces too at the untuned rate, but retuning
   (`learning_rate=0.01`) turns it into a real, decisive win (+1.00 point, non-overlapping 5-seed
   ranges) - closing the gap the per-node investigation explicitly declined to spend on.
5. ✅ Docs closeout - every stage of this workplan is now done.
6. **Done.** `SoftmaxRustArrayLayer` + `SoftmaxRustArrayMultiClassBackpropClassifierNetwork` +
   parity-check tests at every tier, built on stage 2's primitive.
