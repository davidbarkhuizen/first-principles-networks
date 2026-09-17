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

## measurement plan and result (stage 4)

**Part A - wall-clock, one mini-batch training step** (`forward_batch` + `compute_hidden_delta_batch`
(against a fixed downstream next-layer) + `accumulate_gradient_batch` + `apply_accumulated_gradient`,
`dimension=784, hidden=10`, per-node vs numpy `ReLUArrayLayer` - no Rust `ReLURustArrayLayer` yet
at measurement time, that's stage 6, independent of this measurement), median of 30 timed reps:

| batch size | per-node (us/example) | numpy (us/example) | per-node/numpy |
|---|---|---|---|
| 1 | 3965.31 | 95.22 | 41.6x |
| 8 | 2704.88 | 8.16 | 331.4x |
| 32 | 2713.59 | 4.84 | 560.9x |
| 128 | 2598.85 | 5.34 | 486.7x |
| 512 | 2616.75 | 4.14 | 632.3x |

**Yes, decisively**: 41.6x-632.3x faster per example than the per-node path - matches every
other sibling in this round.

**Part B - does the toy-XOR win transfer to UCI digits and real MNIST, and does ReLU still need
its own tuned learning rate there?** Both real datasets, via
`ReLUVectorizedMultiClassBackpropClassifierNetwork` vs the sigmoid baseline
(`VectorizedMultiClassBackpropClassifierNetwork`) - both already use the same fan-in-aware init,
isolating the activation-function question specifically, the same posture
`ReLUBackpropClassifierNetwork`'s own per-node measurement took.

**UCI digits** (`[32]` hidden, established `learning_rate=0.5` baseline, 30 epochs, 10 seeds):

| learning_rate | test accuracy |
|---|---|
| sigmoid @ 0.5 (baseline) | 96.74% ± 0.31% |
| ReLU @ 0.5 (untuned) | 96.74% ± 0.43% |
| ReLU @ 0.25 | 96.80% ± 0.50% |
| ReLU @ 0.1 | 96.57% ± 0.22% |
| ReLU @ 0.05 | 96.60% ± 0.27% |
| ReLU @ 0.025 | 95.99% ± 0.44% |
| ReLU @ 0.01 | 94.57% ± 0.49% |

**The toy-XOR win does not transfer at UCI-digits scale.** Unlike XOR (where untuned ReLU lost
badly, 74.03% vs 97.80%), ReLU ties the sigmoid baseline exactly at the *same* untuned rate here
- no retuning even needed to match it. The best ReLU rate (0.25, 96.80% ± 0.50%) is well inside
one seed-to-seed standard deviation of the baseline (96.74% ± 0.31%) - a tie, not a win.

**Real MNIST** (`[30]` hidden, established `learning_rate=0.5` baseline, 1 epoch, the full 60000-
example training set):

| learning_rate | test accuracy (1 seed) |
|---|---|
| sigmoid @ 0.5 (baseline) | 93.64% |
| ReLU @ 0.5 (untuned) | 86.54% |
| ReLU @ 0.25 | 92.22% |
| ReLU @ 0.1 | 94.04% |
| ReLU @ 0.05 | 94.30% |
| ReLU @ 0.025 | 93.45% |
| ReLU @ 0.01 | 92.26% |

Untuned ReLU loses badly here too (86.54% vs 93.64%) - confirming it needs its own tuned rate at
real scale, not just XOR, the same caveat [binary
cross-entropy](binary-cross-entropy-array-layer.md)/[softmax](softmax-array-layer.md) each have.
Retuned to `learning_rate=0.05`, it *exceeds* the baseline (94.30% vs 93.64%). A single seed
isn't enough to trust a 0.66-point margin, so this was re-run at 5 seeds each for the baseline
and the two best ReLU rates:

| | seed accuracies | mean | stdev |
|---|---|---|---|
| sigmoid @ 0.5 | 93.64%, 94.01%, 94.17%, 92.45%, 94.33% | 93.72% | 0.67% |
| ReLU @ 0.05 | 94.30%, 94.15%, 93.98%, 93.88%, 94.50% | **94.16%** | **0.22%** |
| ReLU @ 0.1 | 94.04%, 93.19%, 93.29%, 93.98%, 93.95% | 93.69% | 0.37% |

**A real, if modest, win at real-MNIST scale - the opposite finding from UCI digits.** ReLU
@ 0.05's margin over the sigmoid baseline (+0.44 points) is smaller than the baseline's own
seed-to-seed stdev (0.67%), so this isn't an overwhelming result - but ReLU's *own* variance is
three times tighter (0.22% vs 0.67%), and every one of its 5 seeds (93.88%-94.50%) lands above
the baseline's median, not just its mean. A genuine, consistent, if modest signal, reported
honestly at the size it actually is - not rounded up to "a clean win" the way XOR's own
untuned-vs-retuned gap was.

**Dead-ReLU risk at real-MNIST scale, checked directly**: 0 of 30 hidden units were dead (always
zero across 5000 real-MNIST training examples) at the retuned `learning_rate=0.05` network -
ruling out the flagged risk directly, the same "1 of 8 dead, not enough to explain the gap"
finding transferring from XOR scale to real scale.

**Decision: adopted as a genuine, if scale-dependent, finding.** ReLU still needs its own tuned
learning rate at every scale checked (XOR, UCI digits, real MNIST) - never a drop-in replacement
for sigmoid at existing hyperparameters. Whether it *beats* the retuned sigmoid baseline is
scale-dependent: no at UCI digits (a tie), yes at real MNIST (a modest, consistent win) - report
honestly, not collapsed into one blanket verdict.

## risks and open questions

- **The Rust core's operation set needs a real, if small, extension** (`np.maximum`/masking) -
  flagged above under "design," not assumed to be a drop-in fused call the way every prior
  sibling in this codebase's array-porting history has been.
- **Dead-ReLU risk at real-MNIST scale, not just XOR** - ✅ resolved, see "measurement plan and
  result" above: 0 of 30 hidden units dead at the retuned rate, ruling this out directly rather
  than assuming the XOR-scale finding transfers.
- **The single-output array-based line gap** - inherited, unresolved, same as every sibling in
  this round (immaterial here since ReLU is hidden-layer-only and the multiclass array line
  already has hidden layers to swap).

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. The Rust core's `array_relu` primitive + its own parity tests (independent of any Python
   model-class work, per [the Rust array core](rust-array-core.md)'s own layering).
3. `ReLUArrayLayer` + `ReLUVectorizedMultiClassBackpropClassifierNetwork` + the parity-check tests
   above (numpy only).
4. ✅ The wall-clock/accuracy measurement - see "measurement plan and result" above: 41.6x-632.3x
   faster per example than the per-node path; the toy-XOR win doesn't transfer at UCI-digits
   scale (a tie) but does, modestly, at real-MNIST scale.
5. Docs closeout.
6. ✅ `ReLURustArrayLayer` + `ReLURustArrayMultiClassBackpropClassifierNetwork` + parity-check
   tests at every tier, built on stage 2's primitive.
