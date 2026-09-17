# an array-based (Rust-matmul-backed) binary cross-entropy sibling

[← back to README](../README.md)

**Status: proposed, not started; depends on [an array-based ensemble
sibling](ensemble-array-layer.md)'s own single-output array network.** Written up front as a
design/measurement plan before any of it exists, per this repo's own practice (see [Adam
optimizer](adam-optimizer.md), [an array-based Adam sibling](adam-array-layer.md) for precedent).

## why this, and why now

`BinaryCrossEntropyBackpropClassifierNetwork` (`binary_cross_entropy_backprop_classifier_network.py`'s
`CrossEntropyOutputNode`/`CrossEntropyOutputLayer`) is still per-node-only. Its own measurement
([binary cross-entropy for
BackpropClassifierNetwork](research-multiclass-and-loss.md#binary-cross-entropy-for-backpropclassifiernetwork))
found it *matches*, not beats, quadratic loss's own tuned performance once retuned to a
substantially lower learning rate (97.60% at `learning_rate=0.1` vs. quadratic's 97.80% at its own
tuned rate) - not a clean win the way ReLU's was, but a real, semantically-correct capability
("genuinely useful, cheap to add," per that class's own docstring) whose extension to real-MNIST
ensemble training was explicitly deferred as separate future work: "not extended to
`EnsembleBackpropClassifierNetwork`'s real-MNIST training as part of this work - that needed its
own dedicated retuning investigation, given the cost of each real training run." That cost is
exactly what an array/Rust port removes.

## scope, and a real, specific blocking dependency found by checking, not assumed away

`BinaryCrossEntropyBackpropClassifierNetwork` is single-output (`output_layer_cls` on
`BackpropClassifierNetwork`) - the same single-output-line gap every sibling in this round
inherits from [an array-based Adam sibling](adam-array-layer.md#risks-and-open-questions).
Unlike momentum/L2/ReLU/dropout, though, this gap can't be worked around the way those did (by
scoping onto the existing `class_count`-many-outputs multiclass array line instead): a genuine,
literal parity check against `CrossEntropyOutputLayer`'s single-node behavior needs a
size-1-output host, and `VectorizedMultiClassBackpropClassifierNetwork`'s own constructor calls
`validate_class_count`, which asserts `class_count >= 2` - checked directly in `bounds.py`, not
assumed - so `class_count=1` is not an option on the existing array line at all. A cross-entropy
output *delta* formula (`a - target`) generalizes fine to any output size (it's exactly
`SoftmaxOutputNode`'s own simplification, applied independently per node instead of jointly), so
a `CrossEntropyArrayLayer(ArrayLayer)` overriding only `compute_output_delta`/
`compute_output_delta_batch` the same way [momentum's](momentum-array-layer.md)/[L2's](l2-array-layer.md)
own single-method overrides do is not the hard part - having a genuine single-output array
network to attach it to, and compare against `BinaryCrossEntropyBackpropClassifierNetwork`
itself, is. This workplan therefore depends on [an array-based ensemble
sibling](ensemble-array-layer.md)'s own proposed single-output array network
(`ArrayBackpropClassifierNetwork`) rather than duplicating that build here.

## design

```python
class CrossEntropyArrayLayer(ArrayLayer):
    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = self.a - reference           # drops the a*(1-a) factor
                                                     # ArrayLayer.compute_output_delta has

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = self.A - reference_batch
```

`forward`/`forward_batch` are inherited unchanged from `ArrayLayer` - unlike `SoftmaxArrayLayer`,
a single sigmoid output needs nothing from any sibling, the same point
`CrossEntropyOutputNode`'s own docstring makes ("a single output node's activation needs nothing
from any sibling - `forward()` is inherited completely unchanged"). Once [the single-output array
network](ensemble-array-layer.md) exists, `CrossEntropyVectorizedArrayBackpropClassifierNetwork`
sets its `output_layer` to a `CrossEntropyArrayLayer` instead of a plain `ArrayLayer` - a single
class-attribute-shaped override, mirroring `BinaryCrossEntropyBackpropClassifierNetwork`'s own
"structurally this is a single class-attribute override" shape exactly.

The same `CrossEntropyArrayLayer` also works unchanged as a `class_count`-sized output on the
*existing* multiclass array line (an independent per-node cross-entropy delta at each of
`class_count` outputs - a one-vs-rest-with-cross-entropy-loss variant, distinct from softmax's
jointly-normalized one) - worth building and parity-testing there too, since it needs no new
host, even though it isn't the literal array counterpart of the single-output
`BinaryCrossEntropyBackpropClassifierNetwork` this workplan is primarily about.

## correctness validation

Fully deterministic given the same starting weights (no RNG in cross-entropy's own delta
formula):

- `test_binary_cross_entropy_array_layer.py`: `CrossEntropyArrayLayer.compute_output_delta`/
  `compute_output_delta_batch` checked directly against `CrossEntropyOutputNode.compute_output_delta`
  across a random sweep of activations/targets - a one-line formula, but still checked the same
  way every other sibling in this codebase's history has been, not assumed correct because it's
  short.
- Once [the single-output array network](ensemble-array-layer.md) exists:
  `test_binary_cross_entropy_vectorized_backprop_model.py`, whole-network parity against
  `BinaryCrossEntropyBackpropClassifierNetwork` itself directly (a genuine per-node reference
  already exists here) via a weight-injection helper in the same shape
  `matching_adam_array_backprop_networks` established.
- The multiclass-line variant (see "design" above) gets its own
  `test_cross_entropy_vectorized_multiclass_backprop_model.py`, parity-checked against a test-only
  per-node reference the same way [momentum's](momentum-array-layer.md#correctness-validation)/
  [L2's](l2-array-layer.md#correctness-validation) workplans propose for theirs.

## measurement plan

- **Wall-clock**: the standard fused-layer benchmark, output-layer shape - expect a large win in
  line with every sibling in this round (this is the cheapest possible override, a single
  elementwise subtract).
- **The retune the per-node investigation explicitly deferred**: once [the single-output array
  network](ensemble-array-layer.md) and its ensemble wrapper exist, sweep `learning_rate` for
  `CrossEntropyArrayLayer`-based ensemble sub-networks on real MNIST, directly answering "does
  this toy-problem finding transfer to MNIST's scale" - the exact open question [binary
  cross-entropy for
  BackpropClassifierNetwork](research-multiclass-and-loss.md#binary-cross-entropy-for-backpropclassifiernetwork)
  left for "its own dedicated retuning investigation," now affordable.

## risks and open questions

- **This workplan is blocked on [an array-based ensemble
  sibling](ensemble-array-layer.md)'s single-output array network landing first** - a real
  sequencing dependency, stated explicitly under "scope" above, not assumed away.
- **A reconfirmed "matches, doesn't beat" result is a legitimate outcome** - same framing as
  every sibling in this round; this workplan's main value is making the deferred real-MNIST
  retune investigation affordable, not a guaranteed new win.

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. `CrossEntropyArrayLayer` (usable standalone against the existing multiclass array line
   immediately - see "design" above) + its own parity-check tests. Independent of stage 3-5's
   dependency on the ensemble workplan.
3. **Blocked on [an array-based ensemble sibling](ensemble-array-layer.md)'s
   `ArrayBackpropClassifierNetwork` landing**: the single-output cross-entropy network + parity
   tests against `BinaryCrossEntropyBackpropClassifierNetwork`.
4. The real-MNIST retune measurement described above.
5. Docs closeout.
6. The Rust-matmul-backed counterpart, once stage 3's numpy version and [the ensemble
   workplan](ensemble-array-layer.md)'s own Rust stage both exist.
