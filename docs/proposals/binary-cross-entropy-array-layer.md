# an array-based (Rust-matmul-backed) binary cross-entropy sibling

[← back to README](../../README.md)

**Status: stages 1-5 done.** `CrossEntropyArrayLayer`, its multiclass-line variant
(`CrossEntropyVectorizedMultiClassBackpropClassifierNetwork`), and the single-output
`CrossEntropyArrayBackpropClassifierNetwork` are built and parity-tested against their per-node
references. The deferred real-MNIST retune measurement (stage 4) is done - see [an array-based
binary cross-entropy
sibling](../research/research-multiclass-and-loss.md#an-array-based-binary-cross-entropy-sibling-the-retune-the-per-node-investigation-deferred)
for the full result: a genuine, if modest, win once retuned (+0.37 points over the documented
ensemble baseline at `learning_rate=0.1`, every seed beating every baseline seed) - a more
favorable outcome than the toy-XOR "matches, doesn't beat" finding this workplan set out to check.
Stage 6 (the Rust-matmul-backed counterpart) remains.

## why this, and why now

`BinaryCrossEntropyBackpropClassifierNetwork` (`binary_cross_entropy_backprop_classifier_network.py`'s
`CrossEntropyOutputNode`/`CrossEntropyOutputLayer`) is still per-node-only. Its own measurement
([binary cross-entropy for
BackpropClassifierNetwork](../research/research-multiclass-and-loss.md#binary-cross-entropy-for-backpropclassifiernetwork))
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
`BackpropClassifierNetwork`) - the same single-output-line gap every array-ported sibling in this
codebase's history has inherited (no array-based `MultiClassBackpropClassifierNetwork` counterpart
supports `class_count=1`). Unlike momentum/L2/ReLU/dropout, though, this gap can't be worked
around the way those did (by scoping onto the existing `class_count`-many-outputs multiclass array
line instead): a genuine, literal parity check against `CrossEntropyOutputLayer`'s single-node
behavior needs a size-1-output host, and `VectorizedMultiClassBackpropClassifierNetwork`'s own
constructor calls `validate_class_count`, which asserts `class_count >= 2` - checked directly in
`bounds.py`, not assumed - so `class_count=1` is not an option on the existing array line at all. A
cross-entropy output *delta* formula (`a - target`) generalizes fine to any output size (it's
exactly `SoftmaxOutputNode`'s own simplification, applied independently per node instead of
jointly), so a `CrossEntropyArrayLayer(ArrayLayer)` overriding only `compute_output_delta`/
`compute_output_delta_batch` the same way momentum's/L2's own single-method overrides do (see
[research and
analysis](../research/research-backprop-siblings.md#an-array-based-momentum-sibling-unchanged-on-stronger-evidence))
is not the hard part - having a genuine single-output array
network to attach it to, and compare against `BinaryCrossEntropyBackpropClassifierNetwork`
itself, is. This workplan therefore depended on [an array-based ensemble
sibling](../design-docs/ensemble/ensemble-array-layer.md)'s own single-output array network
(`ArrayBackpropClassifierNetwork`) landing first, rather than duplicating that build here.

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
from any sibling - `forward()` is inherited completely unchanged"). **Built as**
`CrossEntropyArrayBackpropClassifierNetwork`, not the
`CrossEntropyVectorizedArrayBackpropClassifierNetwork` name this section originally sketched - the
`Vectorized` prefix is this codebase's own naming convention for the `class_count`-wide multiclass
line (`VectorizedMultiClassBackpropClassifierNetwork`), not the single-output line
(`ArrayBackpropClassifierNetwork`, no prefix), checked directly against the landed code rather
than assumed. It sets its `output_layer` to a `CrossEntropyArrayLayer` instead of a plain
`ArrayLayer` - a single class-attribute-shaped override, mirroring
`BinaryCrossEntropyBackpropClassifierNetwork`'s own "structurally this is a single class-attribute
override" shape exactly.

The same `CrossEntropyArrayLayer` also works unchanged as a `class_count`-sized output on the
*existing* multiclass array line (an independent per-node cross-entropy delta at each of
`class_count` outputs - a one-vs-rest-with-cross-entropy-loss variant, distinct from softmax's
jointly-normalized one) - built as `CrossEntropyVectorizedMultiClassBackpropClassifierNetwork`,
mirroring `SoftmaxVectorizedMultiClassBackpropClassifierNetwork`'s own precedent (a wholly separate
class, not a subclass), even though it isn't the literal array counterpart of the single-output
`BinaryCrossEntropyBackpropClassifierNetwork` this workplan is primarily about.

## correctness validation

Fully deterministic given the same starting weights (no RNG in cross-entropy's own delta
formula):

- `test_cross_entropy_array_layer.py`: `CrossEntropyArrayLayer.compute_output_delta`/
  `compute_output_delta_batch` checked directly against `CrossEntropyOutputNode.compute_output_delta`
  across a random sweep of activations/targets - a one-line formula, but still checked the same
  way every other sibling in this codebase's history has been, not assumed correct because it's
  short.
- `test_cross_entropy_array_backprop_model.py`: whole-network parity against
  `BinaryCrossEntropyBackpropClassifierNetwork` itself directly (a genuine per-node reference
  already exists here) via `tests/helpers.py`'s new `matching_cross_entropy_array_backprop_networks`
  helper, in the same shape `matching_adam_array_backprop_networks` established.
- The multiclass-line variant (see "design" above) gets its own
  `test_cross_entropy_vectorized_multiclass_backprop_model.py`, parity-checked against a new
  test-only per-node reference (`CrossEntropyMultiClassBackpropClassifierNetwork`, `tests/helpers.py`),
  the same two-tier convention every array-ported sibling in this codebase uses.

## measurement plan - done

- **Wall-clock**: no separate benchmark needed - `CrossEntropyArrayLayer` differs from plain
  `ArrayLayer` by one elementwise subtract only, and the stage 4 sweep confirmed directly (not
  just assumed) that its per-run wall-clock is indistinguishable from the quadratic baseline's own
  (see the research doc entry below).
- **The retune the per-node investigation explicitly deferred**: done - see [an array-based binary
  cross-entropy
  sibling](../research/research-multiclass-and-loss.md#an-array-based-binary-cross-entropy-sibling-the-retune-the-per-node-investigation-deferred)
  for the full sweep and result. **A different, more favorable outcome than the toy-XOR finding**:
  retuned to `learning_rate=0.1`, cross-entropy doesn't just match the ensemble's documented
  quadratic baseline the way it matched quadratic on toy XOR - it beats it by 0.37 points, every
  cross-entropy seed at that rate outperforming every baseline seed.

## risks and open questions - resolved

- **This workplan was blocked on [an array-based ensemble
  sibling](../design-docs/ensemble/ensemble-array-layer.md)'s single-output array network landing
  first** - resolved: that dependency landed, then this whole workplan was built on top of it.
- **A reconfirmed "matches, doesn't beat" result is a legitimate outcome** - resolved differently
  than expected: the real-MNIST retune came back a genuine, if modest, win instead (see
  "measurement plan" above), not just a reconfirmed match.
- **Stage 6's Rust primitive** - checked directly, not assumed: `SoftmaxArrayLayer`'s own delta
  formula (`self.a - reference`) is algebraically identical to what `CrossEntropyArrayLayer` needs,
  and its existing Rust-fused counterpart (`pa.layer_softmax_output_delta`, `fused.rs`) is already
  shape-agnostic (no softmax-specific math) - stage 6 needs no new Rust primitive at all, it can
  call that existing function directly.

## delivery stages (each its own PR, per this repo's practice)

1. This design document. **Done.**
2. `CrossEntropyArrayLayer` + `CrossEntropyVectorizedMultiClassBackpropClassifierNetwork` (the
   multiclass-line variant from "design" above) + parity-check tests. **Done.**
3. `CrossEntropyArrayBackpropClassifierNetwork` (the single-output network) + parity tests against
   `BinaryCrossEntropyBackpropClassifierNetwork`. **Done.**
4. The real-MNIST retune measurement. **Done** - see "measurement plan" above.
5. Docs closeout. **Done** (this update).
6. The Rust-matmul-backed counterpart - `CrossEntropyRustArrayLayer`,
   `CrossEntropyRustArrayBackpropClassifierNetwork`, and the multiclass-line Rust variant. **Not
   yet started** - see "risks and open questions" above for why it needs no new Rust primitive.
