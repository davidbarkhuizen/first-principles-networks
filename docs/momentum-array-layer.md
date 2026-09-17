# an array-based (Rust-matmul-backed) momentum sibling

[← back to README](../README.md)

**Status: proposed, not started.** Written up front as a design/measurement plan before any of
it exists, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based
Adam sibling](adam-array-layer.md) for precedent).

## why this, and why now

`MomentumBackpropClassifierNetwork` (`momentum_layer.py`'s `make_momentum_node_cls`/
`make_momentum_layer_cls`) is still per-node-only. Momentum has been measured null three times on
that path - the original tuned-XOR sweep ([momentum: measured, not worth
adopting](research-backprop-siblings.md#momentum-measured-not-worth-adopting)), a mini-batch
retest that looked like a "rescue" at first ([momentum under mini-batch
gradients](research-backprop-siblings.md#momentum-under-mini-batch-gradients)), and a follow-up
that found the rescue was a learning-rate/batch-size confound, not momentum
([the learning-rate-vs-batch-size follow-up](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)) -
momentum stayed flat-to-harmful once that confound was removed.

That's real evidence, not a reason to skip vectorizing it. The per-node path made every one of
those sweeps expensive to run (small seed counts, a 320-example toy proxy, never real-MNIST
scale) - exactly the constraint this session hit directly with dropout's own abandoned 40+minute
sweep. Vectorizing momentum is now understood as the *cheap* way to get a better-powered answer
(more seeds, a wider grid, real scale), not an expensive follow-on reserved for siblings that
already won. This workplan proposes building the array/Rust-matmul-backed counterpart and
re-running momentum's own question at a scale and seed count the per-node path couldn't afford,
not assuming the null result holds just because the formula is the same.

## scope

`MomentumArrayLayer` (numpy) and `MomentumRustArrayLayer` (Rust-matmul-backed), mirroring [the
Adam array-layer sibling](adam-array-layer.md)'s own two-stage numpy-then-Rust precedent exactly.
Scoped to `MultiClassBackpropClassifierNetwork`'s array line
(`MomentumVectorizedMultiClassBackpropClassifierNetwork`/
`MomentumRustArrayMultiClassBackpropClassifierNetwork`), for the same reason Adam's own workplan
gave: there is still no single-output (`BackpropClassifierNetwork`-equivalent) array-based line to
extend onto instead - see [an array-based Adam sibling](adam-array-layer.md#risks-and-open-questions)'s
own flagged, unresolved gap. `MomentumBackpropClassifierNetwork` sets both `hidden_layer_cls` and
`output_layer_cls` to the same momentum-configured layer class; this workplan does the same at the
array level (both `layers` and `output_layer` built from `MomentumArrayLayer`).

## design: an override-only `apply_accumulated_gradient`, mirroring Adam's own precedent exactly

`MomentumArrayLayer(ArrayLayer)` overrides only `apply_accumulated_gradient`, the same shape
`AdamArrayLayer` used, but simpler - one previous-delta array per parameter tensor, no bias
correction, no second moment:

```python
class MomentumArrayLayer(ArrayLayer):
    def __init__(self, size: int, input_size: int, momentum: float) -> None:
        super().__init__(size, input_size)
        self._momentum = momentum
        self._prev_delta_W = np.zeros((size, input_size))
        self._prev_delta_b = np.zeros(size)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        delta_W = learning_rate * self._grad_W / batch_size + self._momentum * self._prev_delta_W
        delta_b = learning_rate * self._grad_b / batch_size + self._momentum * self._prev_delta_b
        self.W -= delta_W
        self.b -= delta_b
        self._prev_delta_W, self._prev_delta_b = delta_W, delta_b
        self._reset_gradient_accum()
```

Directly mirrors `make_momentum_node_cls`'s per-node formula (`Δw(n) = η·δ·a + α·Δw(n-1)`), one
array op per parameter tensor instead of a per-weight Python loop. `momentum` is a required
constructor argument, no default - the same posture `make_momentum_node_cls` itself takes (this
codebase's own measurements never found a value worth recommending).

Like Adam's own snapshot/restore note: `_prev_delta_W`/`_prev_delta_b` are momentum-specific
per-parameter state a resumed-training scenario would need, but every measurement in this
codebase uses one-shot train-then-evaluate, so `snapshot()`/`restore()` stay `W`/`b`-only,
unchanged from the base class - the same posture `AdamArrayLayer`'s own workplan already resolved
this way, not a new gap.

`MomentumRustArrayLayer(RustArrayLayer)` mirrors `AdamRustArrayLayer`'s own relationship to
`AdamArrayLayer` one level over - the same update expressed as one fused Rust call
(`layer_momentum_apply_accumulated_gradient`, `fused.rs`) instead of a numpy expression, per [the
Rust production cutover plan](rust-production-cutover.md#0b-fuse-each-layer-operation-into-one-rust-call)'s
"fuse the whole method into one Rust call" treatment.

## correctness validation

Same two-tier convention as every array-based sibling, plus the parity-against-per-node-reference
tier [an array-based Adam sibling](adam-array-layer.md#correctness-validation) established:

- `test_momentum_array_layer.py`: `MomentumArrayLayer.apply_accumulated_gradient` checked
  directly against `MomentumBackpropNode.apply_accumulated_gradient` (via
  `make_momentum_layer_cls`) - same starting weights/gradients/`momentum` coefficient, several
  steps (single-example and batched), `np.allclose`-compared after every step. Fully
  deterministic on both sides (no RNG involved in the update rule itself), so this is a genuine
  bit-close parity check, not a qualitative-shape one the way dropout's own port will need.
- `test_momentum_vectorized_multiclass_backprop_model.py`: whole-network parity against a
  test-only `MomentumMultiClassBackpropClassifierNetwork` reference (`hidden_layer_cls =
  output_layer_cls = make_momentum_layer_cls(momentum)`), the same `tests/helpers.py`-hoisted
  weight-injection pattern `matching_adam_array_backprop_networks` established.
- `test_momentum_rust_array_layer.py`/`test_momentum_rust_array_multiclass_backprop_model.py` +
  `rust/indrajala_ml_array/tests/test_momentum_fused_layer_ops.py`: the same two tiers against
  the Rust-backed classes, mirroring stage 5 of [an array-based Adam
  sibling](adam-array-layer.md#the-rust-matmul-backed-counterpart-stage-5).

## measurement plan

No new algorithmic question in the port itself (same formula, different substrate) - the point is
to afford a better-powered re-run of momentum's own question than the per-node path could:

- **Wall-clock**: the same fused-layer-operation benchmark [an array-based Adam
  sibling](adam-array-layer.md#measurement-plan-and-result-stage-3)'s Part A used
  (`dimension=784, hidden=10`, per-node/numpy/Rust, several batch sizes) - expect a similar
  70x-800x-class win, since momentum's extra term is one elementwise multiply-add on top of the
  same matmul-dominated cost Adam's own control run already measured.
- **Accuracy, at a scale the per-node path never afforded**: re-run [the
  learning-rate-vs-batch-size follow-up](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)'s
  own momentum/learning-rate/batch-size grid, but with (a) more seeds (50-100 instead of ~5-10,
  now affordable), (b) a real-MNIST-ensemble-scale run (784-dim, not the 320-example toy proxy) -
  the scale momentum's own literature is most commonly validated at, and one this codebase has
  never actually tested momentum against. Report honestly either way: if the null holds at scale,
  that's a stronger, better-powered confirmation of the existing decision (still not adopted); if
  it doesn't, that's new information this codebase's own rigor ethos would want surfaced, not
  buried under "already known to be a null."

## measurement plan and result (stage 3)

**A correction found while re-checking, not assumed from this document's own earlier framing**:
the "measurement plan" above calls the original follow-up's setup a "320-example toy proxy" to
be moved off of - re-checking
[research-backprop-siblings.md](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)
directly shows that setup was already the 320-example **real-MNIST digit-3 proxy** (784-dim
pixels, the same one [L2's own stage 3](l2-array-layer.md#measurement-plan-and-result-stage-3)
reused) - only the *original* tuned-XOR sweep (momentum's very first measurement) was a toy 2D
target. This stage's real step up is speed (vectorized, not per-node) and seed count, not
dimensionality - noted here rather than silently inherited.

**Part A - wall-clock, one mini-batch training step** (`forward_batch` + `compute_output_delta_batch`
+ `accumulate_gradient_batch` + `apply_accumulated_gradient`, `dimension=784, hidden=10`, per-node
vs numpy `MomentumArrayLayer` - no Rust `MomentumRustArrayLayer` yet at measurement time, that's
stage 5, independent of this measurement), median of 30 timed reps per config:

| batch size | per-node (us/example) | numpy (us/example) | per-node/numpy |
|---|---|---|---|
| 1 | 4702.58 | 156.19 | 30.1x |
| 8 | 2786.87 | 10.91 | 255.5x |
| 32 | 2999.89 | 7.71 | 389.3x |
| 128 | 3019.16 | 5.71 | 528.5x |
| 512 | 3030.12 | 4.23 | 716.8x |

**Yes, decisively**: 30x-717x faster per example than the per-node path, widening with batch
size - the same pattern (and a comparable range) as every other sibling in this round.

**Part B - the learning-rate-vs-batch-size follow-up, re-run through the array line at 3x the
seed count.** Necessary, honestly-flagged adaptation (the same inherited gap L2's own stage 3
flagged): run through `MomentumVectorizedMultiClassBackpropClassifierNetwork`'s `class_count=2`
one-vs-rest line, not the original single-sigmoid-output per-node scenario bit-for-bit. Same
400-example real-MNIST digit-3 proxy construction (320 train / 80 test,
freshly built - not the identical instance the original follow-up used, matching this codebase's
own reproducibility convention), `[16]` hidden, 5 epochs, `learning_rate = 0.5 * batch_size`
(the linear scaling rule), 30 seeds (vs. the original's 10):

| momentum | batch_size=1 (lr=0.5) | batch_size=8 (lr=4.0) | batch_size=32 (lr=16.0) | batch_size=128 (lr=64.0) |
|---|---|---|---|---|
| 0.0 | 87.67% ± 1.98% | 88.12% ± 2.13% | 89.12% ± 3.23% | 60.75% ± 12.86% |
| 0.3 | 88.12% ± 2.28% | 88.54% ± 1.77% | 88.83% ± 6.88% | 62.58% ± 14.72% |
| 0.5 | 88.96% ± 1.86% | 89.08% ± 1.93% | 87.67% ± 6.89% | 62.04% ± 14.32% |
| 0.7 | 89.38% ± 1.96% | 89.00% ± 2.61% | 84.96% ± 9.95% | 61.71% ± 14.65% |
| 0.9 | 82.04% ± 7.37% | 79.75% ± 12.53% | 79.46% ± 13.75% | 62.42% ± 15.72% |

**Both of the original follow-up's qualitative findings reproduce cleanly at 3x the seed
count.** At `batch_size=32`, `momentum=0.0` is at least as good as every other coefficient
(89.12% ± 3.23%, tightest variance in that column) and accuracy degrades, with rising variance,
as momentum increases past 0.3 - the same "properly-scaled learning rate leaves momentum
nothing to rescue" pattern the original found. `batch_size=128`'s `lr=64.0` collapses
completely and near-identically regardless of momentum (60.75%-62.58%, all with double-digit
stdevs) - the same training-instability regime the original flagged, unrelated to momentum
itself. `momentum=0.9` is the worst and highest-variance coefficient at every batch size,
including `batch_size=1`/`8` where the others cluster tightly (87.67%-89.38%) - actively
destabilizing, not neutral, matching the original's own strongest finding against adopting it.

**Decision: unchanged, on stronger evidence.** `MomentumVectorizedMultiClassBackpropClassifierNetwork`/
`MomentumRustArrayMultiClassBackpropClassifierNetwork` are real, tested capabilities, not adopted
as a default - momentum still doesn't help once the learning-rate confound is controlled for,
and `momentum=0.9` remains actively harmful, now confirmed at 3x the seed count through a
genuinely different (vectorized, matmul-based) execution path.

## risks and open questions

- **A reconfirmed null is a legitimate outcome, not a wasted effort** - per the updated mandate
  behind this workplan, the value here is a cheap, better-powered check, not a guaranteed new
  capability. Flagged explicitly so this isn't mistaken for a claim that momentum will turn out
  to help.
- **The single-output array-based line gap** - inherited from [an array-based Adam
  sibling](adam-array-layer.md#risks-and-open-questions), unresolved there too; this workplan
  doesn't attempt to close it, only works within the existing multiclass array line's scope.
- **Floating-point summation order** - matmul-based batched gradients sum in a different order
  than the per-node path's per-weight Python loop; Adam's own port found this immaterial to its
  own qualitative result ([the Rust core's RNG exception](rust-array-core.md#the-rng-exception)
  aside, which doesn't apply here since momentum's update itself is deterministic) - worth
  reconfirming for momentum specifically, not assumed to transfer.

## delivery stages (each its own PR, per this repo's practice)

1. ✅ This design document.
2. ✅ `MomentumArrayLayer` + `MomentumVectorizedMultiClassBackpropClassifierNetwork` + the
   parity-check tests above (numpy only).
3. ✅ The wall-clock/accuracy measurement - see "measurement plan and result" above: 30x-717x
   faster per example than the per-node path; the reconfirmed null holds at 3x the seed count.
4. ✅ Docs closeout: `structure.md`'s possible-next-steps entry updated to reflect the actual
   result - every stage of this workplan is now done.
5. ✅ `MomentumRustArrayLayer` + `MomentumRustArrayMultiClassBackpropClassifierNetwork` + the fused
   Rust op + parity-check tests at every tier - independent of stage 3's measurement, the same way
   [an array-based Adam sibling](adam-array-layer.md#the-rust-matmul-backed-counterpart-stage-5)'s
   own stage 5 was.
