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
3. The wall-clock/accuracy measurement described above.
4. Docs closeout: `structure.md`'s possible-next-steps entry updated to reflect the actual result.
5. ✅ `MomentumRustArrayLayer` + `MomentumRustArrayMultiClassBackpropClassifierNetwork` + the fused
   Rust op + parity-check tests at every tier - independent of stage 3's measurement, the same way
   [an array-based Adam sibling](adam-array-layer.md#the-rust-matmul-backed-counterpart-stage-5)'s
   own stage 5 was.
