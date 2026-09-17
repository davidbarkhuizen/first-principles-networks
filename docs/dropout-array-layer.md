# an array-based (Rust-matmul-backed) dropout sibling

[← back to README](../README.md)

**Status: all stages done.** Written up front as a design/measurement plan before any of it
existed, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based
Adam sibling](adam-array-layer.md) for precedent) - updated here with stage-by-stage status notes
as it's executed. Both `DropoutArrayLayer`/`DropoutVectorizedMultiClassBackpropClassifierNetwork`
(numpy) and `DropoutRustArrayLayer`/`DropoutRustArrayMultiClassBackpropClassifierNetwork`
(Rust-matmul-backed) are built and correctness-tested - the Rust stage was **not** gated on
stage 3's wall-clock finding the way "delivery stages" below originally proposed: both backends
were built together, deliberately, since the point of this workplan is affording the sweep at
all, not re-litigating whether it's worth affording. The sweep itself - closing
[dropout's own flagged measurement gap](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped) -
has now run, both the original scale and the conditional escalation: **a reconfirmed null**, the
same qualitative finding as L2's own measurement, now at real statistical power (30 seeds, both
scales) the abandoned per-node attempt could never afford - see "measurement plan and result
(stage 3)" below for the full numbers.

## why this, and why now

This is the workplan directly motivated by this session's own concrete failure:
`DropoutBackpropClassifierNetwork` (`dropout_layer.py`) is built and correctness-tested, but its
own stage-1 overfitting-gap measurement was attempted, observed running ~5x slower than serial
calibration predicted, and stopped before completion on track for 40+ minutes -
[dropout](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped) already
names the fix: "building a vectorized/Rust-matmul-backed `DropoutLayer` sibling first, mirroring
Adam's own precedent, to make a seeded sweep practical." This workplan is that fix, written up
front the same way every other array-layer workplan in this codebase has been.

## scope

`DropoutArrayLayer` (numpy) and `DropoutRustArrayLayer` (Rust-matmul-backed), the same two-stage
precedent [an array-based Adam sibling](adam-array-layer.md) established. Hidden-layer-only,
matching `DropoutNode`'s own convention (dropout.md's "scope": "the same scope boundary every
prior weight-update-rule/activation sibling used"). Scoped to `MultiClassBackpropClassifierNetwork`'s
array line (`DropoutVectorizedMultiClassBackpropClassifierNetwork`/
`DropoutRustArrayMultiClassBackpropClassifierNetwork`), the same still-open single-output-line gap
every sibling in this round inherits. This does mean the vectorized measurement below reproduces
[dropout's/L2's own binary digit-3-vs-rest
proxy](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it)
as an equivalent 2-class (`class_count=2`) one-vs-rest multiclass problem rather than a literal
single-output network - not bit-comparable to the per-node numbers, the same category of caveat
[an array-based Adam sibling](adam-array-layer.md#measurement-plan-and-result-stage-3)'s own
"not bit-comparable... but qualitatively the same shape" note already accepted for its own port.

## design: the one real exception to "purely additive" carries over, in array form

[Dropout's own per-node design](dropout.md#design-a-real-minimal-exception-to-purely-additive-not-glossed-over)
already flagged that dropout needs a call-scoped train/eval-mode distinction no prior sibling
needed - that requirement doesn't go away at the array level, and the same two subtleties that
design document found by testing, not by inspection, need re-deriving here rather than assumed to
carry over unchanged.

`DropoutArrayLayer(ArrayLayer)` adds a `training: bool` attribute (default `False`, the same
safe-failure-mode default the per-node design chose) and overrides `forward`/`forward_batch` to
draw and apply an inverted-dropout mask, plus `compute_hidden_delta`/`compute_hidden_delta_batch`
to apply that same mask to the backward pass:

```python
class DropoutArrayLayer(ArrayLayer):
    def __init__(self, size: int, input_size: int, drop_probability: float) -> None:
        super().__init__(size, input_size)
        assert 0.0 <= drop_probability < 1.0
        self._drop_probability = drop_probability
        self._keep_probability = 1.0 - drop_probability
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        base = sigmoid(self.z)
        if self.training:
            self._mask = (np.random.random(self.size) >= self._drop_probability).astype(np.float64)
            self.a = base * self._mask / self._keep_probability
        else:
            self._mask = np.ones(self.size)
            self.a = base
        self._base_activation = base           # needed by compute_hidden_delta, mirroring
        self._was_training = self.training     # DropoutNode's own two forward-time snapshots
        return self.a

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.W.T @ next_layer.delta
        sigmoid_derivative = self._base_activation * (1.0 - self._base_activation)
        scale = (self._mask / self._keep_probability) if self._was_training else 1.0
        self.delta = downstream * sigmoid_derivative * scale
```

`forward_batch`/`compute_hidden_delta_batch` mirror this with one independent mask row per
example (`np.random.random((batch_size, self.size))`), matching the per-node design's own
per-example stochastic draw exactly rather than one shared mask for the whole batch - dropout's
whole point is a fresh, independent draw per forward pass, and a batched forward pass is still
`batch_size` independent forward passes from dropout's perspective.

Both subtleties [the per-node design](dropout.md#design-a-real-minimal-exception-to-purely-additive-not-glossed-over)
found by testing carry over unchanged: the backward pass must use `_base_activation` (pre-mask
sigmoid), not `self.a` (post-mask), for the derivative term, and it must read a forward-time
snapshot of `training`/the mask (`_was_training`, `_mask`), not the live attribute, since
`set_training_mode`-equivalent brackets on the array network toggle `training` back to `False`
before backward runs. A `DropoutArrayLayer.set_training_mode(training)` method mirrors
`DropoutLayer`'s own, called from a `VectorizedMultiClassBackpropClassifierNetwork.learn`/
`learn_batch`-level try/finally bracket the same shape the per-node `learn()` uses.

`DropoutRustArrayLayer(RustArrayLayer)` needs a source of per-call randomness in Rust, not just a
fused arithmetic op - a genuinely new category of primitive for [the Rust array
core](rust-array-core.md), which today has no RNG surface at all (`randomize()`'s own random
draws happen in Python, via `np.random.uniform`, before ever touching the Rust core - see [the
numpy interface subset](numpy-interface-subset.md)'s own `uniform random fill` row). Flagged as a
real, non-trivial scope addition, not a mechanical port.

## correctness validation

The one sibling in this round with a genuine RNG-parity caveat, the same category [an array-based
Adam sibling](adam-array-layer.md#risks-and-open-questions) already accepted for its own
batch-size-robustness measurement, but sharper here since dropout's mask *is* the mechanism under
test, not incidental to it:

- `test_dropout_array_layer.py`: `forward`/`compute_hidden_delta` at `training=False` checked
  directly against `ArrayLayer`'s own plain sigmoid (must match exactly, no rescale) - the array
  analogue of [the per-node design](dropout.md#correctness-validation)'s own inference-mode
  check. At `training=True`, force a known mask via `unittest.mock.patch("numpy.random.random",
  ...)` (the array-world equivalent of the per-node test's own `random.random` patch), then check
  both the forced-kept and forced-dropped cases against the hand-derived chain-rule formula -
  never against a live, unforced numpy RNG draw, since matching an *exact* mask across numpy's and
  Python's own `random` module RNG streams is neither possible nor the property being tested.
- `test_dropout_vectorized_multiclass_backprop_model.py`: whole-network tests via the shared
  `tests/helpers.py` convention - snapshot/restore round-trip (no new snapshot/restore surface:
  `_mask`/`_base_activation`/`training` are per-forward-pass-scoped, the same category
  `_activation`/`delta` already are), plus the same exactly-reproducible-at-inference test
  [dropout's own per-node suite](dropout.md#correctness-validation) added. Not symmetry-breaking:
  no array-backed sibling test file in this codebase has that test (`assert_randomize_breaks_symmetry`
  is shaped around per-node `.nodes`, which array-backed layers don't have) - built instead was a
  genuine cross-implementation parity check no other array sibling's test suite needed to add
  separately, `matching_dropout_array_backprop_networks` (`tests/helpers.py`): dropout is a
  deterministic no-op at eval mode on both sides (training defaults to `False`, and
  `predict_probabilities`/`classify_state` never toggle it), so `predict_probabilities`/
  `classify_state` are checked against a per-node reference
  (`DropoutMultiClassBackpropClassifierNetwork`) across a random sweep, the same shape every
  other array sibling's own eval-mode parity test uses - a `learn()`-step parity check isn't
  attempted, for the same independent-RNG-streams reason noted above.
- `test_dropout_rust_array_layer.py` + `rust/indrajala_ml_array/tests/test_dropout_rng.py` +
  `rust/indrajala_ml_array/tests/test_dropout_fused_layer_ops.py`: the Rust core's RNG primitive,
  parity-tested against numpy's own mask distribution statistically (mean keep-rate over many
  draws within tolerance of `1 - drop_probability`, not a
  literal per-draw match) - a genuinely different validation shape than every other Rust parity
  test in this codebase, which are all bit-close deterministic checks; flagged explicitly rather
  than silently reusing a deterministic-parity test template that doesn't fit.

## measurement plan and result (stage 3)

The whole point of this workplan - closing [dropout's own flagged measurement
gap](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped). Run against
`DropoutVectorizedMultiClassBackpropClassifierNetwork` (numpy), not the Rust backend: numpy's own
RNG is seedable (`np.random.seed`), giving genuinely reproducible per-seed trials, while the Rust
core's hand-rolled generator explicitly can never be (see "design" above and
[rust-array-core.md](rust-array-core.md)'s "the RNG exception") - reproducibility matters more
for an actual measurement than the Rust backend's own extra speed, and the two backends are
already proven equivalent at eval mode by this workplan's own parity tests, so numpy's conclusion
about whether dropout helps transfers.

**Wall-clock, first**: a real, timed calibration run (`[16]` hidden, 20 epochs,
`drop_probability=0.0`, one seed) took 0.85s, projecting the original 50-run sweep shape at well
under a minute - confirmed directly, not estimated, by then actually running it.

**The proxy**: a fixed 400-example real-MNIST digit-3-vs-rest set (200 positive + 200 negative,
stratified evenly across the other 9 digits - the "200/200 balanced, 80/20 split" shape
[structure](structure.md#possible-next-steps)'s own audit item names), 80/20 split (320 train /
80 test), reframed as `class_count=2` one-vs-rest per "scope" above - not the same proxy instance
[L2's own array-stage escalation](l2-array-layer.md#measurement-plan-and-result-stage-3) built
(independently constructed, the same posture every reproducibility check in this codebase takes),
so the numbers below aren't bit-comparable to that document's own, only qualitatively so.

**Stage 1 - reproduction** (`[16]` hidden, 20 epochs - dropout's own originally-planned scale,
`learning_rate=0.5`, 30 seeds, 150 runs, 2.24 minutes total):

| drop_probability | train acc | test acc | gap (pts) |
|---|---|---|---|
| 0.0 | 98.55% ± 0.25% | 88.88% ± 1.18% | 9.68 |
| 0.1 | 98.62% ± 0.29% | 88.29% ± 1.19% | 10.33 |
| 0.2 | 98.70% ± 0.32% | 88.25% ± 1.43% | 10.45 |
| 0.3 | 98.44% ± 0.33% | 89.00% ± 1.04% | 9.44 |
| 0.5 | 97.99% ± 0.52% | 89.25% ± 1.98% | 8.74 |

At matching scale (50 runs: 10 seeds x 5 configs, the abandoned per-node sweep's own shape), the
array path measured ~45 seconds against the per-node path's own projected 40+ minutes - **over
53x**, landing in the same 53x-845x range
[L2's](l2-array-layer.md#measurement-plan-and-result-stage-3) own wall-clock table measured.

A clean null: every `drop_probability`'s test accuracy (88.25%-89.25%) sits well within one
seed-to-seed standard deviation of the unregularized baseline's own 88.88% ± 1.18% - no
meaningful improvement over `drop_probability=0.0`, the same shape L2's own reproduction found on
this identical scale.

**Stage 2 - the overfitting-headroom escalation** (`[128]` hidden, 60 epochs - substantially
larger and longer, the same escalation shape
[L2's own array-stage escalation](l2-array-layer.md#measurement-plan-and-result-stage-3) used,
deliberately chosen to try to induce more overfitting before concluding dropout doesn't help;
`learning_rate=0.5`, 30 seeds, 150 runs, 21.33 minutes total):

| drop_probability | train acc | test acc | gap (pts) |
|---|---|---|---|
| 0.0 | 98.50% ± 0.19% | 88.58% ± 1.47% | 9.92 |
| 0.1 | 98.57% ± 0.25% | 88.33% ± 1.22% | 10.24 |
| 0.2 | 98.55% ± 0.35% | 88.37% ± 1.26% | 10.18 |
| 0.3 | 98.46% ± 0.29% | 88.88% ± 0.75% | 9.58 |
| 0.5 | 98.53% ± 0.39% | 88.79% ± 1.39% | 9.74 |

**A real, honestly-flagged difference from L2's own escalation, not glossed over**: L2's escalation
widened its own train-test gap (10.11 -> 13.03 points), confirming that scale was genuinely more
overfitting-prone before asking whether L2 exploits the extra headroom. This escalation did
**not** - the gap here (9.6-10.2 points) is statistically indistinguishable from stage 1's own
(8.7-10.5 points), and train accuracy stayed capped at ~98.5% regardless of hidden-layer width or
epoch count. This proxy's own train-accuracy ceiling on this particular fixed 320-example split
appears to be reached already at `[16]`/20 epochs (plausibly intrinsic label noise/overlap in a
small, fixed digit-3-vs-rest split, not something more capacity or more training resolves) -
`[128]`/60 epochs never got the extra room to work with that L2's own escalation did. Flagged
explicitly, not hidden: the escalation's own precondition (confirm the gap widens first) wasn't
met this time, unlike every other sibling's own escalation in this round.

**Decision: a reconfirmed null, now at real statistical power (30 seeds, both scales) the
abandoned per-node attempt could never afford, closing [dropout's own flagged measurement
gap](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped) - dropout closes
no held-out-accuracy gap it isn't already given room to close, the same qualitative finding as
L2's own measurement, on both the original and the escalated scale.** Not adopted as a default;
`DropoutBackpropClassifierNetwork`/`DropoutVectorizedMultiClassBackpropClassifierNetwork`/
`DropoutRustArrayMultiClassBackpropClassifierNetwork` remain real, tested capabilities for a
genuinely more overfitting-prone future scenario than this proxy turned out to provide.

## risks and open questions

- **The array-world proxy isn't bit-comparable to the per-node one** - flagged under "scope"
  above; the qualitative shape (does any `drop_probability` improve held-out accuracy over the
  unregularized baseline) is the actual question, not exact numeric parity with a measurement
  that was never even completed on the per-node path to compare against.
- **The Rust core needed a genuinely new category of primitive** (RNG) - flagged under "design"
  above, and confirmed not to be a mechanical fused-arithmetic port the way momentum/L2/Adam's
  Rust stages were: `random.rs`'s `draw_bernoulli_mask`/`bernoulli_mask` (a hand-rolled
  xorshift128+ draw, the same generator `uniform()` already uses, reused rather than a second
  PRNG built from scratch) - resolved, built, and tested (`rust/indrajala_ml_array/tests/test_dropout_rng.py`).
  Built unconditionally rather than gated on stage 3's wall-clock finding, per explicit
  direction: the Rust stage's value (affording the sweep) doesn't depend on how slow the numpy
  stage turns out to be.
- **A reconfirmed null is a legitimate outcome** - same framing as every sibling in this round;
  the value is finally getting a real answer, not a guaranteed regularization win. Confirmed:
  see "measurement plan and result (stage 3)" above.
- **The stage-2 escalation didn't actually widen the train-test gap, unlike L2's own** - flagged
  explicitly above rather than glossed over; this proxy's train accuracy appears capped around
  98.5% on this fixed 320-example split regardless of hidden-layer width or epoch count, so the
  escalation didn't get the extra overfitting headroom L2's own escalation did. Worth
  re-attempting with a genuinely different lever (more training examples held fixed at a smaller
  hidden layer, rather than a larger hidden layer/more epochs) if this question is ever revisited
  - not attempted here, since the stage-1 result alone already answers the actual question this
  workplan exists to close.

## delivery stages (each its own PR, per this repo's practice)

1. ✅ This design document.
2. ✅ `DropoutArrayLayer` + `DropoutVectorizedMultiClassBackpropClassifierNetwork` + the
   parity-check tests above (numpy only) - `tests/test_dropout_array_layer.py` (12 tests),
   `tests/test_dropout_vectorized_multiclass_backprop_model.py` (13 tests).
3. ✅ The wall-clock check, then the stage-1 sweep, then - triggered by stage 1's own null result,
   exactly the conditional escalation discipline this document's "measurement plan" originally
   set - the stage-2 escalation, run right after since stage 1 finished in 2.24 minutes, cheap
   enough not to warrant a separate pause to decide. Closes [dropout's own flagged
   gap](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped) - see
   "measurement plan and result (stage 3)" above for the full numbers, including the
   honestly-flagged difference from L2's own escalation (this one didn't widen the train-test
   gap).
4. ✅ Docs closeout: this document's and [dropout.md](dropout.md)'s own status updated to reflect
   the actual measurement result (a reconfirmed null, both scales).
5. ✅ `DropoutRustArrayLayer` + `DropoutRustArrayMultiClassBackpropClassifierNetwork`, plus the
   Rust core's new `bernoulli_mask`/`draw_bernoulli_mask` RNG primitive (`random.rs`) and fused
   ops (`layer_dropout_forward`/`layer_dropout_forward_batch`/`layer_dropout_hidden_delta`/
   `layer_dropout_hidden_delta_batch`, `fused.rs`) - `tests/test_dropout_rust_array_layer.py`
   (9 tests), `tests/test_dropout_rust_array_multiclass_backprop_model.py` (13 tests),
   `rust/indrajala_ml_array/tests/test_dropout_rng.py` (6 tests),
   `rust/indrajala_ml_array/tests/test_dropout_fused_layer_ops.py` (153 tests, 5 seed-parametrized
   functions x 30 seeds each plus 3 fixed cases). Built **unconditionally**, not gated on stage
   3's wall-clock finding as originally
   proposed here - explicit direction overrode the conditional framing, since affording the
   sweep is this workplan's whole point regardless of how the numpy stage alone measures.
