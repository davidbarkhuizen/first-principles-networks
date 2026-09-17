# an array-based (Rust-matmul-backed) dropout sibling

[← back to README](../README.md)

**Status: stages 2 and 5 (both array-based backends) done; the stage-1 overfitting-gap sweep
(stage 3) not yet run.** Written up front as a design/measurement plan before any of it existed,
per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based Adam
sibling](adam-array-layer.md) for precedent) - updated here with stage-by-stage status notes as
it's executed. Both `DropoutArrayLayer`/`DropoutVectorizedMultiClassBackpropClassifierNetwork`
(numpy) and `DropoutRustArrayLayer`/`DropoutRustArrayMultiClassBackpropClassifierNetwork`
(Rust-matmul-backed) are built and correctness-tested - the Rust stage was **not** gated on
stage 3's wall-clock finding the way "delivery stages" below originally proposed: both backends
were built together, deliberately, since the point of this workplan is affording the sweep at
all, not re-litigating whether it's worth affording. The sweep itself (closing
[dropout's own flagged measurement gap](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped))
remains a separate, not-yet-run follow-on - see "delivery stages" below.

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

## measurement plan

The whole point of this workplan - closing [dropout's own flagged measurement
gap](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped):

- **Wall-clock, first**: confirm the port is actually fast enough before re-running the sweep -
  the same fused-layer-operation benchmark used throughout this round, numpy vs. Rust (per
  [goals and strategy](goals-and-strategy.md#measurement-discipline-the-per-node-paths-two-jobs-and-the-one-it-doesnt-have),
  no fresh per-node timing run - the order of magnitude is already established), plus a direct
  apples-to-apples re-run of dropout's own abandoned 50-run sweep's *serial calibration probe*
  (the thing that predicted ~6 minutes for round one, then was blown through by ~5x) on the array
  path, to get a real, not estimated, expected total wall-clock before launching the full sweep.
- **The stage-1 overfitting-gap sweep itself**: `drop_probability` in 0.0/0.1/0.2/0.3/0.5, 10+
  seeds each (more now that it's affordable), on the 2-class array-equivalent proxy described
  under "scope" above, reporting the same train/test/gap table
  [L2's](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it)
  and [dropout's own](dropout.md#measurement-plan) entries used.
- **The stage-2 conditional escalation, now affordable too**: if stage 1 comes back null the way
  L2's did, the more-overfitting-prone follow-up [dropout's own workplan](dropout.md#measurement-plan)
  proposed but never ran (larger hidden layer or more epochs, to induce visible overfitting
  first) - genuinely conditional, only run if stage 1 calls for it, per that document's own
  escalation discipline.

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
  the value is finally getting a real answer, not a guaranteed regularization win.

## delivery stages (each its own PR, per this repo's practice)

1. ✅ This design document.
2. ✅ `DropoutArrayLayer` + `DropoutVectorizedMultiClassBackpropClassifierNetwork` + the
   parity-check tests above (numpy only) - `tests/test_dropout_array_layer.py` (12 tests),
   `tests/test_dropout_vectorized_multiclass_backprop_model.py` (13 tests).
3. **Not run yet** - the wall-clock check, then the stage-1 (and conditionally stage-2)
   overfitting-gap sweep - closing [dropout's own flagged
   gap](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped).
4. **Not run yet** - docs closeout: both this document's and [dropout.md](dropout.md)'s own
   status updated to reflect the actual measurement result once stage 3 runs.
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
