# dropout: stochastic hidden-unit regularization

[← back to README](../README.md)

**Status: stage 2 (the per-node capability) done; the overfitting-gap measurement itself now run
against an array-based dropout sibling, not the per-node path this document's own stages 3-4
originally targeted.** That array-based sibling (both numpy and Rust-matmul-backed) was built
specifically to make the measurement affordable - see this document's own "the measurement gap"
below for the full story - and [research and
analysis](research-backprop-siblings.md#an-array-based-dropout-sibling-another-reconfirmed-null-now-at-real-power)
now has the real numbers: **a reconfirmed null**, both at the original scale and a deliberately
more overfitting-prone escalation, at far higher statistical power (30 seeds) than the per-node
path's abandoned attempt could ever afford. Written up front as a design/measurement plan before
any of it existed, per this repo's own established practice of writing a plan down before
implementation - updated here with stage-by-stage status notes as it's executed.
`DropoutBackpropClassifierNetwork` is built and correctness-tested, but its actual regularization
effect on this codebase's own benchmark is a known, flagged gap, not a silently-dropped one - see
"the measurement gap" at the end of "measurement plan" below for why, and what would need to be
true to close it.

## why this, and why now

The second item in [structure](structure.md#possible-next-steps)'s "new model primitives" tier:
`L2RegularizedBackpropClassifierNetwork` is the only regularization this codebase has, and its
own measurement ([L2 weight
regularization](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it))
found it closes the train-test accuracy gap but never improves held-out accuracy above the
unregularized baseline, plausibly because a single 16-node hidden layer on 320 training examples
doesn't overfit severely enough for a weight-magnitude penalty to have room to help. Dropout
(Srivastava et al., 2014) is worth testing independently rather than assumed to land the same way:
it's a structurally different mechanism (stochastic per-unit deactivation at the *activation*,
applied fresh every forward pass, rather than a fixed penalty term added to every weight's
*gradient*) - but the same overfitting-headroom risk L2 hit applies here too, and isn't assumed
away just because the mechanism differs (see "risks and open questions" below).

## scope

`DropoutBackpropClassifierNetwork`, a sibling of `BackpropClassifierNetwork` only (single-output,
binary) - the same scope boundary every prior weight-update-rule/activation sibling used
(`MomentumBackpropClassifierNetwork`, `L2RegularizedBackpropClassifierNetwork`,
`ReLUBackpropClassifierNetwork`). Hidden-layer-only, matching `ReLUNode`'s own precedent
(convention, not mathematical necessity - dropping units feeding a sigmoid/softmax output is
possible in principle, just not standard practice and not what's built here). No composition with
any other sibling (momentum/L2/Adam/ReLU) - out of scope, and no existing sibling in this codebase
composes with another either, so this introduces no new gap.

## design: a real, minimal exception to "purely additive," not glossed over

Every prior sibling in this family (momentum, L2, Adam, ReLU, softmax) is purely additive: a new
factory-built `BackpropNode`/`BackpropLayer` subclass overriding `apply_accumulated_gradient` or
`forward`, wired in via `hidden_layer_cls`/`output_layer_cls`, with zero changes to
`BackpropNode`/`BackpropLayer`/`BackpropNetworkBase`/`BackpropClassifierNetwork` themselves.
Dropout can't quite manage that, and this is worth being explicit about rather than discovering
partway through implementation: dropout must be *active* during a training-time forward pass
(`learn`/`learn_batch`) and *inactive* during an inference-time one
(`predict_probability`/`classify_state`) - a distinction no existing sibling needs, since momentum/
L2/Adam only ever touch `apply_accumulated_gradient` (called only during training already) and
ReLU's/softmax's forward-pass changes are unconditional (always active, train or not). Confirmed
directly: nothing in this codebase today has any train/eval-mode concept, and the two call kinds
are genuinely interleaved on the same network object within one training run, not phase-separated
- `train.py`'s own training loops call `student.classify_state(...)` (via
`_training_accuracy`/`class_balanced_disagreement_rate`) after every single `learn()`/`learn_batch()`
step and at every epoch boundary, on the same `student` mid-training. Any train/eval toggle must
therefore be call-scoped (on for the duration of one `learn`/`learn_batch` call, off again
immediately after), not a one-time lifecycle switch.

**Chosen mechanism**: a plain mutable `training: bool` attribute on each `DropoutNode` (default
`False` - the safe failure mode if anything forgets to toggle it is "behaves like an ordinary
node," not "silently drops units at inference"), set via a new `set_training_mode(training: bool)`
method - a no-op by default on `BackpropLayer` (so every non-dropout layer is unaffected and needs
no change at all), overridden on `DropoutLayer` to propagate the flag to each of its nodes. A
matching `BackpropNetworkBase._set_training_mode(training)` loops `self.trainable_layers`.
`BackpropClassifierNetwork.learn()` and `BackpropNetworkBase._learn_batch()` (shared by every
`learn_batch`-supporting sibling, including `MultiClassBackpropClassifierNetwork`) bracket their
own `self._forward(state)` call in a `set_training_mode(True)` / `try` / `finally:
set_training_mode(False)` - the `finally` guarantees the flag can't get stuck `True` if
`_backward`/`_apply_gradients` ever raises. `predict_probability`/`classify_state` need no change
at all - the flag already defaults to `False`.

**Alternative considered and rejected**: threading an explicit `training: bool` parameter through
`forward()`/`_forward_outputs()`'s own signatures instead of a mutable flag. Rejected because it
would touch every layer/node class's `forward()` signature (even ones with nothing to do with
dropout, just to keep the call chain type-consistent) for a wider blast radius than the flag
approach, which only adds one new no-op-by-default method and a small try/finally at the two
call sites that need it. The tradeoff, acknowledged rather than hidden: a mutable instance
attribute is a new kind of state for this class family - every existing per-forward-pass value
(`_activation`, `delta`) is overwritten atomically inside `forward()`/`compute_*_delta()` itself,
never toggled by a caller several stack frames up. Worth re-examining if a future sibling needs
the same mechanism and the shared-flag approach starts feeling brittle.

**The factory itself** (`indrajala_ml/model/dropout_layer.py`, mirroring `momentum_layer.py`/
`l2_regularization_layer.py`/`relu_layer.py`'s shape): `make_dropout_node_cls(drop_probability)`
closes over `drop_probability` and `keep_probability = 1.0 - drop_probability`:

```python
class DropoutNode(BackpropNode):
    def __init__(self, input_nodes, input_node_weights=None, bias=0.0):
        super().__init__(input_nodes, input_node_weights, bias)
        self.training = False
        self._kept = True
        self._base_activation = 0.0  # pre-dropout sigmoid(z()), cached for compute_hidden_delta
        self._was_training = False  # a forward-time snapshot of self.training - see below

    def forward(self) -> float:
        self._base_activation = sigmoid(self.z())
        self._was_training = self.training
        if self.training:
            self._kept = random.random() >= drop_probability
            # inverted dropout: rescale kept units by 1/keep_probability during training, so
            # their expected contribution matches the full (all-units-kept) network - the
            # standard convention specifically so no rescaling is needed at inference time
            self._activation = (self._base_activation / keep_probability) if self._kept else 0.0
        else:
            self._kept = True
            self._activation = self._base_activation
        return self._activation

    def compute_hidden_delta(self, next_layer_nodes, own_index):
        if not self._kept:
            self.delta = 0.0  # matches ReLUNode's own dead-unit precedent: zero delta, zero
            return            # gradient, incoming weights untouched this step
        downstream = sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)
        sigmoid_derivative = self._base_activation * (1.0 - self._base_activation)
        scale = (1.0 / keep_probability) if self._was_training else 1.0
        self.delta = downstream * sigmoid_derivative * scale
```

**A subtlety worth getting right, not glossed over**: the backward formula above must use
`_base_activation` (the *pre*-dropout-scaling sigmoid value) for the sigmoid-derivative term, not
`self.value()` (the *post*-scaling `_activation` a downstream node reads) the way
`BackpropNode.compute_hidden_delta`'s own `a * (1 - a)` does. Since `a = base / keep_probability`
when kept, `a * (1 - a) != base * (1 - base) / keep_probability` in general (the `keep_probability`
term appears squared on one side and linearly on the other) - naively delegating to
`super().compute_hidden_delta()` here would silently compute the wrong gradient. The correct
chain rule (`d(base * mask/keep_probability)/dz = (mask/keep_probability) * base * (1-base)`)
gives exactly the formula above: downstream-delta times the *ordinary* sigmoid derivative on the
*unscaled* activation, times the same `1/keep_probability` rescale forward used.

**A second subtlety, found by the hand-computed regression test below, not eyeballed as
"probably fine" and missed**: `compute_hidden_delta`'s rescale can't read the *live*
`self.training` at backward time - `BackpropClassifierNetwork.learn()` only brackets
`set_training_mode(True)` around the `_forward()` call itself (see below), so by the time
`_backward()` runs, `self.training` is already back to `False` again, and a naive
`scale = (1.0 / keep_probability) if self.training else 1.0` silently applies the wrong
(unscaled) derivative - a real bug the whole-network hand-derived fixture in
`test_dropout_backprop_model.py` caught directly (the computed weight update was exactly half
the independently hand-derived expected value, `1/keep_probability` missing). Fixed by taking
`self._was_training = self.training` as a forward-time snapshot, the same category of
already-established pattern `_kept`/`_base_activation` already are - `compute_hidden_delta` now
reads that snapshot, not the live, possibly-already-reverted flag, which also makes the result
correct regardless of exactly how wide or narrow a caller's `set_training_mode` bracket is.

**A third thing found only by running the full suite, not by design review**: `ConvLayer`
(`conv_layer.py`) is deliberately *not* a `BackpropLayer` subclass (composition, not inheritance -
see its own docstring), so it doesn't automatically inherit `BackpropLayer`'s new no-op
`set_training_mode` the way every other layer type does - `BackpropNetworkBase._set_training_mode`
calling it unconditionally on every `trainable_layer` broke `ConvMultiClassBackpropClassifierNetwork`
outright (`AttributeError`) until `ConvLayer` got the identical no-op added directly, extending
the duck-typed surface its own docstring already enumerates. The one other place this sibling
family's "purely additive" precedent had a real, previously-invisible edge - not just the
train/eval-mode mechanism flagged above.

`make_dropout_layer_cls(drop_probability)` mirrors every other factory's layer counterpart
(`_node_cls = DropoutNode`), plus the one real addition: `DropoutLayer.set_training_mode(training)`
loops `for node in self.nodes: node.training = training`.

`DropoutBackpropClassifierNetwork(BackpropClassifierNetwork)` sets only `hidden_layer_cls` (not
`output_layer_cls`, matching `ReLUBackpropClassifierNetwork`'s own hidden-only shape) to
`make_dropout_layer_cls(drop_probability)` in `__init__`, before `super().__init__()` runs.

## hyperparameters

`drop_probability` is a **required** argument, no default - the same posture `momentum`/
`l2_lambda` take (not Adam's `beta1`/`beta2`/`epsilon`, which got defaults specifically because
they're close to fixed algorithmic constants in real-world use): this codebase's own measurements
haven't found a `drop_probability` this project has an opinion on, and literature-standard values
(often 0.5 for hidden layers) are exactly the kind of tunable-with-real-sensitivity coefficient
momentum/L2's own docstrings argue shouldn't get a silent default. `assert 0.0 <= drop_probability
< 1.0` - `drop_probability=1.0` would make `keep_probability=0`, both a division-by-zero in the
inverted-dropout rescale and a hidden layer that deterministically zeroes every unit, useless by
construction.

## correctness validation

Same two-tier convention as every prior sibling, plus a new axis no prior sibling's test suite
exercises:

- `test_dropout_layer.py`: hand-computed, single/few-node fixtures in the shape
  `test_relu_layer.py` uses (fixed weights/bias/input, worked-out arithmetic in the test's own
  comments) - `forward()` at `training=False` (must exactly match a plain `BackpropNode`'s
  sigmoid, no rescale), `forward()` at `training=True` with a forced kept outcome (activation
  rescaled by `1/keep_probability`) and a forced dropped outcome (activation exactly `0.0`), and
  `compute_hidden_delta()` under both outcomes - checked against the hand-derived chain-rule
  formula above, not re-derived from the implementation under test. Since the mask draw uses this
  codebase's shared global `random` module (the same one `fan_in_aware_weights_and_bias`/
  `randomize()` already use, not a new RNG mechanism), forcing a specific kept/dropped outcome
  means patching `random.random` (`unittest.mock.patch`) rather than relying on `random.seed`'s
  exact output sequence or adding a test-only production seam.
- `test_dropout_backprop_model.py`: whole-network tests via the shared `tests/helpers.py`
  convention - `assert_randomize_breaks_symmetry`, `assert_snapshot_restore_round_trip` (no new
  snapshot/restore surface needed: `_kept`/`_base_activation`/`training` are exactly the same
  category of per-forward-pass-scoped state `_activation`/`delta` already are, already excluded
  from `snapshot_state()`/`restore_state()`, which capture only weights/bias - confirmed directly
  that even Adam's own persistent `m`/`v`/`t` isn't captured there either, so dropout introduces no
  new gap). Plus the one genuinely new test this sibling's train/eval distinction demands:
  `predict_probability`/`classify_state` calls on a `DropoutBackpropClassifierNetwork` must be
  exactly reproducible run-to-run at the same weights (no dropout applied at inference, so no
  stochasticity at all) - a property no prior sibling's test suite needed to check, since none of
  them have any training-only behavior.

## measurement plan

Same target L2 was measured on, and for the same reason: a fixed, finite MNIST proxy so
overfitting is actually observable, rather than XOR's continuously-resampled target where it
isn't. Stage 1 deliberately reuses [L2's own exact
setup](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it)
(400-example real-MNIST digit-3 proxy, 80/20 split, fan-in-aware init, `[16]` hidden,
`learning_rate=0.5`) for a direct, apples-to-apples comparison against L2's own train/test/gap
numbers on identical footing - not just a similarly-shaped rerun. L2's own doc entry doesn't state
its exact epoch/seed count, so exact parity there isn't guaranteed; this measurement will state
its own explicitly when run (a seed count matching this codebase's usual ~10, and enough epochs to
reach a comparable unregularized training accuracy to L2's own ~98.5% baseline).

**Stage 1** - sweep `drop_probability` over a small grid (e.g. 0.0/0.1/0.2/0.3/0.5), report the
same train accuracy / test accuracy / train-test-gap table L2's own entry used, across seeds.

**Stage 2 (conditional)** - if stage 1 comes back a null the way L2's own measurement did (plausibly
for the identical reason: this scale doesn't overfit severely enough for any regularizer to have
room to help), a deliberately more overfitting-prone follow-up before concluding dropout doesn't
help either - a larger hidden layer relative to the same fixed 320 training examples, or
substantially more epochs than stage 1 uses, to induce visible overfitting first, then retest
dropout there. Not committed to up front - only run if stage 1's result actually calls for it, the
same conditional-escalation pattern the Adam optimizer investigation's own stage-3-gates-stage-4
structure used.

### the measurement gap - not run, deliberately, not silently dropped

Stage 1 was attempted, not skipped from the start: a real 50-run sweep (`drop_probability` in
0.0/0.1/0.2/0.3/0.5, 10 seeds each, `epochs=20` - calibrated so the unregularized baseline reaches
a comparable ~98% training accuracy to L2's own ~98.5%) was built and launched against an 8-way
`multiprocessing.Pool`. Measured directly, not guessed: the first round (8 jobs, one per worker)
took ~6 minutes wall-clock - roughly 5x slower than the serial calibration probe's own per-epoch
timing predicted, evidently real contention from running 8 heavy per-node training processes
concurrently. On track for 40+ minutes total, the run was stopped before completion rather than
paid for.

**Why this codebase has no cheaper path available today**: `DropoutBackpropClassifierNetwork` has
only the per-node `BackpropNode`/`BackpropLayer` path (see "scope" above) - no vectorized numpy or
Rust-matmul-backed counterpart exists. This isn't a dropout-specific gap: every sibling in this
family except Adam launches per-node-only, and Adam's own array-based port (see [research and
analysis](research-adam-optimizer.md#an-array-based-rust-matmul-backed-adam-sibling-the-wall-clock-win-the-accuracy-result-was-missing))
was a separate, later, independently-scoped 5-stage workplan, built only *after* Adam's per-node
measurements had already shown a real, substantial win worth chasing to production scale - not
part of Adam's own initial launch either.

**Decision: deliberately not run, and not immediately planned.** Reopening this measurement would
need either (a) tolerating the wall-clock cost outright (a single overnight/background run, not
attempted here), or (b) building a vectorized/Rust-matmul-backed `DropoutLayer` sibling first,
mirroring Adam's own precedent, to make a seeded sweep practical - neither committed to here.
Dropout's actual regularization effect on this codebase's own benchmark therefore remains
genuinely unmeasured, not merely unmeasured-yet-assumed-fine: the "risks and open questions"
section below carries this forward as an open item, not a resolved one.

**Update**: path (b) is now done, and the gap is closed. An array-based dropout sibling
(`DropoutArrayLayer`/`DropoutVectorizedMultiClassBackpropClassifierNetwork` and
`DropoutRustArrayLayer`/`DropoutRustArrayMultiClassBackpropClassifierNetwork`) was built, and its
own stage 3 re-ran the sweep against the numpy-backed array sibling: a real 150-run pass at the
original `[16]`-hidden/20-epoch scale (2.24 minutes total, versus this section's own 40+ minute
per-node projection for a smaller job - over 53x), then, since that came back null, a 150-run
conditional escalation at `[128]` hidden/60 epochs (21.33 minutes). **Result: a reconfirmed null at
both scales**, the same qualitative finding as
[L2's](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it)
own measurement - no `drop_probability` improved held-out accuracy above the unregularized
baseline, and unlike L2's own escalation, this one didn't even widen the train-test gap (flagged
explicitly, not glossed over - see [research and
analysis](research-backprop-siblings.md#an-array-based-dropout-sibling-another-reconfirmed-null-now-at-real-power)
for the full numbers and that honest caveat). Dropout's actual regularization effect on this
codebase's own benchmark is therefore no longer an open question - the "risks and open questions"
section below is updated accordingly.

## risks and open questions

- **The train/eval-mode mechanism is a real, minimal exception to this sibling family's
  "purely additive" precedent** - resolved and built (see "design" above: a no-op-by-default
  `set_training_mode` hook, needed on every layer type including `ConvLayer`'s
  composition-not-inheritance one, plus a try/finally at the two call sites that need it, plus a
  forward-time `_was_training` snapshot so backward never depends on how wide that bracket is).
  The mutable-flag approach is still a new kind of state for this class family, worth revisiting
  if a future sibling needs the same mechanism and it starts feeling brittle.
- **Whether dropout actually helps on this codebase's own benchmark at all** - **resolved: no.**
  The per-node path's own stage-1 sweep was attempted and stopped before completion (too slow -
  see "the measurement gap" above), but [an array-based dropout
  sibling](research-backprop-siblings.md#an-array-based-dropout-sibling-another-reconfirmed-null-now-at-real-power)
  reran it at real power (30 seeds, both the original scale and a conditional escalation): no
  `drop_probability` improved held-out accuracy above the unregularized baseline at either scale,
  the same finding L2's own measurement reached on this identical proxy.
- **Composability with other siblings** - explicitly out of scope; flagged, not chased.
- **Hidden-layer-only scope** - a convention carried over from ReLU's own precedent, not a
  mathematical necessity for dropout specifically (unlike ReLU's genuine output-range mismatch);
  revisit if an output-layer-dropout use case ever comes up.

## delivery stages (each its own PR, per this repo's practice)

1. ✅ This design document.
2. ✅ `dropout_layer.py` (`make_dropout_node_cls`/`make_dropout_layer_cls`) +
   `dropout_backprop_classifier_network.py` (`DropoutBackpropClassifierNetwork`) + the
   train/eval-mode plumbing (`BackpropLayer.set_training_mode`/
   `BackpropNetworkBase._set_training_mode`, wired into `BackpropClassifierNetwork.learn`/
   `BackpropNetworkBase._learn_batch`, plus the same no-op added to `ConvLayer` once the full
   suite caught it wasn't a `BackpropLayer` subclass) + the correctness-validation tests above
   (`test_dropout_layer.py`, `test_dropout_backprop_model.py`, 20 tests) - the actual capability,
   buildable and mergeable independent of any measurement result. Full suite passes (382 tests,
   up from 362). Caught a real bug during implementation - see the "design" section's own
   "second subtlety" above (the backward-pass rescale needed a forward-time snapshot of
   `training`, not a live re-read of it).
3. **Not run, deliberately** - the stage-1 overfitting-gap measurement (same setup as L2's own,
   for direct comparability) was attempted (a real 50-run sweep, launched and observed running
   ~5x slower than its own serial calibration predicted) and stopped before completion rather
   than paid for - see "the measurement gap" above for the full reasoning. A flagged, open gap,
   not a silently-skipped stage.
4. **Not applicable** - the stage-2 conditional follow-up can't be evaluated without stage 1's
   result.
5. ✅ Docs closeout (this update): `structure.md`'s possible-next-steps entry updated to reflect
   the actual state - capability built, measurement gap flagged, not a completed result.
