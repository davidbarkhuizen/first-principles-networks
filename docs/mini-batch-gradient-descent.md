# mini-batch gradient descent

[← back to README](../README.md)

Gradient-based students (`BackpropClassifierNetwork`, `MultiClassBackpropClassifierNetwork`, and
their momentum/L2 siblings) can train on batches, not just one example at a time -
`BackpropNode.accumulate_gradient`/`apply_accumulated_gradient` (plus the matching
`MomentumBackpropNode`/`L2RegularizedBackpropNode` overrides), `BackpropNetworkBase`'s
`_accumulate_gradients`/`_apply_accumulated_gradients`, `learn_batch` on both classifier networks
(and every sibling that inherits it unchanged - momentum, L2, ReLU, fan-in-aware, binary
cross-entropy, softmax), and `train.train_backprop_network_mini_batch`. `LinearClassifierNetwork`'s
discrete minimum-disturbance update rule is untouched - it has no gradient to batch.
`learn()`'s existing per-example path stays available, and is exactly the `batch_size=1` case in
behavior (proven by a required regression gate, below), not replaced.

## the architectural point: an accumulate/apply seam

Every trainable node's `apply_gradient` (`BackpropNode.apply_gradient`) used to do two things in
one call, back to back: compute this example's gradient from `self.delta` and each input node's
*current* `value()`, and immediately write the updated weight - no seam between "compute a
gradient" and "apply it." Mini-batching needs exactly that seam: run forward+backward for each
example in a batch, **accumulate** each one's gradient without touching the weights, then apply
the batch-averaged gradient once. `apply_gradient` is now a thin
`accumulate_gradient(); apply_accumulated_gradient(learning_rate, 1)` wrapper, so `learn()`'s call
sites needed no change. Three call sites needed the identical split, not just the plain node -
`MomentumBackpropNode.apply_gradient` (folds in `momentum * prev`) and
`L2RegularizedBackpropNode.apply_gradient` (adds `l2_lambda * weight` to the gradient) each fully
replace the base formula rather than adjusting its result.

`BackpropNetworkBase._apply_gradients` became `_accumulate_gradients()` (one pass calling
`accumulate_gradient()`, run once per example inside a batch) plus
`_apply_accumulated_gradients(learning_rate, batch_size)` (one pass calling
`apply_accumulated_gradient`, run once per batch). Each classifier network gained
`learn_batch(learning_rate, batch)`: `for example in batch: forward, backward, accumulate` then
one `_apply_accumulated_gradients` call. `indrajala_ml/train.py`'s existing
`train_linear_classifier_network` (shared with `LinearClassifierNetwork`, which has no batch
concept at all) was **not** retrofitted in place - `train_backprop_network_mini_batch` is its own
function for gradient-based students, with its own batch-construction chunking (a final
undersized batch is averaged over its own smaller size; `training_data` is reshuffled each epoch,
standard mini-batch SGD practice the existing per-example loop doesn't need).

## numerical and behavioral risks, and how they were resolved

- **Batch-size-1 parity was a required regression gate**, checked before anything else was
  trusted: a random-weight/random-input sweep comparing old `apply_gradient` against
  `accumulate_gradient` + `apply_accumulated_gradient(batch_size=1)`, for all three node variants
  (plain, momentum, L2) - it passed, and every pre-existing pinned test still passes unchanged.
- **L2's penalty term is not accumulated per example.** The weight itself doesn't change during a
  batch's forward/backward passes, so `l2_lambda * weight` is added once at apply time, not
  accumulated (and implicitly averaged) once per example in the batch - checked with a dedicated
  unit test, not just inferred from the algebra.
- **Averaging, not summing, the batch gradient** (`grad / batch_size`) keeps `learning_rate`'s
  meaning comparable to the existing tuned per-example values as `batch_size` varies - matching
  [the vectorized classes](structure.md#vectorized-array-based-classes)' own
  `grad_W = (delta_batch.T @ X_batch) / batch_size` formula.

This work does not by itself speed anything up: forward/backward still runs once per example, one
Python object at a time, in a pure-Python loop - only the timing of the weight write changes. Its
purpose was purely to unblock the momentum retest below, and to lay groundwork that's a
prerequisite for (but independent of) [the vectorized classes](structure.md#vectorized-array-based-classes)'
own batched design.

## the momentum retest: suggestive, but confounded

`MomentumBackpropClassifierNetwork` exists and is tested, but was measured
(see [research and analysis](research-and-analysis.md#momentum-measured-not-worth-adopting)) to
*hurt* at its canonical
coefficient across a learning-rate sweep, and land statistically indistinguishable from plain SGD
at finer coefficients - a flat null, not a win. The leading explanation: this codebase's
per-example *online* SGD produces gradients far noisier than the batch/mini-batch gradients
momentum's own literature is validated against, so accumulating velocity across noisy individual
steps amplifies noise instead of smoothing signal. This mini-batch infrastructure exists to
re-test that hypothesis under lower-noise batch gradients.

The retest (momentum coefficients 0.0-0.9 crossed with batch sizes 1/8/32/128, 10 seeds each, 200
runs on the same real-MNIST proxy the original investigation used) came back inconclusive, not a
clean win: `batch_size=1`/`8` (closest to the original per-example regime) still shows no clear
momentum benefit, matching the original finding; `batch_size=32`/`128` shows a striking rescue
effect from higher momentum, but it's confounded with `learning_rate=0.5` never being scaled up
for larger batches (the standard mini-batch SGD practice this sweep didn't apply), not clean
evidence for the original gradient-noise hypothesis. See
[research and analysis](research-and-analysis.md#momentum-under-mini-batch-gradients) for the
full numbers. `MomentumBackpropClassifierNetwork` remains not adopted as a default. A follow-up
sweep that scales `learning_rate` with `batch_size` could still change that, but hasn't been run
- see [structure](structure.md#possible-next-steps).
