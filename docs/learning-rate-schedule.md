# a learning-rate schedule (warmup)

[← back to README](../README.md)

**Status: all five stages done.** Written up front as a design/measurement plan before any of it
existed, per this repo's own practice (see [Adam optimizer](adam-optimizer.md),
[mini-batch gradient descent](mini-batch-gradient-descent.md) for precedent) - updated here with
stage-by-stage status notes as it was executed. Warmup confirmed to fix the documented
`batch_size=128` divergence, most cleanly paired with `momentum=0.9` at `warmup_steps>=25` - see
[research and
analysis](research-backprop-siblings.md#the-batch_size128-divergence-retested-with-warmup) for
the full measurement.

## why this, and why now

Every training loop in this codebase (`train_linear_classifier_network`,
`train_backprop_network_mini_batch`) uses one fixed `learning_rate` for every step of every
epoch - untested whether a schedule changes convergence or final accuracy on any target here.
This was a general, somewhat hypothetical gap until [the learning-rate-vs-batch-size
follow-up](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)
found a real, reachable failure case: at `batch_size=128` with the standard linear-scaling-rule
rate (`learning_rate = base_lr * batch_size = 64.0`), training diverges completely regardless of
momentum (every momentum value collapses to a 54-58% coin-flip with double-digit stdevs). That
entry's own interpretation named the standard literature fix directly: Goyal et al. 2017's own
linear-scaling-rule paper pairs it with a *gradual warmup*, specifically because applying the full
scaled rate from step one is unstable. This gives the schedule question a concrete, already-measured
target to fix, not a speculative "might this help somewhere" - the same posture every other
primitive in this codebase was built and measured against.

## scope

**Warmup only.** Decay (cosine, step, or otherwise) is a separate, independently-motivated
question (trading final-epoch precision for early-epoch speed) with no equivalently concrete
failure case driving it yet in this codebase's own measurements - deferred the same way RMSprop
was deferred out of the Adam workplan until Adam's own result justified it. If warmup's own
measurement surfaces a decay-shaped question for free, it gets noted, not chased as part of this
stage.

Scoped to `train_linear_classifier_network`/`train_backprop_network_mini_batch` (every
gradient-based student that already goes through them) - `LinearClassifierNetwork`'s discrete
minimum-disturbance rule has no `learning_rate` concept to schedule at all, so it's untouched.

## design: a callable learning_rate, duck-typed like everything else here

Both training functions currently take `learning_rate: float`. Widened to
`learning_rate: float | Callable[[int], float]` - every existing caller (a plain float) is
unaffected; a schedule is just a function from the current iteration index to a rate. At each
call site (`train_linear_classifier_network`'s `student.learn(learning_rate, ...)`,
`train_backprop_network_mini_batch`'s `student.learn_batch(learning_rate, batch)`), resolve the
rate once per iteration:

```python
current_lr = learning_rate(iterations) if callable(learning_rate) else learning_rate
student.learn(current_lr, reference_state, reference_category)
```

`iterations` already exists in both functions (incremented once per `learn`/`learn_batch` call,
used for the existing `convergence` series) - the natural, already-correct step index, needing no
new counter.

A new `indrajala_ml/lr_schedule.py` holds schedule-function factories, starting with:

```python
def linear_warmup(target_rate: float, warmup_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return target_rate * min((step + 1) / warmup_steps, 1.0)
    return schedule
```

**Open decision, flagged rather than assumed**: the `step + 1` above (so the very first call, at
`iterations=0`, gets `target_rate / warmup_steps` rather than `0.0`) is a specific, deliberate
choice - a literal `0.0` first step wastes one iteration entirely, learning nothing - but it's a
choice, not a derivation, and should be pinned by a hand-derived regression test during
implementation (the same convention every prior sibling's first-step behavior was pinned by), not
left to whatever the code happens to compute.

## correctness validation

Two-tier, matching this codebase's own convention:

- `test_lr_schedule.py`: `linear_warmup` in isolation - a few hand-computed steps
  (`step=0`/`warmup_steps-1`/`warmup_steps`/`warmup_steps+1`, confirming it holds at `target_rate`
  once past `warmup_steps`, via `pytest.approx`).
- `test_train.py`/`test_train_mini_batch.py`: an integration test confirming
  `train_linear_classifier_network`/`train_backprop_network_mini_batch` actually call a given
  schedule with the right, monotonically-increasing step indices (a fake recording schedule
  function, asserting the exact call sequence) - not just that training still runs without error.

## measurement plan

Retest the exact documented failure case: same 320-example real-MNIST digit-3 proxy, fan-in-aware
init, `[16]` hidden, 5 epochs, `batch_size=128`, `learning_rate` scaled to `64.0` (the value that
diverged), now wrapped in `linear_warmup(64.0, warmup_steps=N)` for a small sweep of `N` (e.g.
5/10/25/50 batches - `batch_size=128` on this 320-example proxy produces only ~3 batches/epoch, so
`N` here means batches, not epochs; a real-MNIST-scale retest, if this proxy result justifies it,
would need `N` re-thought in real batch-count terms). Compares against the documented
54-58%-coin-flip baseline (momentum 0.0-0.9, no warmup) - success is a `batch_size=128` result
that closes materially back toward `batch_size=32`'s own stable ~91-92% band, not just "less bad."

## risks and open questions

- **How many warmup steps is enough** - not derivable in advance from the failure mode alone;
  the sweep above is exactly how this gets answered empirically rather than guessed.
- **Per-step vs. per-epoch warmup** - `iterations` counts individual `learn`/`learn_batch` calls,
  not epochs; warmup measured in steps (as designed above) rather than epochs, since `batch_size`
  already determines how many steps an epoch contains and the failure this targets is itself a
  per-step instability (a large first move in weight-space), not an epoch-scale phenomenon.
- **Interaction with Adam** - stage 4 of the Adam workplan found Adam needs the *opposite*
  prescription from SGD/momentum at `batch_size=128` (a **fixed** rate, not scaled at all - see
  [research and
  analysis](research-adam-optimizer.md#adam-under-batch-size-a-much-bigger-cleaner-win-stage-4-of-the-adam-optimizer-workplan)),
  so warmup-on-top-of-scaling is a question about SGD/momentum specifically, not assumed to
  generalize to Adam - not chased here unless it comes up for free.

## delivery stages (each its own PR, per this repo's practice)

1. ✅ This design document.
2. ✅ `lr_schedule.py` (`linear_warmup`) + `test_lr_schedule.py`'s isolated regression tests - the
   schedule function itself, buildable and mergeable independent of the training-loop wiring.
3. ✅ Widened `train_linear_classifier_network`/`train_backprop_network_mini_batch`'s
   `learning_rate` type + the integration tests above - the actual capability, independent of any
   measurement result.
4. ✅ The `batch_size=128` warmup-step sweep, written up in [research and
   analysis](research-backprop-siblings.md#the-batch_size128-divergence-retested-with-warmup) (its
   own themed sub-doc, per the split convention - the same doc the original
   learning-rate-vs-batch-size follow-up this retests already lives in). Warmup confirmed to fix
   the divergence, most cleanly at `momentum=0.9`/`warmup_steps>=25` (90.25% ± 2.00% at
   `warmup_steps=50`, closing back to the documented `batch_size=32` stable band) - `momentum=0.0`
   improves too but less cleanly (81.38% ± 13.42% at the same `warmup_steps`), an unexpected
   asymmetry flagged but not chased further.
5. ✅ Docs closeout: `structure.md`'s possible-next-steps entry updated to reflect the actual
   result.
