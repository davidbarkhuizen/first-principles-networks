# a learning-rate schedule (warmup)

[← back to README](../../../README.md)

Design reference for `indrajala_ml/lr_schedule.py` and the widened `learning_rate: float |
Callable[[int], float]` training-loop parameter, built to fix a documented
`batch_size=128`/`learning_rate=64.0` divergence (every momentum value collapsing to a
54-58%-coin-flip). Built and measured; see [research and
analysis](../../research/research-backprop-siblings.md#the-batch_size128-divergence-retested-with-warmup) for
the full measurement and decision.

## design: a callable learning_rate, duck-typed like everything else here

Both `train_linear_classifier_network`/`train_backprop_network_mini_batch` take
`learning_rate: float | Callable[[int], float]` - every existing caller (a plain float) is
unaffected; a schedule is just a function from the current iteration index to a rate. At each
call site (`train_linear_classifier_network`'s `student.learn(learning_rate, ...)`,
`train_backprop_network_mini_batch`'s `student.learn_batch(learning_rate, batch)`), the rate is
resolved once per iteration:

```python
current_lr = learning_rate(iterations) if callable(learning_rate) else learning_rate
student.learn(current_lr, reference_state, reference_category)
```

`iterations` already exists in both functions (incremented once per `learn`/`learn_batch` call,
used for the existing `convergence` series) - the natural, already-correct step index, needing no
new counter. `iterations` counts individual `learn`/`learn_batch` calls, not epochs - warmup is
measured in steps rather than epochs, since `batch_size` already determines how many steps an
epoch contains and the failure this targets is itself a per-step instability (a large first move
in weight-space), not an epoch-scale phenomenon.

`indrajala_ml/lr_schedule.py` holds schedule-function factories, starting with:

```python
def linear_warmup(target_rate: float, warmup_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return target_rate * min((step + 1) / warmup_steps, 1.0)
    return schedule
```

`step + 1` (rather than `step`) is deliberate: the very first call, at `iterations=0`, gets
`target_rate / warmup_steps` rather than a wasted `0.0` - a literal `0.0` first step learns
nothing. A specific, deliberate choice, pinned by a hand-derived regression test rather than left
to whatever the code happened to compute.

## correctness validation

Two-tier, matching this codebase's own convention:

- `test_lr_schedule.py`: `linear_warmup` in isolation - a few hand-computed steps
  (`step=0`/`warmup_steps-1`/`warmup_steps`/`warmup_steps+1`, confirming it holds at `target_rate`
  once past `warmup_steps`, via `pytest.approx`).
- `test_train.py`/`test_train_mini_batch.py`: an integration test confirming
  `train_linear_classifier_network`/`train_backprop_network_mini_batch` actually call a given
  schedule with the right, monotonically-increasing step indices (a fake recording schedule
  function, asserting the exact call sequence) - not just that training still runs without error.
