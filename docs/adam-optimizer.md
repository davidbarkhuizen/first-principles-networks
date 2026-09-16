# Adam: a per-parameter adaptive-learning-rate optimizer

[← back to README](../README.md)

**Status: proposed, not started.** Written up front as a design/measurement plan before any of it
exists, per this repo's own practice of writing a plan down before implementation (see
[dataset sourcing](dataset-sourcing-proposal.md), [the Rust production cutover
plan](rust-production-cutover.md) for precedent) - to be updated with stage-by-stage status notes,
and corrected against whatever the actual build/measurements turn up, as it's executed.

## why this, and why now

The first item in [structure](structure.md#possible-next-steps)'s "new model primitives" tier:
every gradient-based sibling in this codebase (plain SGD, momentum, L2) shares one
`learning_rate` across every weight, for the life of training. Adam (Kingma & Ba, 2014) is a
structurally different mechanism worth its own measurement, not assumed to land the way momentum
did: a per-parameter *adaptive* learning rate, driven by running estimates of each weight's own
gradient mean and variance, rather than a single global velocity term.

Momentum was measured here to actively hurt at its canonical coefficient, and land as a flat null
otherwise (see [research and analysis](research-and-analysis.md#momentum-measured-not-worth-adopting)),
and the mini-batch retest ([research and
analysis](research-and-analysis.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help))
found momentum flat-to-actively-harmful even under lower-noise batch gradients, once a real
learning-rate confound was controlled for. Adam is worth testing independently rather than
assumed to fare the same way: it doesn't accumulate a single shared velocity the way momentum
does, and its per-parameter second-moment estimate is specifically designed to normalize away the
kind of gradient-scale variation this codebase's own measurements keep surfacing (e.g. the
`batch_size >= 32` learning-rate sensitivity above).

## scope

`AdamBackpropClassifierNetwork`, a sibling of `BackpropClassifierNetwork` only (single-output,
binary) - the same scope boundary every prior weight-update-rule sibling used
(`MomentumBackpropClassifierNetwork`, `L2RegularizedBackpropClassifierNetwork`); no prior sibling
in this family has launched into `MultiClassBackpropClassifierNetwork` at the same time it was
first built. RMSprop (Adam minus its momentum term) is deliberately out of scope for this
workplan - it becomes a cheap ablation once Adam's `AdamBackpropNode` exists (zero out its
momentum-like first-moment term), not a separate build, and is only worth doing if Adam's own
result says something worth isolating further.

## design: fits the existing accumulate/apply seam, no new mechanism needed

Same factory pattern as `momentum_layer.py`/`l2_regularization_layer.py`:

- `indrajala_ml/model/adam_layer.py`: `make_adam_node_cls(beta1, beta2, epsilon)` -> a
  `BackpropNode` subclass overriding only `apply_accumulated_gradient` (not
  `accumulate_gradient` - the existing batch-gradient-accumulation mechanism is already generic
  and correct as-is, the same override point momentum uses).
- `indrajala_ml/model/adam_backprop_classifier_network.py`: `AdamBackpropClassifierNetwork`,
  setting `hidden_layer_cls`/`output_layer_cls` to the Adam-configured layer class in `__init__`
  before `super().__init__()` runs - structurally identical to `MomentumBackpropClassifierNetwork`.

Per-parameter state, mirroring momentum's `_prev_weight_deltas`/`_prev_bias_delta`:

- `_weight_m`, `_weight_v` (lists, zero-initialized, one entry per input weight)
- `_bias_m`, `_bias_v` (scalars)
- `_t` (an integer step counter, zero-initialized, incremented at the top of every
  `apply_accumulated_gradient` call)

Update, using `g = accum / batch_size` exactly where momentum/plain SGD plug in the averaged
batch gradient:

```
t += 1
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
w -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

applied independently to every weight and to the bias.

**Why a per-node local `t` needs no shared/global counter**: every trainable node gets exactly
one `apply_accumulated_gradient` call per `learn()`/`learn_batch()` invocation - the network
never calls it for some nodes and not others on a given iteration - so a per-node counter stays
numerically identical to a hypothetical global one, the same reasoning that already lets
momentum's velocity live per-node with no shared state between nodes or layers.

## hyperparameters

`learning_rate` stays the normal, explicit, per-call training-time argument, unchanged from every
other sibling. For `beta1`/`beta2`/`epsilon`: proposing to default them to Kingma & Ba's own
published values (`0.9`, `0.999`, `1e-8`) rather than requiring them explicitly the way
`momentum`/`l2_lambda` are - those two were made required specifically because no coefficient
this codebase measured was safe to recommend as a default, whereas beta/epsilon are closer to
fixed algorithmic constants in virtually all real-world Adam usage, not tunable knobs this project
has an opinion on. **Open decision, flagged rather than assumed**: could instead require them
explicitly for consistency with momentum/L2's posture, at the cost of every call site needing to
state values nobody expects to vary.

## correctness validation

Same convention as `test_momentum_backprop_model.py`: a hand-computed (independently, via a
standalone script - not derived from the implementation under test) 2-3-step regression fixture
using `wire_fixed_single_hidden_node`, pinning exact weight/bias values via `pytest.approx`, plus
a sanity check that the first step (`t=1`, `m`/`v` starting at zero) reduces to a specific,
independently-computable value the way momentum's own first-step test reduces to plain SGD.

## measurement plan

Two stages, cheapest and most comparable to existing results first:

1. **The pinned tuned-XOR scenario** (`test_backprop_training_pipeline.py`'s pinned scenario,
   10-15 seeds) - establishes whether Adam helps or hurts at all on this codebase's smallest,
   most-measured target. Correction, found while executing this stage: momentum's own baseline was
   actually measured on the 320-example real-MNIST proxy, not this XOR scenario (see
   [research and analysis](research-and-analysis.md#momentum-measured-not-worth-adopting)) - this
   stage instead reuses the same pinned XOR scenario the binary-cross-entropy and ReLU
   investigations used.
2. **The 320-example real-MNIST proxy, crossed with `batch_size`** - reusing the exact scratch-sweep
   pattern from the learning-rate-vs-batch-size sweep (fork-based multiprocessing pool, ~25 minutes
   at this scale, see [research and
   analysis](research-and-analysis.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)).
   This is where Adam's actual selling point (per-parameter adaptive rates smoothing noisy
   gradients) should show up if it's going to, especially at `batch_size=1`'s per-example-noisy
   regime where momentum specifically failed.

A third, larger stage (real-MNIST-ensemble scale, via `ensemble_train.py`) is conditional on stage
2 showing something worth confirming at full scale - `ensemble_train.py`'s
`classifier_cls.randomized(layer_sizes, dimension, input_bounds)` call doesn't currently pass
through extra hyperparameters, so plugging Adam in there would need either a small
`ensemble_train.py` extension or a hand-rolled ad hoc parallel-training script (the same
uncommitted-prototype pattern several other real-scale investigations in this codebase already
used) - a real added step, not assumed away.

## risks and open questions

- **beta1/beta2/epsilon as defaults vs. required arguments** - see "hyperparameters" above; not
  resolved here, deliberately left as a call to make (or revisit) during stage 1.
- **Whether Adam's adaptive per-parameter scaling interacts with `batch_size=128`'s
  linear-scaling-rule divergence** (see the learning-rate-vs-batch-size writeup) is itself an
  interesting side question stage 2 could surface, without being this workplan's main goal -
  worth noting in the writeup either way, not chasing as a separate stage unless it comes up for
  free.
- **RMSprop as a follow-on ablation** - out of scope here (see "scope" above), noted so it isn't
  silently forgotten if Adam's result makes it worth doing.

## delivery stages (each its own PR, per this repo's practice)

1. ✅ This design document.
2. ✅ `adam_layer.py` (factory) + `adam_backprop_classifier_network.py` + hand-derived regression
   tests (`test_adam_layer.py`, isolated node-level; `test_adam_backprop_model.py`, whole-network,
   the same two-tier convention every prior sibling used) - the actual capability, buildable and
   mergeable on its own, independent of any measurement result. `beta1`/`beta2`/`epsilon` default
   to Kingma & Ba's own published values (0.9/0.999/1e-8) - the "hyperparameters" open decision
   above was resolved this way rather than requiring them explicitly, since they're closer to
   fixed algorithmic constants in real-world use than a knob momentum's own posture was about.
   Mini-batch (`learn_batch`) verified working with no extra code, inherited unchanged the same
   way every prior sibling's did. Full suite (850 tests, up from 845) passes.
3. ✅ The tuned-XOR-scale measurement, written up in [research and
   analysis](research-and-analysis.md#adam-tuned-xor-measurement-stage-3-of-the-adam-optimizer-workplan).
   At the demo-tuned `learning_rate=1.0` Adam collapses badly (58.27% vs. sigmoid's 97.80% mean,
   10 seeds) - the same "no default is safe without retuning" pattern cross-entropy/ReLU/momentum
   all showed. Retuned to `learning_rate=0.01-0.05`, Adam modestly beats the sigmoid baseline's
   mean with a visibly tighter spread, confirmed at 15 seeds (98.09-98.38% vs. 97.33% mean,
   0.75-1.14% vs. 1.66% stdev) - a real but modest edge on this small, near-ceiling toy problem,
   not a dramatic win the way ReLU's retuned result was, and not a null the way momentum's own
   retuned sweep was.
4. The real-MNIST-proxy batch-size sweep, written up (reusing the learning-rate-vs-batch-size
   sweep's own script skeleton).
5. Conditional: real-MNIST-ensemble-scale validation, only if stage 4's result justifies the
   ~30-minute-run cost.
6. Docs closeout: `structure.md`'s possible-next-steps entry updated to reflect the actual result
   (decision either way, the same posture every other sibling investigation here has taken),
   the same pattern the momentum re-test's own closeout PRs followed.
