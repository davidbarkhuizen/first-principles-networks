# research and analysis: backprop sibling measurements

[← back to research and analysis](research-and-analysis.md)

The investigation thread around `BackpropClassifierNetwork`'s alternate-init, weight-update-rule,
and activation siblings (Xavier/Glorot init, momentum, ReLU, L2, convolutional layers) - see
[research and analysis](research-and-analysis.md) for what this collection of docs is for.

## Xavier/Glorot init: measured, not worth adopting

A follow-up audit asked whether Glorot & Bengio 2010's Xavier init (`limit =
sqrt(6/(fan_in+fan_out))`, tailored for sigmoid networks specifically) would beat the fan-in-only
scheme already adopted (`limit = 1/sqrt(fan_in)`, closer to LeCun 1998's practical
simplification). Measured on the same proxy as [the ensemble
investigation](research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation) (320
examples, digit 3, 5 epochs, 5 seeds):

| config | mean test accuracy |
|---|---|
| current fan-in-only (production) | 93.75% |
| Xavier/Glorot, random bias | 93.75% |
| Xavier/Glorot, zero bias | 93.75% |

A clean null - (B) and (C) even produced identical per-seed results. Likely reason: once the
dominant saturation pathology is fixed (which both schemes do equally well), Xavier/Glorot's
further refinement (accounting for backward gradient variance via fan_out, not just forward
variance via fan_in) mainly matters for deeper networks or unusual layer-width ratios; this
codebase's networks are shallow throughout. **Not adopted** - the proxy result was unambiguous
enough that a ~30-minute real-scale run to re-confirm a null this clean wasn't judged worth it.

## momentum: measured, not worth adopting

Rumelhart, Hinton & Williams 1986's own generalized delta rule (this codebase's cited backprop
reference) includes a momentum term (`Δw_ji(n) = η·δ_j·a_i + α·Δw_ji(n-1)`) `apply_gradient`
never implemented. Measured on the same proxy (320 examples, digit 3, fan-in-aware init):

### canonical coefficient (α=0.9): hurts across a learning-rate sweep

Canonical coefficient (α=0.9) across a learning-rate sweep, 5 seeds, 5 epochs:

| config | mean test accuracy |
|---|---|
| no momentum, lr=0.5 (baseline) | 93.75% |
| momentum=0.9, lr=0.5 | 91.75% |
| momentum=0.9, lr=0.25 | 90.75% |
| momentum=0.9, lr=0.1 | 91.50% |
| momentum=0.9, lr=0.05 | 91.25% |

### a finer sweep at the tuned rate: initially promising, then a flat null at n=15

α=0.9 robustly *hurt* across a 10x learning-rate range. A finer sweep at the tuned lr=0.5, more
seeds (15) after an initial 5-seed run looked misleadingly promising:

| momentum | mean test accuracy | stdev |
|---|---|---|
| 0.0 (baseline) | 93.83% | 0.57% |
| 0.3 | 93.83% | 1.10% |
| 0.4 | 93.67% | 1.10% |
| 0.5 | 94.00% | 0.70% |
| 0.6 | 93.67% | 1.20% |
| 0.7 | 93.67% | 1.10% |

All six land within 0.33 points of each other, well inside the noise band at n=15 - a flat,
statistically indistinguishable landscape (an earlier 5-seed run of momentum=0.5 had shown
94.50%, apparently beating baseline; at n=15 it settled to 94.00%, within noise - exactly the
small-sample mirage re-measuring exists to catch).

### interpretation and decision

**Interpretation:** momentum's canonical value is actively harmful here, plausibly because
per-example online SGD's gradients are noisier than the mini-batch/batch gradients momentum's
literature validates against, so accumulating velocity amplifies noise rather than smoothing
signal. **Decision:** not adopted as a default - `MomentumBackpropClassifierNetwork` requires an
explicit `momentum` argument rather than defaulting to any value tested. Built and kept anyway
(unlike the Xavier/Glorot finding) as a real capability, e.g. for retesting once mini-batch
gradients exist (see [mini-batch gradient descent](mini-batch-gradient-descent.md)).

## ReLU hidden-layer activation: a clean win, once retuned

Built as `ReLUBackpropClassifierNetwork` (a new `hidden_layer_cls` extension point). Measured
against the same pinned XOR scenario as the binary cross-entropy investigation, 10 seeds, at the
demo-tuned `learning_rate=1.0`:

| | sigmoid | ReLU (lr=1.0, untuned) |
|---|---|---|
| mean | 97.80% | 74.03% |

A large, consistent gap. Dead units were checked directly and ruled out (only 1 of 8 hidden
units dead - not enough to explain a 24-point gap). Sweeping `learning_rate` down for ReLU alone:

| learning_rate | mean training accuracy |
|---|---|
| 1.0 | 74.03% |
| 0.5 | 82.60% |
| 0.25 | 97.73% |
| 0.1 | 99.27% |
| 0.05 | 99.20% |
| 0.01 | 99.47% |

At `learning_rate=0.1`, ReLU's mean (99.27%) *exceeds* sigmoid's own tuned performance (97.80%).
The mechanism: ReLU's derivative in its active region is a flat 1.0, with no damping term the
way sigmoid's `a(1-a)` provides - the same root cause identified for binary cross-entropy's and
momentum's own learning-rate sensitivity, applied to a third case. **Decision:** adopted as a
genuine, positive finding, not just "didn't lose" - but still needs its own tuned learning rate
wherever used, not a drop-in replacement for sigmoid at existing hyperparameters.

## L2 weight regularization: closes the overfitting gap, doesn't improve it

Measured on a fixed, finite MNIST proxy (400 examples, digit 3, 80/20 split, fan-in-aware init,
`learning_rate=0.5`) rather than XOR's continuously-resampled target, since L2's whole purpose is
generalization:

| l2_lambda | mean train accuracy | mean test accuracy | train-test gap |
|---|---|---|---|
| 0.0 (none) | 98.50% | 93.75% | 4.75 pts |
| 0.0001 | 98.31% | 92.75% | 5.56 pts |
| 0.001 | 98.19% | 92.25% | 5.94 pts |
| 0.01 | 93.75% | 93.75% | 0 pts |
| 0.1 | 55.13% | 59.25% | −4.13 pts |
| 0.5 | 55.13% | 59.25% | −4.13 pts |

Three regimes: too weak (0.0001-0.001, slightly worse on both metrics), the gap genuinely closes
at 0.01 but by training accuracy *falling to meet* test accuracy rather than the reverse, and
collapse at 0.1+ (weights decayed to near-zero, predictions collapsed to a single constant
class - confirmed directly, not assumed from the identical numbers). No `l2_lambda` improved
held-out accuracy above the unregularized baseline - plausibly because a single 16-node hidden
layer on 320 examples doesn't have severe enough overfitting for a weight-magnitude penalty to
have room to help. **Decision:** not adopted as a default, for the same reason as momentum -
`L2RegularizedBackpropClassifierNetwork` requires an explicit `l2_lambda`. Kept as a real,
tested capability for a more overfitting-prone future scenario.

## momentum under mini-batch gradients

[Mini-batch gradient descent](mini-batch-gradient-descent.md) exists specifically to retest
momentum's null finding under lower-noise batch gradients. Same proxy as the original momentum
investigation (320 real MNIST examples, digit 3, fan-in-aware init, `learning_rate=0.5`, 5
epochs, `[16]` hidden), momentum coefficients (0.0, 0.3, 0.5, 0.7, 0.9) crossed with batch sizes
(1, 8, 32, 128), 10 seeds each, 200 runs:

| momentum | batch_size=1 | batch_size=8 | batch_size=32 | batch_size=128 |
|---|---|---|---|---|
| 0.0 | 91.43% ± 1.19% | 91.39% ± 0.87% | 84.65% ± 6.78% | 71.24% ± 13.75% |
| 0.3 | 91.39% ± 1.05% | 91.50% ± 0.76% | 88.30% ± 2.79% | 72.26% ± 16.03% |
| 0.5 | 91.65% ± 0.95% | 91.59% ± 0.79% | 89.75% ± 1.21% | 77.02% ± 12.45% |
| 0.7 | 91.74% ± 1.07% | 91.85% ± 0.73% | 90.82% ± 1.12% | 84.29% ± 6.29% |
| 0.9 | 89.06% ± 1.46% | 90.82% ± 1.17% | 91.64% ± 0.77% | **87.66% ± 2.62%** |

`batch_size=1` replicates the original finding (momentum=0.9 hurts, 0.3-0.7 flat within noise) -
a methodology sanity check that passed. But accuracy at fixed `learning_rate=0.5` collapses as
batch size grows regardless of momentum (91.43% -> 71.24% at momentum=0.0), and momentum visibly
rescues that collapse (84.65% -> 91.64% at batch_size=32, momentum 0.0 -> 0.9). That rescue is
real and reproducible, but confounded: `learning_rate=0.5` was never scaled up for larger batches
(a batch of `b` produces `320/b` update steps per epoch - far fewer total steps unless the rate
compensates, the standard mini-batch "linear scaling rule"). Momentum's velocity term is a
plausible partial substitute for that missing scaling, which would explain the rescue *without*
it being evidence about gradient noise specifically - the mechanism this retest was meant to
isolate. **Decision:** not a clean confirmation or reversal - `MomentumBackpropClassifierNetwork`
remains not adopted as a default. A follow-up sweep scaling `learning_rate` with `batch_size`
would be needed to isolate the effect cleanly; not yet run.

## the learning-rate-vs-batch-size follow-up: the confound was real, and momentum still doesn't help

The follow-up flagged above, run to isolate the previous entry's large-batch momentum "rescue"
from the untuned-learning-rate confound it identified. Identical setup (same 320-example
real-MNIST digit-3 proxy, fan-in-aware init, `[16]` hidden, 5 epochs, momentum 0.0-0.9 crossed
with batch sizes 1/8/32/128, 10 seeds each, 200 runs, parallelized the same fork-based way, ~25.3
minutes wall-clock) with one change: `learning_rate` now scales with `batch_size` via the
standard mini-batch "linear scaling rule" (`learning_rate = 0.5 * batch_size` - 0.5/4.0/16.0/64.0)
instead of a fixed 0.5.

| momentum | batch_size=1 (lr=0.5) | batch_size=8 (lr=4.0) | batch_size=32 (lr=16.0) | batch_size=128 (lr=64.0) |
|---|---|---|---|---|
| 0.0 | 92.33% ± 0.38% | 92.39% ± 0.71% | **92.20% ± 1.00%** | 56.10% ± 11.48% |
| 0.3 | 91.99% ± 0.41% | 91.67% ± 0.49% | 92.28% ± 0.62% | 58.05% ± 11.15% |
| 0.5 | 91.71% ± 0.46% | 91.60% ± 1.10% | 91.62% ± 0.75% | 58.46% ± 16.53% |
| 0.7 | 91.98% ± 0.78% | 91.96% ± 0.82% | 91.77% ± 0.81% | 57.47% ± 14.70% |
| 0.9 | 88.58% ± 1.76% | 86.30% ± 10.97% | 87.47% ± 6.00% | 54.50% ± 12.51% |

### two clean results, in opposite directions

Two clean results, in opposite directions:

- **The confound hypothesis is confirmed at `batch_size=32`.** With `learning_rate` properly
  scaled, `momentum=0.0` alone reaches 92.20% ± 1.00% - matching `batch_size=1`'s own baseline
  (92.33%) and *exceeding* the previous entry's own momentum=0.9-assisted "rescue" figure at this
  batch size (91.64%) - with no momentum contribution at all. The previous sweep's rescue effect
  was exactly what it was flagged as being: an artifact of the untuned learning rate, not evidence
  that momentum helps under lower-noise mini-batch gradients. Once the confound is removed,
  momentum has nothing left to rescue.
- **Momentum still doesn't help, and the picture against it gets stronger, not weaker.** At every
  batch size where training is stable (1, 8, 32), `momentum=0.9` is now both lower-mean and
  higher-variance than `momentum=0.0` - most strikingly at `batch_size=8`, where it drops to
  86.30% with a 10.97-point stdev (individual seeds visibly diverging), compared to `momentum=0.0`'s
  tight 92.39% ± 0.71%. Under a properly-scaled learning rate, momentum is actively
  destabilizing, not neutral-to-harmful the way the original per-example finding suggested - the
  clearest evidence yet against adopting it.
- **`batch_size=128`'s `lr=64.0` diverges completely, independent of momentum.** Every momentum
  value collapses to a coin-flip 54-58% with double-digit stdevs - a real training-instability
  regime, not a null result to read anything into. A 128x learning-rate multiplier from a single
  update step is a large first move in weight-space regardless of what comes after; this matches
  real large-batch-training practice (e.g. Goyal et al. 2017's own linear-scaling-rule paper pairs
  it with a *gradual warmup*, specifically because applying the full scaled rate from step one is
  unstable) - this codebase's version of that same finding, hit directly rather than assumed.

### decision

**Decision:** the original momentum-under-mini-batch-gradients question (does momentum help more
once gradients are less noisy) is now cleanly answered: **no** - once the learning-rate confound
is removed, momentum is flat-to-harmful at every batch size that trains stably, and the earlier
"rescue" at `batch_size=32`/`128` fully explained by the untuned rate rather than by momentum
itself. `MomentumBackpropClassifierNetwork` remains not adopted as a default, on stronger evidence
than before. The `batch_size=128` divergence is a separate, real finding of its own (naive linear
LR scaling needs warmup to be usable at large batch multipliers) - not investigated further here,
since it answers "is scaling alone sufficient" (no) rather than the momentum question this sweep
was designed around.

## convolutional layers on UCI digits

[Convolutional layers](convolutional-layers.md)'s `ConvMultiClassBackpropClassifierNetwork` was
validated against the dense baseline on full UCI digits (1797 samples, the same fixed
`split_train_test(seed=0)` split `demo_uci_digit_recognition.py` uses, `learning_rate=0.5`,
`epochs=30`), at a deliberately comparable trainable-parameter budget:

| architecture | shape | trainable parameters |
|---|---|---|
| dense | `[32]` hidden | 2410 |
| conv | `kernel_size=3`, `channel_count=4` conv -> `[16]` dense | 2530 |

8 seeds each (fixed split, varying only init/training randomness):

| architecture | train accuracy | test accuracy |
|---|---|---|
| dense | mean 99.57%, stdev 0.08% | mean 96.69%, stdev 0.62% |
| conv | mean 99.78%, stdev 0.10% | mean 96.52%, stdev 0.73% |

The two test-accuracy means differ by 0.17 points - smaller than either architecture's own
stdev, both landing in the same 95.3%-97.5% per-seed band. A flat, statistically
indistinguishable result, honestly reported as a null rather than stretched into a win for
either side - plausibly because UCI digits' 8x8 images and 1797 samples are already close to
saturated for a dense network at this parameter budget, leaving little room for convolution's
real advantages (translation invariance, fewer effective parameters per feature) to show up as a
test-accuracy improvement. **Decision:** not adopted as a demonstrated win at this scale;
`ConvMultiClassBackpropClassifierNetwork` remains a real, correct, tested capability. Whether
real MNIST's larger spatial structure would separate the two differently is a real, further
wall-clock cost (~30 minutes per architecture per seed) this result leaves open rather than
answered - see [structure](structure.md#possible-next-steps).
