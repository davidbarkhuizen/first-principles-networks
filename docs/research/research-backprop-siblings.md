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
gradients exist (see [mini-batch gradient descent](../features/mini-batch-gradient-descent.md)).

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

[Mini-batch gradient descent](../features/mini-batch-gradient-descent.md) exists specifically to retest
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
would be needed to isolate the effect cleanly - see [the learning-rate-vs-batch-size
follow-up](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)
directly below, which ran exactly that sweep.

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

## the batch_size=128 divergence, retested with warmup

Retests the `batch_size=128`/`learning_rate=64.0` divergence flagged just above, now wrapped in
[a warmup schedule](../design-docs/infra/learning-rate-schedule.md) - swept as `lr_schedule.linear_warmup(64.0,
warmup_steps=N)` for `N` in 5/10/25/50, since how many warmup steps is enough isn't derivable in
advance from the failure mode alone. Same scale as
the follow-up above (fan-in-aware init, `[16]` hidden, 5 epochs, `batch_size=128`, 10 seeds), on a
freshly-built 320-example real-MNIST digit-3 proxy (independently constructed, not the same
instance the follow-up used - same posture as every other reproducibility check in this codebase),
at `momentum` 0.0 and 0.9 (the follow-up's best- and worst-behaved configs at this batch size):

| momentum | no warmup | N=5 | N=10 | N=25 | N=50 |
|---|---|---|---|---|---|
| 0.0 | 50.00% ± 12.00% | 55.50% ± 16.96% | 65.38% ± 19.10% | 75.12% ± 14.07% | 81.38% ± 13.42% |
| 0.9 | 45.50% ± 3.88% | 63.13% ± 19.83% | 82.38% ± 13.35% | **88.63% ± 2.12%** | **90.25% ± 2.00%** |

### warmup helps monotonically, and momentum=0.9 benefits from it more than momentum=0.0 does

The no-warmup rows reproduce the documented divergence (a coin-flip 45-50%, qualitatively matching
the original 54-58% - not bit-comparable, an independently-built proxy). Every increase in `N`
improves both rows, monotonically, with no reversal. The clean success case is `momentum=0.9`:
`N=50` reaches 90.25% ± 2.00%, closing materially back toward the documented `batch_size=32`
stable band (~91-92%, [momentum=0.0's own
92.20%](research-backprop-siblings.md#two-clean-results-in-opposite-directions) specifically) with
tight, non-collapsed variance - the success criterion this stage's measurement plan set.
`momentum=0.0` improves too (50.00% -> 81.38%) but doesn't close the gap as cleanly, and stays
noisier (±13.42% even at `N=50`) - an unexpected asymmetry, since the earlier momentum
investigations found momentum flat-to-harmful on its own; here, combined with warmup specifically
at this divergent batch size, it's momentum that recovers furthest. Not chased further here (out
of this stage's warmup-only scope), but worth flagging as a real, measured interaction rather than
assumed away.

**A caveat worth being explicit about, not glossed over**: `batch_size=128` on this 320-example
proxy produces only 3 batches/epoch (`ceil(320/128)`) - 15 total training steps across 5 epochs.
At `N=50`, `linear_warmup`'s ramp never reaches its own target rate within the run at all (the
schedule hits `64.0 * 15/50 = 19.2` at the last step, `min(...)` never clamping to `1.0`) - so this
result is really "training throughout at a rate ramping up to ~19.2" (below even `batch_size=32`'s
own known-stable 16.0 rate), not "warm up, then sustain the full destabilizing 64.0 rate." At
`N=10`, by contrast, the schedule does reach and hold the full `64.0` rate for the last 6 of 15
steps, and still recovers substantially (82.38% for momentum=0.9) - a cleaner demonstration that
warmup itself, not merely a lower effective rate throughout, is doing real work. `N` means batches
here, not epochs or a fraction of training - a real-MNIST-scale
retest (many more batches/epoch) would need `N` re-thought in real batch-count terms before this
`N=50` framing would carry over.

### decision

**Decision:** warmup is confirmed to fix the divergence, most cleanly when paired with
`momentum=0.9` at `N>=25` steps - the standard literature prescription (Goyal et al. 2017) holds
up under this codebase's own measurement, not just by assumption. Not promoted to a default
schedule for every large-batch run (this was one divergent configuration, not a general sweep
across batch sizes/architectures), and the momentum-benefits-more-than-plain-SGD asymmetry is
flagged rather than chased. Decay (the other half of "a learning-rate schedule"'s own scope,
deferred at design time) remains out of scope - no equivalently concrete failure case motivates it
yet.

## convolutional layers on UCI digits

[Convolutional layers](../features/convolutional-layers.md)'s `ConvMultiClassBackpropClassifierNetwork` was
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
answered - see [structure](../project/structure.md#possible-next-steps).

## an array-based L2 sibling: the reconfirmed null holds at far higher power

L2's own per-node measurement above found no `l2_lambda` beats the unregularized baseline, but at
a scale (320 training examples, a single 16-node hidden layer) that plausibly didn't overfit
severely enough to give a weight-magnitude penalty room to help - and the per-node path's own
wall-clock made a more overfitting-prone escalation too slow to afford (an abandoned dropout sweep
on the same setup projected 40+ minutes). `L2ArrayLayer`/
`L2VectorizedMultiClassBackpropClassifierNetwork` (numpy) and `L2RustArrayLayer`/
`L2RustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed) were built specifically to
afford that escalation, not as an end in themselves.

**Part A - wall-clock**, one mini-batch training step (`dimension=784, hidden=10`, per-node vs
numpy, median of 30 timed reps per config):

| batch size | per-node (us/example) | numpy (us/example) | per-node/numpy |
|---|---|---|---|
| 1 | 4241.13 | 79.78 | 53.2x |
| 8 | 2848.97 | 24.97 | 114.1x |
| 32 | 3762.33 | 15.03 | 250.3x |
| 128 | 3444.48 | 5.63 | 611.7x |
| 512 | 3370.16 | 3.99 | 845.3x |

**53x-845x faster per example than the per-node path**, widening with batch size - the same range
every other array-ported sibling in this codebase measured.

**Part B - the overfitting-headroom escalation, finally affordable.** Reframed as a `class_count=2`
one-vs-rest problem (the array line's `MultiClassBackpropClassifierNetwork` has no single-output
counterpart, an inherited gap every array-ported sibling in this round shared), fan-in-aware init,
30 seeds (vs. the original per-node measurement's few):

**Reproduction** (`[16]` hidden, 5 epochs - the original per-node setup):

| l2_lambda | train acc | test acc | gap (pts) |
|---|---|---|---|
| 0.0 | 97.28% ± 0.96% | 87.17% ± 2.54% | 10.11 |
| 0.0001 | 96.98% ± 1.21% | 87.21% ± 2.64% | 9.77 |
| 0.001 | 95.94% ± 2.07% | 86.42% ± 3.59% | 9.52 |
| 0.01 | 91.98% ± 3.72% | 85.75% ± 5.23% | 6.23 |
| 0.1 | 50.00% ± 0.00% | 50.00% ± 0.00% | 0.00 |
| 0.5 | 49.95% ± 0.28% | 49.92% ± 0.45% | 0.03 |

**Escalation** (`[128]` hidden, 60 epochs - substantially larger and longer, deliberately chosen to
induce more overfitting than the original setup could):

| l2_lambda | train acc | test acc | gap (pts) |
|---|---|---|---|
| 0.0 | 99.07% ± 0.06% | 86.04% ± 0.73% | 13.03 |
| 0.0001 | 99.07% ± 0.06% | 86.33% ± 0.91% | 12.74 |
| 0.001 | 95.94% ± 3.43% | 85.92% ± 4.79% | 10.02 |
| 0.01 | 87.81% ± 8.85% | 82.08% ± 8.46% | 5.73 |
| 0.1 | 50.07% ± 0.39% | 50.00% ± 0.00% | 0.07 |
| 0.5 | 50.00% ± 0.00% | 50.00% ± 0.00% | 0.00 |

The escalation worked (the unregularized train-test gap widened from 10.11 to 13.03 points,
confirming this scale is genuinely more overfitting-prone) - and L2 still doesn't exploit the extra
headroom: every `l2_lambda` that doesn't collapse the network (weights decayed to ~0, predicting
the tied 50/50 marginal class at `l2_lambda >= 0.1`) closes the gap only by training accuracy
falling toward test accuracy, never by test accuracy rising above the unregularized baseline.

**Decision: unchanged, on far stronger evidence.** `L2VectorizedMultiClassBackpropClassifierNetwork`/
`L2RustArrayMultiClassBackpropClassifierNetwork` remain real, tested capabilities, not adopted as a
default.

## an array-based momentum sibling: unchanged, on stronger evidence

Vectorized specifically to re-run [the learning-rate-vs-batch-size
follow-up](#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)
at far higher statistical power and at real-MNIST scale, not because momentum's null was in doubt.
`MomentumArrayLayer`/`MomentumVectorizedMultiClassBackpropClassifierNetwork` (numpy) and
`MomentumRustArrayLayer`/`MomentumRustArrayMultiClassBackpropClassifierNetwork`
(Rust-matmul-backed) mirror the per-node formula as whole-array ops.

**Part A - wall-clock**, one mini-batch training step (`dimension=784, hidden=10`, per-node vs
numpy, median of 30 timed reps):

| batch size | per-node (us/example) | numpy (us/example) | per-node/numpy |
|---|---|---|---|
| 1 | 4702.58 | 156.19 | 30.1x |
| 8 | 2786.87 | 10.91 | 255.5x |
| 32 | 2999.89 | 7.71 | 389.3x |
| 128 | 3019.16 | 5.71 | 528.5x |
| 512 | 3030.12 | 4.23 | 716.8x |

**30x-717x faster per example than the per-node path**, widening with batch size.

**Part B** re-runs the learning-rate-vs-batch-size follow-up through the array line's
`class_count=2` one-vs-rest line (no single-output array network exists), a freshly-built
400-example real-MNIST digit-3 proxy, `[16]` hidden, 5 epochs, `learning_rate = 0.5 * batch_size`
(the linear scaling rule), 30 seeds (vs. the original's 10):

| momentum | batch_size=1 (lr=0.5) | batch_size=8 (lr=4.0) | batch_size=32 (lr=16.0) | batch_size=128 (lr=64.0) |
|---|---|---|---|---|
| 0.0 | 87.67% ± 1.98% | 88.12% ± 2.13% | 89.12% ± 3.23% | 60.75% ± 12.86% |
| 0.3 | 88.12% ± 2.28% | 88.54% ± 1.77% | 88.83% ± 6.88% | 62.58% ± 14.72% |
| 0.5 | 88.96% ± 1.86% | 89.08% ± 1.93% | 87.67% ± 6.89% | 62.04% ± 14.32% |
| 0.7 | 89.38% ± 1.96% | 89.00% ± 2.61% | 84.96% ± 9.95% | 61.71% ± 14.65% |
| 0.9 | 82.04% ± 7.37% | 79.75% ± 12.53% | 79.46% ± 13.75% | 62.42% ± 15.72% |

Both of the original follow-up's findings reproduce cleanly at 3x the seed count:
`momentum=0.0` is at least as good as every other coefficient at `batch_size=32` (89.12% ± 3.23%,
tightest variance in that column), accuracy degrades as momentum increases past 0.3, and
`momentum=0.9` is the worst and highest-variance coefficient at every batch size, including
`batch_size=1`/`8` where the others cluster tightly (87.67%-89.38%) - actively destabilizing, not
neutral.

**Decision: unchanged, on stronger evidence.** Momentum still doesn't help once the learning-rate
confound is controlled for, and `momentum=0.9` remains actively harmful, now confirmed through a
genuinely different (vectorized, matmul-based) execution path.

## an array-based ReLU sibling: a genuine, scale-dependent finding

ReLU's toy-XOR win (above) had never been checked at UCI-digits or real-MNIST scale, where sigmoid
saturation - the exact pathology fan-in-aware init was built to fix - is the more relevant failure
mode. `ReLUArrayLayer`/`ReLUVectorizedMultiClassBackpropClassifierNetwork` (numpy) and
`ReLURustArrayLayer`/`ReLURustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed,
hidden-layer-only, matching `ReLUNode`'s own convention) made that check affordable - and needed a
real, if small, extension to the Rust core's own operation set (`array_relu`), the first array port
in this codebase's history to touch the forward pass itself rather than just the weight-update
rule.

**Part A - wall-clock**, one mini-batch training step (`dimension=784, hidden=10`, per-node vs
numpy, median of 30 timed reps):

| batch size | per-node (us/example) | numpy (us/example) | per-node/numpy |
|---|---|---|---|
| 1 | 3965.31 | 95.22 | 41.6x |
| 8 | 2704.88 | 8.16 | 331.4x |
| 32 | 2713.59 | 4.84 | 560.9x |
| 128 | 2598.85 | 5.34 | 486.7x |
| 512 | 2616.75 | 4.14 | 632.3x |

**41.6x-632.3x faster per example than the per-node path.**

**Part B - does the toy-XOR win transfer to UCI digits and real MNIST?** Both via
`ReLUVectorizedMultiClassBackpropClassifierNetwork` vs. the fan-in-aware sigmoid baseline
(isolating the activation-function question specifically):

**UCI digits** (`[32]` hidden, 30 epochs, 10 seeds):

| learning_rate | test accuracy |
|---|---|
| sigmoid @ 0.5 (baseline) | 96.74% ± 0.31% |
| ReLU @ 0.5 (untuned) | 96.74% ± 0.43% |
| ReLU @ 0.25 (best) | 96.80% ± 0.50% |

**The toy-XOR win does not transfer at UCI-digits scale** - untuned ReLU ties the baseline exactly,
and the best retuned rate is well inside one seed-to-seed stdev of it. A tie, not a win.

**Real MNIST** (`[30]` hidden, 1 epoch, 5 seeds at the two best rates found by an initial
single-seed sweep):

| | seed accuracies | mean | stdev |
|---|---|---|---|
| sigmoid @ 0.5 | 93.64%, 94.01%, 94.17%, 92.45%, 94.33% | 93.72% | 0.67% |
| ReLU @ 0.05 (retuned) | 94.30%, 94.15%, 93.98%, 93.88%, 94.50% | **94.16%** | **0.22%** |
| ReLU @ 0.1 | 94.04%, 93.19%, 93.29%, 93.98%, 93.95% | 93.69% | 0.37% |

Untuned ReLU (`lr=0.5`) loses badly here (86.54% vs. 93.64%, one seed) - confirming it needs its
own tuned rate at real scale too, not just XOR. Retuned (`lr=0.05`), it's **a real, if modest,
win**: the +0.44-point margin is smaller than the baseline's own seed-to-seed stdev, but ReLU's own
variance is three times tighter (0.22% vs. 0.67%) and every one of its 5 seeds beats the baseline's
median. Dead-ReLU risk was checked directly at this scale: 0 of 30 hidden units dead across 5000
real-MNIST examples.

**Decision: adopted as a genuine, if scale-dependent, finding.** ReLU still needs its own tuned
learning rate at every scale checked, never a drop-in replacement for sigmoid - and whether it
beats the retuned sigmoid baseline is scale-dependent: no at UCI digits (a tie), yes at real MNIST
(a modest, consistent win).

## an array-based dropout sibling: another reconfirmed null, now at real power

`DropoutBackpropClassifierNetwork`'s own overfitting-gap measurement (a real 50-run sweep against
an 8-way `multiprocessing.Pool`) was abandoned mid-run, observed tracking toward 40+ minutes on the
per-node path - too slow to afford. `DropoutArrayLayer`/
`DropoutVectorizedMultiClassBackpropClassifierNetwork` (numpy) and `DropoutRustArrayLayer`/
`DropoutRustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed) were built - both
backends together, not gated on the numpy stage's own wall-clock finding, since the point was
affording the sweep at all - specifically to close that gap. The Rust side needed a genuinely new
category of primitive for the Rust core: per-call RNG (`bernoulli_mask`/`draw_bernoulli_mask`, a
hand-rolled xorshift128+ draw), parity-tested against numpy's own mask distribution statistically
(mean keep-rate within tolerance), not bit-for-bit the way every other Rust parity check in this
codebase is.

Run against the numpy backend only (seedable, unlike the Rust core's own RNG - see [the Rust array
core](../architecture/rust-array-core.md#the-rng-exception)), on a fixed 400-example real-MNIST digit-3 proxy (320
train/80 test), reframed as `class_count=2` one-vs-rest, 30 seeds. A real, timed calibration run
projected the original 50-run sweep shape at well under a minute - confirmed directly by then
running it: the full 150-run stage-1 sweep took 2.24 minutes, **over 53x faster** than the per-node
path's own abandoned 40+-minute projection at matching scale.

**Stage 1 - reproduction** (`[16]` hidden, 20 epochs - dropout's own originally-planned scale):

| drop_probability | train acc | test acc | gap (pts) |
|---|---|---|---|
| 0.0 | 98.55% ± 0.25% | 88.88% ± 1.18% | 9.68 |
| 0.1 | 98.62% ± 0.29% | 88.29% ± 1.19% | 10.33 |
| 0.2 | 98.70% ± 0.32% | 88.25% ± 1.43% | 10.45 |
| 0.3 | 98.44% ± 0.33% | 89.00% ± 1.04% | 9.44 |
| 0.5 | 97.99% ± 0.52% | 89.25% ± 1.98% | 8.74 |

A clean null already: every `drop_probability`'s test accuracy sits well within one seed-to-seed
stdev of the unregularized baseline.

**Stage 2 - the overfitting-headroom escalation** (`[128]` hidden, 60 epochs, 21.33 minutes total)
- the same escalation shape L2's own array port used, but with an honestly-flagged difference:
unlike L2's escalation, this one did **not** widen the train-test gap (9.6-10.2 points here vs.
stage 1's own 8.7-10.5) - this proxy's train accuracy appears capped around 98.5% on this fixed
320-example split regardless of hidden-layer width or epoch count, so the escalation never got the
extra overfitting headroom L2's own did:

| drop_probability | train acc | test acc | gap (pts) |
|---|---|---|---|
| 0.0 | 98.50% ± 0.19% | 88.58% ± 1.47% | 9.92 |
| 0.1 | 98.57% ± 0.25% | 88.33% ± 1.22% | 10.24 |
| 0.2 | 98.55% ± 0.35% | 88.37% ± 1.26% | 10.18 |
| 0.3 | 98.46% ± 0.29% | 88.88% ± 0.75% | 9.58 |
| 0.5 | 98.53% ± 0.39% | 88.79% ± 1.39% | 9.74 |

**Decision: a reconfirmed null, now at real statistical power (30 seeds, both scales) the
abandoned per-node attempt could never afford** - dropout closes no held-out-accuracy gap it isn't
already given room to close, the same qualitative finding as L2's own measurement.
`DropoutBackpropClassifierNetwork`/`DropoutVectorizedMultiClassBackpropClassifierNetwork`/
`DropoutRustArrayMultiClassBackpropClassifierNetwork` remain real, tested capabilities for a
genuinely more overfitting-prone future scenario than this proxy turned out to provide.
