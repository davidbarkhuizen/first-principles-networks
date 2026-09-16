# research and analysis

[← back to README](../README.md)

Write-ups of the investigations behind this codebase's design decisions - the measurements that
drove a choice, not just the choice itself. Unlike [structure](structure.md) (what the code does
now) or [demos](demos.md) (what each demo shows), this is where the *why* lives, condensed to the
question asked, what was measured, and the decision it produced.

## parallelizing MNIST training

Training `MultiClassBackpropClassifierNetwork` on full real MNIST (60000 images, 784 dims, pure
Python) benchmarked at ~12.5 minutes/epoch on this 8-CPU machine - the question was whether
`multiprocessing` could cut that meaningfully.

**Rejected: data-parallel synchronous weight averaging.** Shard the data 8 ways, train 8 replicas
of the same network for one epoch, average their weights, broadcast back. Measured (2000-image
subset, 2 epochs, 8 workers): only ~2x speedup (each epoch re-shipped the entire shard over IPC),
and a real accuracy drop (71.3% vs. 92.7% train) - averaging the weights of several replicas of a
*nonlinear* network isn't the same as averaging the functions they learned, and synchronizing
more often to fix that trades the accuracy loss back for less parallelism. No setting of that
dial was clearly good on all three axes (speed/accuracy/sync frequency), so this was dropped
rather than tuned further.

**Adopted: an ensemble of independently-trained one-vs-rest classifiers.** Rather than sharing a
hidden layer, train 10 completely independent `BackpropClassifierNetwork`s, one per digit, each
on its own small class-balanced binary dataset - genuinely independent sub-problems, so there is
no synchronization step of any kind. Measured on 3 digits (0, 1, 7): 98.21% 3-way ensemble argmax
accuracy, comparable to or better than the shared-hidden-layer approach on the full 10-way
problem. Full-scale result: **~26 minutes wall-clock** (10 jobs, 8 parallel workers, 2 rounds)
vs. ~2-2.5 hours for the original single-network plan - a ~5-6x speedup *alongside* better
accuracy, the opposite tradeoff the rejected approach hit, because the sub-problems are genuinely
independent rather than artificially decoupled and then reconciled.

**Building it at full scale hit a real memory-exhaustion failure**, worked through by testing
each hypothesis directly rather than trusting the most plausible-looking one:

- Not a multiprocessing deadlock - `free -h` showed swap thrashing, confirmed by killing the
  stuck processes and watching memory free immediately.
- Not `Pool.imap`'s laziness - a minimal repro showed `imap` consumes its entire input generator
  immediately regardless of worker speed.
- Not `worker_count` too high - a memory-aware cap was a real, necessary fix on its own, but
  still thrashed; the memory estimate itself undercounted the real cost by ~10x.
- Not pyarrow's import footprint, despite matching a known Python/multiprocessing gotcha:
  removing pyarrow from the picture entirely (2380 MB peak worker RSS with it vs. 2254 MB
  without) barely moved the number.
- **The real cause**: the main process decoded the *entire* 60000-example dataset into
  `(tuple-of-784-floats, label)` pairs before any subsampling - 47 million boxed Python floats,
  duplicated into every forked worker.

**The fix** (measured before/after on identical work): `mnist_data.load_mnist_labels` (label
bytes only) plus `load_mnist_records_at_indices` (direct seek, decodes only requested examples)
replace eager full-dataset decoding, so no process ever holds the full dataset decoded at once -
peak worker RSS dropped from 2254.7 MB to 374.4 MB (~6x). Verified end to end: full 60000-image,
10-class training, 8 workers - **31.2 minutes wall-clock, memory stable, 89.4% held-out test
accuracy**.

## softmax/cross-entropy re-alignment

An audit against canonical literature (Rumelhart/Hinton/Williams's generalized delta rule;
Nielsen's BP1-BP4 equations, which `backprop_node.py`'s math matches essentially exactly) found
one deliberate deviation: `MultiClassBackpropClassifierNetwork` treats multi-class classification
as one-vs-rest (`class_count` independent sigmoid outputs, quadratic loss) rather than the
canonical softmax + cross-entropy treatment for a *mutually exclusive* target like digit
classification. The only reason recorded anywhere for one-vs-rest was incidental ("needed no
code changes") - an unexamined default, not a considered tradeoff.

Softmax's cross-node coupling (`a_i = e^z_i / Σ_j e^z_j`) only affects the forward pass - the
delta simplifies to `activation - target`, exactly as per-node-independent as one-vs-rest's own
delta, so the backward-pass plumbing needed zero changes. This made the fix purely additive: two
extension points (`BackpropLayer._node_cls`, `BackpropNetworkBase.output_layer_cls`), a new
`SoftmaxOutputNode`/`SoftmaxOutputLayer`, and `SoftmaxMultiClassBackpropClassifierNetwork` (a
single class-attribute override on top of `MultiClassBackpropClassifierNetwork`, which is kept
unchanged as a simpler reference point).

Measured on the full bundled UCI digits set (`demo_uci_digit_recognition.py`'s own architecture):

| | training accuracy | best epoch | held-out test accuracy |
|---|---|---|---|
| one-vs-rest (MSE) | 99.51% | 22/30 | 95.54% |
| softmax (cross-entropy) | 100.00% | 11/30 | 96.94% |

Softmax reached full training accuracy in about half the epochs and generalized slightly better
(one seeded run, not a statistically powered comparison, but real evidence it isn't worse, plus
the semantic correctness win: `predict_probabilities()` now genuinely sums to 1.0).
`EnsembleBackpropClassifierNetwork`'s own one-vs-rest design is a *different*, independently
motivated tradeoff (parallelizability across independent processes, which softmax's cross-output
coupling would undo) and is deliberately untouched by this re-alignment.

## binary cross-entropy for BackpropClassifierNetwork

The binary case was deferred above and investigated as a follow-up. Architecturally simpler than
softmax (a single output node's delta needs nothing from any sibling): `CrossEntropyOutputNode`
overrides `compute_output_delta` to `self.delta = self.value() - reference_value`, reusing the
same `_node_cls`/`output_layer_cls` hooks the softmax work added.

Measured against `test_backprop_training_pipeline.py`'s pinned XOR scenario, 10 seeds: quadratic
loss meaned 97.80%, cross-entropy meaned 91.87% - cross-entropy *underperformed* on every seed,
the opposite of softmax's clean win. The mechanism: cross-entropy's delta drops the `a(1-a)`
damping term quadratic loss has, so at a fixed learning rate its effective gradient magnitude is
larger and can overshoot. Sweeping `learning_rate` down for cross-entropy alone:

| learning_rate | mean training accuracy |
|---|---|
| 1.0 | 91.87% |
| 0.5 | 95.47% |
| 0.25 | 96.67% |
| 0.1 | 97.60% |

At `learning_rate=0.1`, cross-entropy matches quadratic's own tuned-rate mean (97.80%) - not
worse in principle, but not a drop-in replacement at this codebase's existing per-consumer tuned
hyperparameters either. `BinaryCrossEntropyBackpropClassifierNetwork` was built as a standalone
additive sibling regardless (genuinely useful, cheap to add); it was **not** extended to
`EnsembleBackpropClassifierNetwork`'s real-MNIST training as part of this work - that needed its
own dedicated retuning investigation, given the cost of each real training run and the open
question of whether this toy-problem finding transfers to MNIST's scale (see next entry).

## the ensemble/real-MNIST investigation

The deferred retuning question above, worked through in three steps, cheapest first:

**Step 1 - a cheap check found a bigger issue than expected.** Sampling real MNIST inputs
through a freshly-`randomize()`d `BackpropClassifierNetwork` at the ensemble's architecture
(`[16]` hidden, `dimension=784`) found hidden-layer `z` ranging -83 to +87, with **83.5% of
hidden activations already saturated** before any training - the exact sigmoid-saturation
pathology `MultiClassBackpropClassifierNetwork.randomize()` already fixed via fan-in-aware init,
but never applied to `BackpropClassifierNetwork` (which every ensemble sub-network uses). A
confound independent of loss function, dominating any loss-function comparison run on top of it.

**Step 2 - a cheap proxy sweep isolating init from loss function** (320 real MNIST examples, one
digit's balanced binary target, 5 epochs, 5 seeds):

| config | mean test accuracy |
|---|---|
| quadratic, current init | 82.75% |
| quadratic, fan-in-aware init | **93.75%** |
| cross-entropy, current init | 87.50% |
| cross-entropy, fan-in-aware init | 92.50% |

Fixing initialization alone closed an 11-point gap - a substantially bigger, more certain effect
than the loss-function switch (once init is fixed, quadratic and cross-entropy come out roughly
tied).

**Step 3 - real, full-scale validation** (actual 60000/10000 MNIST, `[16]` hidden,
`learning_rate=0.5`, 5 epochs):

| config | wall-clock | test accuracy |
|---|---|---|
| current init, quadratic (documented baseline) | ~31.2 min | 89.4% |
| **fan-in-aware init, quadratic** | 29.6 min | **96.01%** |
| fan-in-aware init, cross-entropy (untuned lr) | 33.5 min | 92.82% |

Fan-in-aware init alone: **+6.6 points** at the same wall-clock cost. Cross-entropy at the
quadratic-tuned learning rate trailed fan-in-aware-quadratic by 3.19 points - the same
undamped-gradient/overshoot pattern confirmed at all three scales tested (XOR, small proxy, full
MNIST) now.

**Decision:** `BackpropClassifierNetwork`'s init scheme (fixed via
`FanInAwareBackpropClassifierNetwork`, now what `EnsembleBackpropClassifierNetwork` is built
from) was the dominant, already-actionable finding - not the original loss-function question,
which is considered adequately answered (cross-entropy doesn't help at production's current
learning rate; retuning would cost its own real ~30-minute run for a gain no measurement here
exceeded fan-in-aware quadratic's result).

## Xavier/Glorot init: measured, not worth adopting

A follow-up audit asked whether Glorot & Bengio 2010's Xavier init (`limit =
sqrt(6/(fan_in+fan_out))`, tailored for sigmoid networks specifically) would beat the fan-in-only
scheme already adopted (`limit = 1/sqrt(fan_in)`, closer to LeCun 1998's practical
simplification). Measured on the same proxy as the ensemble investigation (320 examples, digit
3, 5 epochs, 5 seeds):

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

Canonical coefficient (α=0.9) across a learning-rate sweep, 5 seeds, 5 epochs:

| config | mean test accuracy |
|---|---|
| no momentum, lr=0.5 (baseline) | 93.75% |
| momentum=0.9, lr=0.5 | 91.75% |
| momentum=0.9, lr=0.25 | 90.75% |
| momentum=0.9, lr=0.1 | 91.50% |
| momentum=0.9, lr=0.05 | 91.25% |

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

## softmax on real full-scale MNIST

`SoftmaxMultiClassBackpropClassifierNetwork` had only been validated on small-scale UCI digits
(where it beat one-vs-rest, 96.94% vs. 95.54%). A feasibility check first (500-record timing:
4.03ms/iter one-vs-rest vs. 4.68ms/iter softmax, both comfortably fast; 1861 MB peak RSS for the
full decoded set) cleared the way for a real run - full 60000/10000 MNIST, `[16]` hidden,
`learning_rate=0.5`, 5 epochs, both trained sequentially (softmax's joint output can't be split
across processes the way the ensemble's ten binary classifiers can):

| | held-out test accuracy | training time |
|---|---|---|
| one-vs-rest | 92.75% | 31.4 min |
| softmax | 89.12% | 35.5 min |
| ensemble (fan-in-aware x10, documented baseline) | 96.01% | ~29.6 min |

The opposite result from the small-scale UCI comparison - softmax trailed at every epoch, and its
training accuracy wasn't even monotonic (dropped from epoch 4 to 5). Consistent with the same
mechanism confirmed three times already: softmax's `activation - target` delta drops the same
`a(1-a)` damping term binary cross-entropy's does, and this run used the quadratic-tuned
`learning_rate=0.5`, never retuned for softmax - the non-monotonic curve looks like overshoot,
not a representational disadvantage. **Decision:** not retuned further (the same cost/value
tradeoff the binary cross-entropy investigation already declined to spend on) - considered
adequately answered. `MultiClassBackpropClassifierNetwork` remains the better full-scale
one-network option at today's tuning; the ensemble remains the best of all three by a wide
margin.

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

## the Rust array core's 65-335x-slower-than-numpy finding was a debug-build artifact

[The production cutover plan](rust-production-cutover.md)'s own gating benchmark measured the
Rust core's naive per-op forward pass (a line-for-line port composed from individual `Array`
operator calls) as 65.7x-335.4x slower than numpy at this codebase's real layer size
(`dimension=784, hidden=16`), widening with batch size - the premise phase 0's entire fix-and-
gate structure was built around. Re-measuring it immediately before starting phase 0a's matmul
loop reorder surfaced the actual cause: `./cli setup`/`./cli build-rust` built the crate with a
plain `maturin develop` - a debug (unoptimized) build - and that debug build, not anything
inherent to the per-op composition approach, was almost the entire gap.

Same forward pass, same architecture, four build/loop-order combinations, batch sizes 1/8/32/128/512
(ratio = Rust wall-clock / numpy wall-clock; correctness held throughout, max elementwise
difference 1e-15 to 2e-14, float64 noise at every combination):

| batch | debug build, original loop order | release build, original loop order | release build, after 0a's loop reorder |
|---|---|---|---|
| 1 | 71.5x | 3.4x | 2.4x |
| 8 | 160.2x | 3.8x | 1.7x |
| 32 | 228.4x | 5.0x | 2.6x |
| 128 | 427.6x | 10.4x | 6.5x |
| 512 | 374.9x | 9.5x | 4.7x |

The debug-build column reproduces the plan's original 65.7x-335.4x figures closely, confirming
the cause. Under a release build - the only build any real deployment would ship - the naive
per-op composition was already only ~3-10x slower than numpy before any of phase 0's fixes, and
the matmul loop reorder alone (0a) brings it to ~2-7x, without yet fusing per-layer calls (0b).

**Decision:** `./cli build-rust`/`./cli setup` now build with `maturin develop --release` (a
one-line fix - see the `cli` script and [setup](setup.md)) - a benchmark's headline number is only
as good as the build it was measured against, and this one went unquestioned through two doc
revisions before being re-run.

## phase 0b: fusing closes some further ground, but matmul is now the binding constraint

With the release-build baseline corrected and 0a's loop reorder in, phase 0b fused each
`ArrayLayer` method into a single Rust call (`fused.rs` - `layer_forward`, `layer_forward_batch`,
`layer_output_delta`, `layer_hidden_delta`/`_batch`, `layer_accumulate_gradient`/`_batch`,
`layer_apply_accumulated_gradient`), eliminating the per-op FFI-crossing count entirely (one
Python-to-Rust call per layer method, not four to six). Re-measuring `layer_forward_batch` alone
against numpy, same architecture as every prior table here (`dimension=784, hidden=16`):

| batch size | numpy | fused Rust | ratio |
|---|---|---|---|
| 1 | 14.4 us | 27.7 us | 1.9x slower |
| 8 | 21.9 us | 45.1 us | 2.1x slower |
| 32 | 57.6 us | 135.1 us | 2.4x slower |
| 128 | 154.2 us | 666.4 us | 4.3x slower |
| 512 | 489.1 us | 2082.5 us | 4.3x slower |

Fusing closed some further ground over 0a alone (batch 128 was 6.5x slower after 0a alone;
4.3x after 0b) but not decisively - matmul, still a naive triple loop with no SIMD/blocking, is
now the dominant remaining cost at these batch sizes, and fusing doesn't touch it. A fuller
comparison - one whole mini-batch training step (`layer_forward_batch` + `layer_output_delta` +
`layer_accumulate_gradient_batch` + `layer_apply_accumulated_gradient` vs. the equivalent numpy
`ArrayLayer` method sequence, `dimension=784, hidden=10` - this codebase's real output-layer
shape) tells a more textured story than the forward-pass-only table above:

| batch size | numpy | fused Rust | ratio |
|---|---|---|---|
| 1 | 68.0 us | 30.5 us | **0.45x - Rust is faster** |
| 8 | 67.2 us | 64.6 us | ~parity |
| 32 | 108.3 us | 189.6 us | 1.75x slower |
| 128 | 201.5 us | 798.9 us | 4.03x slower |
| 512 | 1740.3 us | 3478.3 us | 2.00x slower |

(Correctness held throughout both tables - max weight difference after a full training step ≤
5e-16, float64 noise; every fused function checked against `array_layer.py`'s own `ArrayLayer`
methods directly, not an independently-written reference formula - see
`rust/indrajala_ml_array/tests/test_fused_layer_ops.py`.)

**The split result:** the fused Rust core beats numpy for per-example training (`batch_size=1`,
`ArrayLayer.learn`'s own call shape) but loses to it, by a widening margin, for realistic
mini-batch training (`batch_size >= 32`, `learn_batch`'s shape) - the opposite of what [the
production cutover plan](rust-production-cutover.md#risks-and-open-questions)'s own "risks and
open questions" guessed before this was measured (it expected batching to be the Rust core's
strength and per-example calls its weakness, on the theory that a single Python↔Rust call still
costs something no pure-Python-calling-numpy path pays - true, but overwhelmed at larger batches
by numpy's BLAS-backed matmul pulling further ahead of this crate's still-naive one). **Decision:
build phase 1 in full anyway.** The user clarified (2026-09-16) that adoption of the Rust core as
the production backend is unconditional, not gated on beating numpy first - numpy stays on
permanently as the benchmark comparison, and the `batch_size >= 32` gap above is tracked as
follow-on optimization work (SIMD, blocked/tiled matmul, threading), not a reason to withhold
`RustArrayMultiClassBackpropClassifierNetwork` or scope it down to the per-example path alone. See
[the production cutover plan](rust-production-cutover.md#the-decisive-finding-this-plan-has-to-answer-first)
("adoption is unconditional, not gated on this benchmark") for the full framing change.

## phase 2 tier 2: real per-example training is a genuine win at both scales measured

Good news the synthetic phase-0b benchmark only hinted at: both of this codebase's actual
production training paths (`demo_vectorized_uci_digit_recognition.py`,
`demo_vectorized_mnist_recognition.py`) train via `train_linear_classifier_network`, i.e.
`learn()`'s per-example (`batch_size=1`) path exclusively - checked directly, neither demo calls
`learn_batch`. That is exactly the regime phase 0b's benchmark found the fused Rust core *beats*
numpy in, not the `batch_size >= 32` regime it loses in. Tier 2 (docs/rust-production-cutover.md's
phase 2) measures this on real training runs, not synthetic per-step timings: full
`randomize()`-to-trained-accuracy runs, multiple independent seeds per class (the RNG mismatch
means seeds can't be matched across classes - see [the Rust core](rust-array-core.md#the-rng-exception) -
so seeds are independent draws, compared by distribution, not by seed-for-seed identity).

**UCI digits** (`[32]` hidden, `dimension=64`, same fixed `split_train_test(seed=0)` split
`demo_uci_digit_recognition.py` uses, `learning_rate=0.5`, `epochs=30`, 8 seeds each):

| network | train accuracy | test accuracy | wall-clock |
|---|---|---|---|
| numpy (`VectorizedMultiClassBackpropClassifierNetwork`) | mean 99.57%, stdev 0.07% | mean 96.62%, stdev 0.47% | 4.27s |
| Rust (`RustArrayMultiClassBackpropClassifierNetwork`) | mean 99.58%, stdev 0.03% | mean 96.90%, stdev 0.22% | 1.26s |

**3.40x faster**, with test accuracy distributions overlapping well inside each other's stdev band
- statistically indistinguishable, matching this codebase's own established
"correctness/accuracy first, then compare" standard.

**Real MNIST** (`[30]` hidden, `dimension=784` - `demo_vectorized_mnist_recognition.py`'s own
already-documented architecture, `learning_rate=0.5`, 1 real epoch over the full 60000-example
training set, 3 seeds each - fewer seeds than UCI digits purely because each one costs over ten
times as long):

| network | train accuracy | test accuracy | wall-clock |
|---|---|---|---|
| numpy | mean 93.13%, stdev 0.01% | mean 92.96%, stdev 0.14% | 15.1s |
| Rust | mean 92.86%, stdev 0.11% | mean 92.66%, stdev 0.17% | 11.6s |

**1.31x faster**, smaller than UCI digits' win - real MNIST's much larger `dimension=784` matmul
gives the naive triple loop more relative work per call even at `batch_size=1`, eating into the
per-call-overhead advantage that dominates at UCI digits' smaller `dimension=64` - but still a
genuine, positive speedup, with test accuracy within 0.3 points, comparable to each network's own
seed-to-seed spread. Both figures are honest, measured numbers from real training runs on real
data, not extrapolated from the synthetic phase 0b benchmark.

## build-flag tuning measured as a null

First candidate tried for closing the naive matmul's remaining `batch_size >= 32` gap (see
[structure](structure.md#possible-next-steps)): release-profile/codegen tuning, the cheapest
possible lever since it changes no algorithm - motivated directly by this same document's own
debug-vs-release discovery, which showed build configuration can matter more than assumed here.
Two variants measured against the same full mini-batch training-step benchmark phase 0b used
(`layer_forward_batch` + `layer_output_delta` + `layer_accumulate_gradient_batch` +
`layer_apply_accumulated_gradient`, `dimension=784, hidden=10`), each run twice to separate a
real effect from noise:

| variant | batch 1 | batch 8 | batch 32 | batch 128 | batch 512 |
|---|---|---|---|---|---|
| baseline (`--release`) | 0.39x, 0.50x | 0.99x, 1.09x | 1.79x, 1.79x | 3.28x, 3.77x | 2.23x, 1.80x |
| `+ lto=true, codegen-units=1` | 0.43x | 1.08x | 1.76x | 4.14x | 1.93x |
| `+ RUSTFLAGS="-C target-cpu=native"` (AVX2 available) | 0.37x, 0.52x | 1.14x, 1.06x | 1.55x, 1.63x | 3.88x, 3.70x | 2.15x, 2.09x |

(ratio = Rust wall-clock / numpy wall-clock; lower is better for Rust.) Every variant's numbers
fall inside the baseline's own run-to-run spread (e.g. batch 128 alone ranges 3.28x-4.14x across
runs of the *identical* binary) - neither `lto`/`codegen-units` nor `target-cpu=native` produced a
change distinguishable from noise, at any batch size. **Decision: not adopted** - the
`Cargo.toml`/`RUSTFLAGS` changes were reverted rather than kept for no measured benefit, the same
"correctness-validated, not adopted for performance reasons" treatment momentum/L2/Xavier-Glorot
got. Plausible reason: this crate is small enough (~900 lines) that a normal `--release` build's
default codegen-unit count already gives LLVM everything it needs to inline across module
boundaries, so `lto`/`codegen-units=1` had nothing further to unlock; `target-cpu=native`'s lack
of effect suggests the matmul inner loop (`linalg.rs`, indexed slice access,
`out_row[col] += a_value * b_row[col]`) isn't auto-vectorizing even with AVX2 available - the next
candidates (blocking, threading, explicit SIMD intrinsics) target that directly instead of hoping
the compiler finds it unassisted.

## cache-blocked matmul: a real, shape-dependent win, gated by size

Second candidate for the `batch_size >= 32` gap: cache-block `linalg.rs::matmul`'s 2D×2D case
over `row` and `k` (fixed-size slabs reused across multiple output rows before moving on),
instead of the plain `row -> k -> col` loop phase 0a left in place. Measured directly with raw
matmul calls (not the full training-step benchmark) at three shapes, to isolate the effect from
noise: two of this codebase's own real matmul shapes, plus one exaggerated size to confirm
blocking works at all before tuning it for this codebase's scale:

| shape | `b` size | unblocked | unconditionally blocked | ratio |
|---|---|---|---|---|
| `forward_batch`-like: `(512,784)@(784,16)` | ~100KB | 1.94ms | 2.21ms | **0.87x - a regression** |
| `accumulate_gradient_batch`-like: `(10,512)@(512,784)` | ~3.2MB | 1.28ms | 0.96ms | **1.33x faster** |
| exaggerated: `(2048,2048)@(2048,2048)` | 32MB | 5251ms | 2527ms | **2.08x faster** |

Blocking only helps once `b` (the operand that gets re-streamed once per output row in the
unblocked loop) is big enough to not already fit in cache - at `dimension=784, hidden=16`, `b` is
only ~100KB, already cache-resident, so blocking's extra bookkeeping was pure overhead with
nothing to relieve. **Fix, not abandonment**: gate blocking on `b`'s size
(`c1 * c2 * size_of::<f64>()`), with a 256KB threshold comfortably below a typical machine's L2
cache - below it, run the plain unblocked loop (verified back to baseline speed, not just
"not worse"); at or above it, block. Re-measured on the full training-step benchmark
(`dimension=784, hidden=10`, 3 runs): `batch_size=512`'s ratio improved from a 1.80x-2.23x range
(no blocking) to a 1.53x-1.74x range (size-gated blocking) - a real, reproducible ~15-25%
reduction in that batch size's gap, with `batch_size=1`/`8` unaffected (matches expectation -
`batch_size=1`'s `learn()` path never even reaches the 2D×2D matmul case; `layer_forward_batch`'s
own matmul stays under the threshold at this codebase's real `dimension=784` architecture
regardless of batch size, since `c1`/`c2` there are `dimension`/`hidden`, not `batch`). **Decision:
adopted** - both branches verified bit-identical to the previous single-path implementation (525
existing tests unchanged; blocking only restructures loop order, not summation order), and this is
a genuine, if modest, step in closing docs/structure.md's tracked follow-on gap, not a full
resolution of it. Threading and SIMD intrinsics remain open next steps for the rest of that gap.

## threaded matmul: a real further win, one real bug caught, one refinement rejected

Third candidate for the `batch_size >= 32` gap, on top of blocking above: split
`linalg.rs::matmul`'s 2D×2D case's output rows across `std::thread::scope` workers once there's
enough total work (`r1 * c1 * c2 >= 4,000,000`, a deliberately conservative floor) to plausibly
pay for thread spawn overhead. Row-splitting needs no cross-thread reduction (each worker owns
complete output rows end to end), so results are bit-identical to the single-threaded path
regardless of thread count - confirmed by all 525 existing tests passing unchanged.

**A real implementation bug caught before it shipped**: the first version called
`std::thread::available_parallelism()` on every matmul dispatch (to decide the thread count) -
measured directly at **~50 microseconds per call** (not cached by the standard library,
presumably a cgroup/proc filesystem read each time). Since this codebase's own actual per-call
matmul times are themselves in the tens of microseconds, this silently dominated everything:
`batch_size=1` on the full training-step benchmark went from Rust *beating* numpy (0.38x-0.50x) to
Rust *losing badly* (2.20x-2.31x) - a regression at the one batch size that mattered most,
caught only because the full benchmark suite (not just the new matmul's own isolated shapes) was
re-run before trusting the change. Fixed by caching the result once behind a `OnceLock` -
`available_parallelism_cached()` - and by checking the (cheap) flops threshold *before* ever
reading it, so tiny matmuls never pay even the cached read's small remaining cost.

**A refinement that looked right in isolation but wasn't, measured against the real target
metric**: an isolated raw-matmul-only benchmark at three shapes -

| shape | `b` size | single-threaded (blocked) | + threading |
|---|---|---|---|
| `forward_batch`-like: `(512,784)@(784,16)` | ~100KB | 1.94-2.21ms | **0.87-1.07ms - ~2x faster** |
| `accumulate_gradient_batch`-like: `(10,512)@(512,784)` | ~3.2MB | 0.94-1.09ms | 0.91-1.16ms - no clear change |
| exaggerated: `(2048,2048)@(2048,2048)` | 32MB | 2.35-2.67s | 2.06-2.68s - no real change |

- showed the small-`r1` shape (`r1=10`) getting no benefit from 8 threads doing barely more than
one row each, which looked like the natural fix: require a minimum rows-per-thread (32) before
threading engages at all, so `r1=10` falls back to single-threaded. That change *did* clean up
the isolated shape's own number - but re-measured against the actual target metric (the full
mini-batch training-step benchmark, not one matmul shape in isolation), it made
`batch_size=512`'s real ratio *worse* (1.24x-1.79x, back up from where unrestricted threading had
it), reproducibly across three separate runs. **Not adopted** - the isolated shape's own
regression didn't generalize to the composite workload it's actually part of, so the simpler
`min(available_parallelism, 8, r1)` thread count (no rows-per-thread floor) was kept instead. A
concrete instance of this codebase's own "measure, don't assume" standard applying recursively -
even a measurement-driven refinement of an earlier measurement needs checking against the real
metric, not just the proxy that motivated it.

The exaggerated `2048x2048` shape's lack of improvement from threading was itself directly
verified as a memory-bandwidth ceiling, not a threading bug: `os.times()`-measured CPU-time-to-
wall-time ratio was **7.17x** (near-perfect use of 8 cores) while wall-clock stayed flat - the
matrix (32MB) doesn't fit any per-core cache, so 8 cores contending for the same DRAM bandwidth
gain little from parallelism regardless of how well the threads themselves are balanced. Not a
concern for this codebase's actual matrix sizes (`dimension<=784`), included here only because
it's what surfaced the effect clearly enough to identify it.

**Net result, full training-step benchmark** (`dimension=784, hidden=10`, three runs):
`batch_size=512`'s ratio improved from blocking-alone's 1.53x-1.74x range to **1.19x-1.21x** - a
further, real, reproducible ~25-30% reduction on top of blocking's own earlier win.
`batch_size=1`/`8`/`32` are unaffected (`batch_size=1`'s `learn()` path never reaches the 2D×2D
matmul case at all; the other two stay below the flops threshold at this architecture).
**Decision: adopted**, with the rows-per-thread refinement explicitly rejected per the
measurement above. Explicit SIMD intrinsics remain the one open candidate left for the rest of
this gap.

## explicit SIMD intrinsics: a real further win, with fused multiply-add kept consistent across every path

Fourth and (per docs/rust-production-cutover.md's original reasoning) last candidate for the
`batch_size >= 32` gap: `linalg.rs::matmul_2d_row_range`'s innermost accumulate step
(`out_row[col] += a_value * b_row[col]`) is a textbook AXPY pattern, vectorizable 4 `f64` lanes at
a time on this machine's AVX2. Rather than rely on LLVM's autovectorizer opportunistically doing
this (which the earlier `target-cpu=native` build-flag experiment, measured as a null, gave no
evidence was happening reliably), the accumulate step was factored into one shared helper
(`axpy_row`, called from both the blocked and unblocked loops - previously duplicated 3-line
loops) with an explicit AVX2 path using `std::arch::x86_64::_mm256_fmadd_pd`, runtime-gated by
`is_x86_feature_detected!("avx2", "fma")` with a portable scalar fallback for machines without it.

**The design question this raised**: `_mm256_fmadd_pd` is a genuine fused multiply-add (one
rounding for `a*b+c`), not a separate multiply-then-add (two roundings) like the old `+=` loop.
Used naively, that would have made the AVX2 path produce different last-bit results than the
scalar fallback - silently breaking the bit-identical invariant that blocking (above) and
threading (above) were both explicitly built to preserve, and that a machine without AVX2 would
then train slightly different numbers than one with it, given enough steps. **Resolved by making
FMA the accumulate semantics on *every* path**, not just the new one: the scalar fallback
(`axpy_row_scalar`) now uses `f64::mul_add` (Rust's portable FMA, lowering to the same hardware
instruction when available) instead of `+=`, so the AVX2 path is a wider version of what the
scalar path already does, not a numerically different one.

**Verified directly, not just argued**: this crate has no native `cargo test` support (`pyo3`'s
`extension-module` feature doesn't link libpython, so a `cargo test` binary fails at link time -
this codebase's whole correctness suite runs through pytest instead, which is why nothing
previously caught this class of regression). Checked empirically instead: built once with the
AVX2 path forced off, captured `matmul` output as exact IEEE-754 bit patterns (`struct.pack("<d",
...).hex()`, not `pytest.approx`) across five shapes spanning both the blocked/unblocked and
remainder/exact-multiple-of-4 boundaries, including this codebase's real `batch_size=512` shape;
rebuilt with the AVX2 path restored and re-captured - **byte-for-byte identical across all five
shapes**. All 845 top-level tests (525 crate-level) also pass unchanged on the final build.

**Net result, full training-step benchmark** (`dimension=784, hidden=10`, `batch_size=512`,
`learn_batch`, three runs, AVX2-forced-off vs AVX2-on): per-step wall time dropped from
~31.4-33.1ms to ~25.7-26.6ms, roughly a **17-20% reduction** on top of blocking+threading's
earlier combined win. Measured directly against numpy on the same shape (three trials): the
Rust/numpy ratio moved to **0.84x-1.04x** - Rust now matches or beats numpy at `batch_size=512` in
most trials, versus the 1.19x-1.21x (Rust slower) recorded after threading alone. **Decision:
adopted** - this was the last candidate docs/rust-production-cutover.md identified for closing
the `batch_size >= 32` gap, and unlike blocking/threading it closes it in Rust's favor rather than
just narrowing it.

**AVX-512 checked and ruled out - not a further candidate on this machine**: `/proc/cpuinfo`/
`lscpu`'s flags list `avx`/`avx2` but no `avx512*` variant. This machine's CPU (AMD Ryzen 7 3700U,
Zen+/"Picasso", a 2019 mobile part) predates AMD's AVX-512 support entirely - that arrived with
Zen 4 in 2022. AVX2's 256-bit registers (4 `f64` lanes/instruction) are this hardware's actual
ceiling, so `axpy_row_avx2_fma` is already using the widest vector width available here; there is
no `_mm512_fmadd_pd` upside to chase on this machine. (Would need re-checking on different
hardware - `axpy_row`'s runtime `is_x86_feature_detected!` gate means a future AVX-512-capable
machine falls back to the AVX2 path today, not a crash, but also not the extra width until an
AVX-512 path is added.)

## SIMD for the matvec production path: a bigger win than the batch>=32 work it followed

The `batch_size >= 32` work above targeted `linalg.rs::matmul`'s 2D×2D case - a shape no current
production path actually uses. Once that work closed out, a stage-0-style measurement (same
"quantify before writing code" discipline as docs/rust-production-cutover.md's own "the decisive
finding") checked whether the *other* two matmul cases (`Matrix @ Vector`, `Vector @ Matrix`) -
untouched by any of the blocking/threading/SIMD work above - had any real headroom left, rather
than trusting `linalg.rs`'s own prior doc comment, which asserted (unverified) that they "never
exercised... at a size where it would matter."

**That assumption was wrong.** At the real production shape (`dimension=784, hidden=16`), a
microbenchmark (200,000 reps, warmed up first) found `W @ x` (12,544 FLOPs) cost **~10.2us**, of
which **~97% was the matmul itself**, not PyO3 call overhead (a bare 1-element-read FFI round
trip measured ~0.29us) - and this one matvec call was **~3.5x slower than numpy** at this exact
shape, accounting for ~97% of a fused `layer_forward` call's cost and ~27-29% of a full `learn()`
step's wall-clock. The `Vector @ Matrix` case (unused by the current class design, per the
interface subset's own note) wasn't checked at production scale for the same reason it was never
optimized before: nothing calls it today.

**Why the matrix@vector case couldn't just reuse `axpy_row`.** `axpy_row` vectorizes across the
*output* dimension while keeping the reduction over `k` strictly sequential - that's what makes
it bit-identical regardless of which path runs. A per-row dot product's reduction dimension *is*
the dimension SIMD would vectorize, so there's no grouping of 4 parallel lanes that's also
bit-identical to a naive left-to-right sequential sum (float64 addition isn't associative - a
different grouping is a different value, typically by ~1 ULP). Resolved by picking one canonical
grouping - 4 interleaved partial sums (lane `j` accumulates indices `j, j+4, j+8, ...`), combined
pairwise at the end - and using that *same* grouping in both the scalar fallback
(`dot_product_scalar`) and the new AVX2+FMA path (`dot_product_avx2_fma`), via a shared
`combine_lanes`/`dot_product_tail` so the two paths can't accidentally diverge. This does change
the *value* matmul produces from the old naive-sequential-sum baseline (last-few-ULPs noise) - the
same category of already-accepted risk as numpy's own internal reduction order not matching
Python's sequential sum (see "summation-order rounding" in
[vectorized array-based classes](vectorized-array-classes.md#numerical-parity-validation)), not a
new one, and every parity check against numpy/the pure-Python reference already tolerates it via
`rtol`, not exact equality. The `Vector @ Matrix` case needed no new reduction logic at all - it
turned out to be structurally identical to `axpy_row`'s own row-scaling accumulate (one output
"row" instead of many, `b`'s stride over `k` staying row-contiguous), so it was wired straight
through the existing, already-bit-identical `axpy_row` instead.

**Verified the same way as the earlier SIMD work**: built once with the AVX2 path forced off,
captured `matmul`'s `Matrix @ Vector` output as exact IEEE-754 bit patterns across 7 shapes
(spanning exact-multiples-of-4, remainders, and a 1x1 degenerate case, including the real
`16x784`/`10x16` production shapes); rebuilt with AVX2 restored and re-captured - **byte-for-byte
identical across all 7 shapes**. All 845 tests (525 crate-level) pass unchanged.

**Net result:**

| measurement | before | after |
|---|---|---|
| raw matvec, `16x784 @ 784` (production shape) | ~10.2us | **~4.3us** (2.4x faster) |
| Rust/numpy ratio, same shape | 3.5x slower | **1.48x slower** |
| full `learn()` step | ~73-75us | ~67us (~9% faster) |
| UCI digits training run, Rust/numpy | 3.40x | **~3.6x** (3 trials: 3.53x-3.64x) |
| real MNIST training run, Rust/numpy | 1.31x | **~2.6x** (5 trials: 2.59x-2.66x) |

Test accuracy stayed statistically indistinguishable (UCI digits: 96.94% numpy vs. 96.66% Rust;
real MNIST: 92.88% numpy vs. 93.18% Rust - both within the same RNG-mismatch noise band this
codebase's other Rust-vs-numpy comparisons already show, not a regression). The real-MNIST result
is the more striking one: `dimension=784, hidden=30` is exactly the shape where this matvec
dominates most, and the Rust/numpy ratio nearly doubled. **Decision: adopted.** Unlike the
`batch_size >= 32` work above (which targeted a shape no production path uses), this closes a gap
in the shape *every* production training step actually runs.

## Adam: tuned-XOR measurement (stage 3 of the Adam optimizer workplan)

[`AdamBackpropClassifierNetwork`](adam-optimizer.md) (stage 2, correctness-validated only) measured
for the first time against a real training run: the same pinned XOR scenario
`test_backprop_training_pipeline.py` uses (`BackpropClassifierNetwork([8], 2,
square_bounds(10.0))`, 300 examples, 100 epochs), 10 (data-generation seed, weight-init seed) pairs,
`beta1`/`beta2`/`epsilon` at their Kingma & Ba defaults. Note: `docs/adam-optimizer.md`'s original
"same scenario momentum's own baseline used" framing for this stage doesn't hold up - momentum's
own baseline was actually measured on the 320-example real-MNIST proxy, not XOR (see "momentum:
measured, not worth adopting" above); this stage instead reuses the pinned XOR scenario the
binary-cross-entropy and ReLU investigations used, which is what `test_backprop_training_pipeline.py`
actually pins.

At the demo-tuned `learning_rate=1.0` every other sibling here was first measured at:

| config | mean training accuracy | stdev |
|---|---|---|
| sigmoid, no Adam, lr=1.0 (tuned baseline) | 97.80% | 0.92% |
| Adam, lr=1.0 (untuned) | 58.27% | 9.09% |

Adam collapses badly at this rate - worse than cross-entropy's or ReLU's own untuned-rate dips.
Mechanism: Adam's effective per-step move is `~learning_rate` regardless of gradient magnitude
(the `m_hat / (sqrt(v_hat) + epsilon)` term is normalized to roughly unit scale once `t` is a few
steps in), so a rate tuned for undamped sigmoid+quadratic SGD is far too large once every step is
already near that scale by construction. Sweeping `learning_rate` down for Adam alone (still 10
seeds):

| learning_rate | mean training accuracy | stdev |
|---|---|---|
| 1.0 | 58.27% | 9.09% |
| 0.5 | 71.73% | 7.78% |
| 0.25 | 75.17% | 1.72% |
| 0.1 | 90.73% | 10.49% |
| 0.05 | 97.97% | 1.26% |
| 0.01 | 98.63% | 0.67% |

`lr=0.01` and `lr=0.05` both reach or exceed the sigmoid baseline's mean, with a smaller spread -
re-measured at 15 seeds to check this wasn't the same small-sample mirage momentum's own re-test
caught:

| config (n=15) | mean training accuracy | stdev |
|---|---|---|
| sigmoid, no Adam, lr=1.0 (tuned baseline) | 97.33% | 1.66% |
| Adam, lr=0.05 | 98.09% | 1.14% |
| Adam, lr=0.01 | 98.38% | 0.75% |

The direction holds at n=15: both retuned Adam configs beat the sigmoid baseline's mean by
~0.8-1.0 points *and* have a visibly tighter spread (0.75-1.14% stdev vs. 1.66%) - one sigmoid
seed (92.33%) is the kind of outlier Adam's per-parameter normalization looks to be damping out.
The margin is still small relative to this toy problem's ~97-99% ceiling, so this reads as a real
but modest edge, not a dramatic win.

**Interpretation:** unlike momentum (hurts at its canonical value, flat-to-harmful once retuned)
and unlike ReLU (loses badly untuned, then *clearly* wins retuned), Adam lands in between - it
needs its own small learning rate to avoid actively hurting (the same "no default is safe without
retuning" pattern cross-entropy/ReLU/momentum all showed at `lr=1.0`), and once retuned it's a
modest, consistent-direction improvement in both mean and variance, not a null the way momentum's
retuned sweep was. **Decision:** kept as a real, adopted capability, consistent with stage 2's
default-`beta1`/`beta2`/`epsilon` posture - but, like ReLU and cross-entropy, not a drop-in
replacement for sigmoid+quadratic's own tuned `learning_rate` without retuning. The real-MNIST-proxy
batch-size sweep (stage 4) is where Adam's actual per-parameter-adaptive-rate selling point -
smoothing per-example-noisy gradients, specifically at `batch_size=1` where momentum failed - has
room to show a larger effect than this small, already near-ceiling toy problem could.

## Adam under batch size: a much bigger, cleaner win (stage 4 of the Adam optimizer workplan)

The real-MNIST-proxy batch-size sweep the previous entry deferred to: same scale as the
momentum-under-mini-batch-gradients investigations (fan-in-aware init, `[16]` hidden, 5 epochs,
`batch_size` in 1/8/32/128, 10 seeds each), on a freshly-built 320-example real-MNIST digit-3
proxy (200 positive/200 negative examples, an 80/20 train/test split, fixed once rather than
resampled per seed - not the exact same proxy instance the original momentum sweeps used, since
that script was never committed, but the same construction procedure/scale). Sigmoid (no Adam)
and Adam run in the same script against the same proxy and seeds for a directly paired comparison,
crossed with both learning-rate regimes the earlier momentum sweep used - a fixed base rate, and
that rate linearly scaled by `batch_size` (`learning_rate = base_lr * batch_size`) - rather than
trusting the earlier sweep's own numbers out of context:

| config | batch_size=1 | batch_size=8 | batch_size=32 | batch_size=128 |
|---|---|---|---|---|
| sigmoid, fixed lr=0.5 | 89.62% ± 3.87% | 87.00% ± 1.88% | 83.62% ± 2.79% | 71.50% ± 13.33% |
| sigmoid, scaled lr=0.5×batch | 89.62% ± 3.87% | 87.62% ± 2.97% | 88.88% ± 2.24% | 57.12% ± 12.82% |
| Adam, fixed lr=0.01 | 90.50% ± 2.65% | 90.38% ± 1.87% | 89.38% ± 1.35% | **86.75% ± 1.05%** |
| Adam, scaled lr=0.01×batch | 90.50% ± 2.65% | 84.75% ± 12.95% | 76.00% ± 18.31% | 50.00% ± 3.95% |

The sigmoid rows are a genuine reproducibility check, not a rerun of the same numbers: on an
independently-constructed proxy subset, they qualitatively reproduce the
[learning-rate-vs-batch-size follow-up](#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)'s
own shape - a fixed rate degrades steadily as batch size grows (fewer update steps/epoch), scaling
the rate compensates at `batch_size=32`, and scaling diverges outright at `batch_size=128`
(high-variance collapse).

Adam's numbers are a materially bigger, cleaner result than stage 3's modest XOR edge:

- **A fixed learning rate barely degrades Adam at all across two orders of magnitude of batch
  size** (90.50% -> 86.75%, `batch_size=1` to `128`), while sigmoid's fixed rate collapses over the
  same range (89.62% -> 71.50%, with the variance exploding to ±13.33% as training destabilizes).
  At `batch_size=128` specifically, Adam beats *every* sigmoid regime tried, fixed or scaled
  (86.75% vs. 71.50%/57.12%), with roughly a tenth of the variance (±1.05% vs. ±13.33%/±12.82%).
  This is Adam's selling point showing up for real - not primarily at `batch_size=1`'s
  per-example-noisy regime as the workplan expected to look first (Adam and sigmoid are close
  there, 90.50% vs. 89.62%), but at the large-batch end, where sigmoid/momentum's shared weakness
  (needing a compensating learning-rate bump for fewer update steps/epoch) doesn't apply to Adam at
  all.
- **Scaling Adam's rate with batch size - the fix that helps sigmoid at `batch_size=32` - actively
  destroys Adam instead.** By `batch_size=128` (`lr=0.01*128=1.28`), Adam collapses to a
  coin-flip 50.00% with high, unstable variance across seeds. This directly answers the workplan's
  flagged open question ("whether Adam's adaptive per-parameter scaling interacts with
  `batch_size=128`'s linear-scaling-rule divergence"): yes, and the interaction is that the linear
  scaling rule is actively the *wrong* prescription for Adam, not merely unnecessary. Mechanism,
  consistent with stage 3's own finding: Adam's per-step move is already `~learning_rate` in scale
  regardless of gradient magnitude or batch size (the `m_hat/(sqrt(v_hat)+epsilon)` term stays
  roughly unit-scaled once `t` is a few steps in), so there's no "fewer, larger raw-gradient steps"
  problem for the linear scaling rule to compensate for in the first place - scaling `learning_rate`
  by `batch_size` just makes an already-appropriately-sized step 128x too large.

**Decision:** Adam should be used with a **fixed** `learning_rate` across batch sizes, the opposite
prescription from SGD/momentum's linear scaling rule - and, at this proxy's scale, that fixed rate
gives Adam a real, substantial robustness advantage over sigmoid at every batch size tested, most
dramatically at `batch_size=128` where sigmoid's training destabilizes regardless of which
learning-rate regime is used. This is the clearest positive result the Adam workplan has produced
so far, and directly validates the "worth its own measurement" rationale in
[Adam optimizer](adam-optimizer.md)'s opening motivation: unlike momentum, Adam's per-parameter
adaptive scaling genuinely does something SGD-with-momentum's single shared velocity term
couldn't, once gradients come from batches rather than single examples.
