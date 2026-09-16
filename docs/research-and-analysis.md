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
`rust/perceptron_array/tests/test_fused_layer_ops.py`.)

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
