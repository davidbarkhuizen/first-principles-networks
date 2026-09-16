# research and analysis: multi-class architecture and loss functions

[← back to research and analysis](research-and-analysis.md)

The investigation thread around how multi-class classification is structured (one-vs-rest vs.
softmax, ensembles vs. a shared hidden layer) and which loss function each uses - see [research
and analysis](research-and-analysis.md) for what this collection of docs is for.

## parallelizing MNIST training

Training `MultiClassBackpropClassifierNetwork` on full real MNIST (60000 images, 784 dims, pure
Python) benchmarked at ~12.5 minutes/epoch on this 8-CPU machine - the question was whether
`multiprocessing` could cut that meaningfully.

### rejected: data-parallel synchronous weight averaging

**Rejected: data-parallel synchronous weight averaging.** Shard the data 8 ways, train 8 replicas
of the same network for one epoch, average their weights, broadcast back. Measured (2000-image
subset, 2 epochs, 8 workers): only ~2x speedup (each epoch re-shipped the entire shard over IPC),
and a real accuracy drop (71.3% vs. 92.7% train) - averaging the weights of several replicas of a
*nonlinear* network isn't the same as averaging the functions they learned, and synchronizing
more often to fix that trades the accuracy loss back for less parallelism. No setting of that
dial was clearly good on all three axes (speed/accuracy/sync frequency), so this was dropped
rather than tuned further.

### adopted: an ensemble of independently-trained one-vs-rest classifiers

**Adopted: an ensemble of independently-trained one-vs-rest classifiers.** Rather than sharing a
hidden layer, train 10 completely independent `BackpropClassifierNetwork`s, one per digit, each
on its own small class-balanced binary dataset - genuinely independent sub-problems, so there is
no synchronization step of any kind. Measured on 3 digits (0, 1, 7): 98.21% 3-way ensemble argmax
accuracy, comparable to or better than the shared-hidden-layer approach on the full 10-way
problem. Full-scale result: **~26 minutes wall-clock** (10 jobs, 8 parallel workers, 2 rounds)
vs. ~2-2.5 hours for the original single-network plan - a ~5-6x speedup *alongside* better
accuracy, the opposite tradeoff the rejected approach hit, because the sub-problems are genuinely
independent rather than artificially decoupled and then reconciled.

### a real memory-exhaustion failure at full scale

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

### step 1: a cheap check found a bigger issue than expected

**Step 1 - a cheap check found a bigger issue than expected.** Sampling real MNIST inputs
through a freshly-`randomize()`d `BackpropClassifierNetwork` at the ensemble's architecture
(`[16]` hidden, `dimension=784`) found hidden-layer `z` ranging -83 to +87, with **83.5% of
hidden activations already saturated** before any training - the exact sigmoid-saturation
pathology `MultiClassBackpropClassifierNetwork.randomize()` already fixed via fan-in-aware init,
but never applied to `BackpropClassifierNetwork` (which every ensemble sub-network uses). A
confound independent of loss function, dominating any loss-function comparison run on top of it.

### step 2: a cheap proxy sweep isolating init from loss function

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

### step 3: real, full-scale validation

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

### decision

**Decision:** `BackpropClassifierNetwork`'s init scheme (fixed via
`FanInAwareBackpropClassifierNetwork`, now what `EnsembleBackpropClassifierNetwork` is built
from) was the dominant, already-actionable finding - not the original loss-function question,
which is considered adequately answered (cross-entropy doesn't help at production's current
learning rate; retuning would cost its own real ~30-minute run for a gain no measurement here
exceeded fan-in-aware quadratic's result).

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
