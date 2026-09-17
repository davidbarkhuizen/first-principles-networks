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

### step 4: an array/Rust-backed ensemble, and the parallel-vs-serial question

**Step 4 - vectorizing the ensemble itself**, once every other multiclass sibling in this
codebase's history had already been array/Rust-ported (see [backprop sibling
measurements](research-backprop-siblings.md)) but the ensemble - this codebase's own
best-performing, production-facing real-MNIST capability - still had no vectorized counterpart at
all. `ArrayBackpropClassifierNetwork`/`RustArrayBackpropClassifierNetwork` (a new single-output
array-backed network line, since every prior array port had only ever needed the existing
`class_count`-wide multiclass line) and `EnsembleArrayBackpropClassifierNetwork`/
`EnsembleRustArrayBackpropClassifierNetwork` (thin wrappers mirroring
`EnsembleBackpropClassifierNetwork`'s own composition) were built and parity-tested against the
per-node `FanInAwareBackpropClassifierNetwork` reference - see [an array-based ensemble
sibling](../design-docs/ensemble/ensemble-array-layer.md) for the full design.

The Rust port was built unconditionally, not gated behind the numpy stage's own wall-clock
verdict - per [the Rust production cutover](../architecture/rust-production-cutover.md)'s own
"unconditional production backend" precedent.

**Per-classifier wall-clock** (fused single-output layer, `dimension=784, hidden=[16]`, numpy vs.
Rust): a much smaller gap than every other sibling in this codebase's history (30-800x+) -
Rust beats numpy by only 1.79x at `batch_size=1` (the ensemble's own training shape), and the two
are roughly at parity by `batch_size=32` and beyond (0.81x-1.00x). A single-node output layer has
very little real matmul work per call, so both backends' fixed per-call overhead (Python-level
dispatch for numpy, the PyO3 call boundary for Rust) dominates the total cost far more than it
does for a `class_count`-wide output layer - see [possible next
steps](../project/structure.md#possible-next-steps) for concrete follow-ups this suggests.

**The real question - does vectorizing change how this codebase should train the ensemble at
all**, measured directly at real MNIST scale (`[16]` hidden, `learning_rate=0.5`, 5 epochs - the
same configuration step 3's own documented baseline used), across every training-path
combination this raised:

| configuration | wall-clock | vs. documented baseline | test accuracy |
|---|---|---|---|
| per-node parallel (documented baseline, step 3) | 29.6 min | 1.0x | 96.01% |
| **array/Rust, parallel** (`train_ensemble_parallel_from_indices`) | **21.4s** | **83.0x faster** | 96.07% |
| array/numpy, parallel (`train_ensemble_parallel_from_indices`) | 49.8s | 35.7x faster | 96.39% |
| array/Rust, serial (`train_ensemble_serial_from_indices`) | 69.1s | 25.7x faster | 95.96% |
| array/numpy, serial (`train_ensemble_serial_from_indices`) | 113.8s | 15.6x faster | 96.05% |

Every array/Rust configuration lands within seed-to-seed noise of the documented 96.01% accuracy
(95.96%-96.39%), the usual RNG-stream caveat applying (numpy's and Rust's own draws are never
seed-comparable to Python's `random`, or to each other - see [the RNG
exception](../architecture/rust-array-core.md#the-rng-exception)).

Two real findings, neither assumed going in:

- **`indrajala_ml_array.Array` does not support pickling** (confirmed directly:
  `pickle.dumps` raises `TypeError`) - initially a real blocker for a Rust-backed
  `multiprocessing.Pool` worker returning its trained snapshot, closed by converting a worker's
  return value to plain, always-picklable nested lists before it crosses the process boundary
  (`ensemble_train._picklable_snapshot`), reconstructed back into `indrajala_ml_array.Array` on
  the collecting side (`RustArrayBackpropClassifierNetwork.restore()`'s own tolerance for
  receiving either representation). Once closed, the *existing*
  `train_ensemble_parallel_from_indices` trains a Rust-backed ensemble unchanged - no new public
  training function was needed for the parallel case.
- **Multiprocessing is still worth keeping, even after vectorization** - the proposal's own
  single most consequential open question. Serial (single-process) training is dramatically
  faster than the old per-node path (15.6x-25.7x), but parallel dispatch still beats serial for
  both backends (numpy: 2.3x faster parallel vs. serial; Rust: 3.2x faster parallel vs. serial) -
  parallelism and vectorization compound rather than substitute. `ensemble_train.py`'s
  `multiprocessing.Pool` machinery (the memory-aware worker-count capping, the index-based record
  loading fix) remains worth its complexity.

**Decision:** array/Rust-backed, parallel-trained (`train_ensemble_parallel_from_indices` with
`classifier_cls=RustArrayBackpropClassifierNetwork`) is the new recommended path for this
architecture - 83x faster than the documented per-node baseline, at statistically indistinguishable
accuracy.

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

## an array-based softmax sibling: the retune the per-node investigation declined to spend on

The per-node softmax investigation above explicitly declined to retune `learning_rate` for softmax
at real-MNIST scale, citing the cost of a ~30+-minute training run. `SoftmaxArrayLayer`/
`SoftmaxVectorizedMultiClassBackpropClassifierNetwork` (numpy) and `SoftmaxRustArrayLayer`/
`SoftmaxRustArrayMultiClassBackpropClassifierNetwork` (Rust-matmul-backed, output-layer-only - the
Rust core's operation set needed a new row-wise max/sum-and-normalize primitive, `array_softmax`)
remove that excuse: a sweep that was too expensive to run once becomes cheap to run many times.

**Part A - wall-clock**, one mini-batch training step (output-layer shape `dimension=784,
class_count=10`, per-node vs. numpy, median of 30 timed reps):

| batch size | per-node (us/example) | numpy (us/example) | per-node/numpy |
|---|---|---|---|
| 1 | 4127.89 | 54.89 | 75.2x |
| 8 | 2864.95 | 8.13 | 352.4x |
| 32 | 2663.63 | 4.27 | 623.7x |
| 128 | 2642.41 | 2.42 | 1089.7x |
| 512 | 2634.65 | 3.06 | 862.1x |

**75.2x-1089.7x faster per example than the per-node path.**

**Part B - reproduce both existing results at array speed, then spend the now-cheap retune.** UCI
digits (`[32]` hidden, `learning_rate=0.5`, 30 epochs) reproduces the per-node win:

| | training accuracy | held-out test accuracy |
|---|---|---|
| one-vs-rest (numpy) | 99.58% | 96.66% |
| softmax (numpy) | **100.00%** | **98.33%** |

Real MNIST at the same untuned rate reproduces the per-node loss too (89.77%/90.16% vs.
one-vs-rest's 93.71%/93.01%), confirming it isn't a per-node-path artifact before spending the
retune. Sweeping `learning_rate` down for softmax alone finds a real peak around 0.01-0.05, not a
monotonic trend run off the sweep's edge; confirmed at 5 seeds against the untuned baseline's own
tuned rate:

| | seed accuracies | mean | stdev |
|---|---|---|---|
| one-vs-rest @ 0.5 | 93.01%, 93.02%, 92.40%, 93.25%, 92.55% | 92.85% | 0.36% |
| softmax @ 0.01 (retuned) | 94.17%, 94.10%, 93.67%, 93.70%, 93.61% | **93.85%** | 0.26% |

**A real, decisive win at real-MNIST scale** - softmax's retuned margin (+1.00 point) exceeds
either side's own seed-to-seed stdev, and every one of softmax's 5 seeds beats every one of the
baseline's 5 seeds (non-overlapping ranges). This reverses the untuned per-node investigation's
"remains the better full-scale one-network option" conclusion - scoped, as that entry always was,
to *untuned* softmax at the quadratic-tuned rate, not softmax generally. The ensemble (96.01%)
remains the best of all three real-MNIST options by a wide margin; this result is about the
single-network comparison specifically.

**Decision: adopted.** Softmax needs its own tuned learning rate at real-MNIST scale, never a
drop-in replacement for one-vs-rest at existing hyperparameters - but once retuned, it's a genuine,
decisive win at both scales checked.

## an array-based binary cross-entropy sibling: the retune the per-node investigation deferred

"binary cross-entropy for BackpropClassifierNetwork" above found cross-entropy *matches*, not
beats, quadratic loss on toy XOR once retuned (97.60% at `learning_rate=0.1` vs. quadratic's
97.80%), and explicitly deferred extending that to `EnsembleBackpropClassifierNetwork`'s
real-MNIST training as its own dedicated investigation, "given the cost of each real training
run." `CrossEntropyArrayLayer`/`CrossEntropyArrayBackpropClassifierNetwork` (see
[an array-based binary cross-entropy
sibling](../design-docs/array-siblings/binary-cross-entropy-array-layer.md)) remove that cost the
same way the softmax entry above did.

**Same architecture/config as the documented ensemble baseline** (`demo_mnist_ensemble_recognition.py`,
"the ensemble/real-MNIST investigation" step 3/4 above): `[16]` hidden, `dimension=784`,
`class_count=10`, 5 epochs, real 60000-train/10000-test MNIST, via
`train_ensemble_parallel_from_indices`. Swept `learning_rate` for
`classifier_cls=CrossEntropyArrayBackpropClassifierNetwork` against a matched-seed quadratic
(`ArrayBackpropClassifierNetwork`) baseline at `learning_rate=0.5` (the documented 96.39%
single-seed figure in step 4's own table above is one run only; this reruns it at 3 seeds for a
genuine like-for-like comparison), 3 seeds each:

| configuration | seed test accuracies | mean | stdev |
|---|---|---|---|
| quadratic @ lr=0.5 (baseline) | 95.85%, 95.92%, 96.14% | 95.97% | 0.15% |
| cross-entropy @ lr=0.5 (untuned) | 92.25%, 93.46%, 92.66% | 92.79% | 0.62% |
| cross-entropy @ lr=0.25 | 95.63%, 95.55%, 95.67% | 95.62% | 0.06% |
| cross-entropy @ lr=0.1 | 96.30%, 96.32%, 96.41% | **96.34%** | 0.06% |
| cross-entropy @ lr=0.05 | 96.00%, 96.08%, 96.21% | 96.10% | 0.11% |

At `learning_rate=0.5` (the quadratic-tuned rate, unchanged), cross-entropy underperforms by 3.18
points - the same undamped-gradient overshoot pattern found at every other scale checked (XOR,
small proxy, full MNIST quadratic-vs-cross-entropy in step 2/3 above). But once retuned to
`learning_rate=0.1`, cross-entropy doesn't just match the quadratic baseline the way it did on toy
XOR - **it comes out ahead, +0.37 points, and every cross-entropy seed at this rate beats every
quadratic-baseline seed** (96.30%-96.41% vs. 95.85%-96.14%, non-overlapping ranges) - the same
"every seed beats every seed" separation the softmax entry above used to call its own result
decisive, not just noise. `learning_rate=0.05` also beats the baseline (96.10%) but by less than
0.1's own margin - 0.1 is the better of the two rates checked below the untuned one, not simply
"lower is better."

**This is a different, more favorable outcome than the toy-scale finding** - worth stating plainly
rather than assuming the toy result would simply transfer. A plausible mechanism: MNIST's binary
one-vs-rest sub-problems (each ensemble classifier's "is this digit?" target) are far more
class-imbalanced than XOR's own even 50/50 split, and cross-entropy's undamped gradient - the same
property that makes it overshoot at the quadratic-tuned rate - may push harder against the
frequent negative examples once the learning rate is small enough not to overshoot, versus
quadratic's `a(1-a)`-damped gradient doing comparatively less work against them. Not verified
further here (out of scope for this workplan); flagged as the most likely explanation, not
asserted as confirmed.

**Wall-clock**: no separate table needed, per
[goals and strategy](../project/goals-and-strategy.md#measurement-discipline-the-per-node-paths-two-jobs-and-the-one-it-doesnt-have)'s
own discipline - `CrossEntropyArrayLayer` differs from plain `ArrayLayer` by one elementwise
subtract in `compute_output_delta`/`compute_output_delta_batch` only, so its cost is
indistinguishable from `ArrayBackpropClassifierNetwork`'s own already-measured per-classifier
wall-clock (see "step 4" above). The sweep above confirms this directly, incidentally: every one
of its 15 runs (quadratic and cross-entropy alike, all `[16]`-hidden/5-epoch/full-MNIST) landed
within 81-99 seconds of each other on the machine this sweep ran on, with no systematic gap
between the two loss functions.

**Decision: adopted, with its own tuned learning rate.** Cross-entropy is not a drop-in
replacement for quadratic loss at the ensemble's existing tuned rate (`learning_rate=0.5`) - but
once retuned to `learning_rate=0.1`, it's a genuine, if modest, win over the documented ensemble
baseline at real-MNIST scale, reversing the "matches, doesn't beat" framing the toy-scale
investigation left this workplan with. Not adopted as the new default (the margin is real but
small, and `learning_rate=0.5`-tuned quadratic remains a perfectly reasonable choice); recorded
here as a validated, available option for anyone retuning this architecture from scratch.
