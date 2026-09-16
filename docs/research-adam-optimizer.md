# research and analysis: Adam optimizer

[← back to research and analysis](research-and-analysis.md)

The measurement results for [the Adam optimizer workplan](adam-optimizer.md) - see [research and
analysis](research-and-analysis.md) for what this collection of docs is for.

## Adam: tuned-XOR measurement (stage 3 of the Adam optimizer workplan)

[`AdamBackpropClassifierNetwork`](adam-optimizer.md) (stage 2, correctness-validated only) measured
for the first time against a real training run: the same pinned XOR scenario
`test_backprop_training_pipeline.py` uses (`BackpropClassifierNetwork([8], 2,
square_bounds(10.0))`, 300 examples, 100 epochs), 10 (data-generation seed, weight-init seed) pairs,
`beta1`/`beta2`/`epsilon` at their Kingma & Ba defaults. Note: `docs/adam-optimizer.md`'s original
"same scenario momentum's own baseline used" framing for this stage doesn't hold up - momentum's
own baseline was actually measured on the 320-example real-MNIST proxy, not XOR (see ["momentum:
measured, not worth adopting"](research-backprop-siblings.md#momentum-measured-not-worth-adopting));
this stage instead reuses the pinned XOR scenario the binary-cross-entropy and ReLU investigations
used, which is what `test_backprop_training_pipeline.py` actually pins.

### the demo-tuned rate: Adam collapses badly

At the demo-tuned `learning_rate=1.0` every other sibling here was first measured at:

| config | mean training accuracy | stdev |
|---|---|---|
| sigmoid, no Adam, lr=1.0 (tuned baseline) | 97.80% | 0.92% |
| Adam, lr=1.0 (untuned) | 58.27% | 9.09% |

Adam collapses badly at this rate - worse than cross-entropy's or ReLU's own untuned-rate dips.
Mechanism: Adam's effective per-step move is `~learning_rate` regardless of gradient magnitude
(the `m_hat / (sqrt(v_hat) + epsilon)` term is normalized to roughly unit scale once `t` is a few
steps in), so a rate tuned for undamped sigmoid+quadratic SGD is far too large once every step is
already near that scale by construction.

### sweeping the learning rate down

Sweeping `learning_rate` down for Adam alone (still 10 seeds):

| learning_rate | mean training accuracy | stdev |
|---|---|---|
| 1.0 | 58.27% | 9.09% |
| 0.5 | 71.73% | 7.78% |
| 0.25 | 75.17% | 1.72% |
| 0.1 | 90.73% | 10.49% |
| 0.05 | 97.97% | 1.26% |
| 0.01 | 98.63% | 0.67% |

### re-measured at 15 seeds

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

### interpretation and decision

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

### the sigmoid rows: a reproducibility check

The sigmoid rows are a genuine reproducibility check, not a rerun of the same numbers: on an
independently-constructed proxy subset, they qualitatively reproduce [the
learning-rate-vs-batch-size follow-up](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)'s
own shape - a fixed rate degrades steadily as batch size grows (fewer update steps/epoch), scaling
the rate compensates at `batch_size=32`, and scaling diverges outright at `batch_size=128`
(high-variance collapse).

### a materially bigger, cleaner result than stage 3

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

### decision

**Decision:** Adam should be used with a **fixed** `learning_rate` across batch sizes, the opposite
prescription from SGD/momentum's linear scaling rule - and, at this proxy's scale, that fixed rate
gives Adam a real, substantial robustness advantage over sigmoid at every batch size tested, most
dramatically at `batch_size=128` where sigmoid's training destabilizes regardless of which
learning-rate regime is used. This is the clearest positive result the Adam workplan has produced
so far, and directly validates the "worth its own measurement" rationale in
[Adam optimizer](adam-optimizer.md)'s opening motivation: unlike momentum, Adam's per-parameter
adaptive scaling genuinely does something SGD-with-momentum's single shared velocity term
couldn't, once gradients come from batches rather than single examples.

## Adam at real-MNIST-ensemble scale: the proxy result holds (stage 5 of the Adam optimizer workplan)

Stage 4's proxy-scale result was strong enough to justify the ~30-minute-per-config real-scale
run it was gated on (see the delivery-stages decision in [Adam
optimizer](adam-optimizer.md#delivery-stages-each-its-own-pr-per-this-repos-practice)). Real full
60000/10000 MNIST, via the same `EnsembleBackpropClassifierNetwork` architecture (10 independent
`[16]`-hidden binary sub-networks, one per digit) `demo_mnist_ensemble_recognition.py` uses in
production, fan-in-aware init, 5 epochs, `batch_size=128` - the batch size stage 4's proxy result
was most dramatic at - each optimizer at its own fixed, untuned-for-batch-size rate (sigmoid
`lr=0.5`, Adam `lr=0.01`, both the same per-example rates every other measurement here uses),
seed=0, single run per config (matching this codebase's own established precedent for real-scale
runs - each one too costly for a multi-seed sweep, see [the ensemble/real-MNIST
investigation](research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)):

`ensemble_train.py`'s own parallel-training path doesn't wire `batch_size` through its worker
functions (only `train_linear_classifier_network`'s per-example path) - measuring this needed a
small hand-rolled script mirroring `train_ensemble_parallel_from_indices`'s job/worker structure
but calling `train_backprop_network_mini_batch` instead, the "hand-rolled ad hoc parallel-training
script" alternative [Adam optimizer](adam-optimizer.md)'s own scope section named up front, rather
than extending the production module for a one-off measurement.

| config | test accuracy | wall-clock |
|---|---|---|
| sigmoid, fixed lr=0.5, batch_size=128 | 89.06% | 30.5 min |
| Adam, fixed lr=0.01, batch_size=128 | **94.62%** | 34.9 min |

Adam beats sigmoid by **5.56 points** at this batch size, real scale - the proxy-scale gap
(86.75% vs. 71.50%, a 15.25-point gap on the 320-example proxy) held up qualitatively, if smaller
in absolute terms, on the real dataset. Both numbers sit below the documented `batch_size=1`
production baseline (96.01%, see [the ensemble/real-MNIST
investigation](research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)) - batching
at all costs *some* accuracy at this architecture/epoch count regardless of optimizer, consistent
with every batch-size sweep in this codebase so far - but Adam recovers almost all of it (1.39
points off the `batch_size=1` baseline) while sigmoid gives up nearly 7 points (6.95) doing the
same.

**Decision:** the stage 4 finding is confirmed at real production scale, not just the toy proxy -
Adam at a fixed `learning_rate` is the more batch-size-robust optimizer, real MNIST included. This
doesn't yet translate into a wall-clock win on its own (both configs cost about the same ~30-35
minutes - `learn_batch`'s own docs already establish that mini-batching alone "does not by itself
speed anything up," only changes when the weight write happens, see [mini-batch gradient
descent](mini-batch-gradient-descent.md)) - the practical payoff would come from pairing this
result with the array-based, Rust-matmul-backed Adam sibling flagged in
[structure](structure.md#possible-next-steps) ("deepening what's already here"): genuine batched
matmul is what would make large batches actually cheap in wall-clock terms, and this result is
what says that combination wouldn't cost the accuracy tax sigmoid pays for it.

## RMSprop: the second-moment term alone accounts for Adam's batch-size win (stage 7 of the Adam optimizer workplan)

Stage 4's win (a fixed `learning_rate` barely degrading Adam across `batch_size`, where sigmoid
collapses) triggered this workplan's own flagged condition for the RMSprop ablation
(`AdamBackpropClassifierNetwork(..., beta1=0.0)` - see [Adam optimizer](adam-optimizer.md#delivery-stages-each-its-own-pr-per-this-repos-practice)'s
stage 7): no new code needed, a pure measurement stage isolating whether the win comes from
Adam's per-parameter second-moment normalization alone or needs its first-moment (momentum-like)
smoothing too.

Repeats stage 4's real-MNIST-proxy batch-size sweep methodology exactly (fan-in-aware init, `[16]`
hidden, 5 epochs, `batch_size` in 1/8/32/128, 10 seeds each, fixed `learning_rate`, fork-based
multiprocessing pool) on a freshly-built 320-example real-MNIST digit-3 proxy (independently
constructed, not the same instance stage 4 used - the same "reproducibility check, not a rerun of
the same numbers" posture stage 4's own sigmoid rows used against the original momentum sweep),
with sigmoid and Adam rerun alongside the new RMSprop row for a directly paired three-way
comparison rather than trusting stage 4's numbers out of context:

| config | batch_size=1 | batch_size=8 | batch_size=32 | batch_size=128 |
|---|---|---|---|---|
| sigmoid, fixed lr=0.5 | 92.12% ± 1.59% | 91.87% ± 1.15% | 91.12% ± 1.63% | 68.12% ± 18.72% |
| Adam, fixed lr=0.01 | 92.25% ± 1.09% | 92.62% ± 0.88% | 94.38% ± 0.84% | 92.50% ± 0.79% |
| RMSprop, fixed lr=0.01 | 92.75% ± 1.56% | 92.75% ± 0.50% | 94.50% ± 1.00% | 92.38% ± 2.59% |

### the sigmoid/Adam rows: a reproducibility check

Same shape as stage 4's own reproducibility check: sigmoid degrades sharply and destabilizes at
`batch_size=128` (68.12% ± 18.72%, variance exploding) while Adam stays flat across two orders of
magnitude of batch size (92.25%-94.38%) - qualitatively identical to stage 4's finding on an
independently-constructed proxy subset, if not bit-comparable (different random examples, so the
absolute numbers differ from stage 4's own 89.62%-71.50%/90.50%-86.75%).

### RMSprop tracks Adam within seed-to-seed noise at every batch size

**RMSprop (`beta1=0.0`) is statistically indistinguishable from Adam at every batch size tested** -
the largest gap between the two (`batch_size=32`: 94.50% vs. 94.38%) is a tenth of a point, far
inside either row's own stdev (0.84%-2.59%), and RMSprop is actually marginally ahead of Adam at
three of the four batch sizes (1, 8, 32), behind only at `batch_size=128` by 0.12 points - noise,
not a trend. This directly answers the question this stage was scoped to isolate: **the
batch-size-robustness win is fully explained by the per-parameter second-moment normalization
alone** - zeroing out Adam's first-moment (momentum-like) term costs nothing measurable at this
proxy's scale. Consistent with, and a sharper version of, [the momentum
finding](research-backprop-siblings.md#momentum-measured-not-worth-adopting): a first-moment
smoothing term keeps showing up as not worth its own complexity in this codebase's measurements,
whether attached to plain SGD (momentum) or riding along inside Adam.

### decision

**Decision:** RMSprop is not adopted as a separate capability - it requires no new code
(`beta1=0.0` on the already-shipped `AdamBackpropClassifierNetwork`/`AdamArrayLayer`/
`AdamRustArrayLayer` families already *is* RMSprop), and this measurement found no accuracy or
robustness edge over Adam itself that would justify promoting it to a named, separately-recommended
option. Adam stays the recommended choice for batch-size robustness; RMSprop remains available as
a zero-code special case (`beta1=0.0`) for anyone who wants to isolate the second-moment term
specifically, not as a distinct sibling this codebase steers users toward.
