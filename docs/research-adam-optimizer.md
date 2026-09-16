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
