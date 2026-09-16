# demos

[← back to README](../README.md)

All demos run through a single entrypoint, `. cli demo`, which launches
`perceptron/demos/menu.py` - a small text-based REPL rather than a separate CLI command per
demo. It prints a numbered table of every demo (name + one-line summary, sourced from
`perceptron/demos/registry.py`), prompts for a number, then prints that demo's longer
description before importing its module and calling its `main()`. Once the demo finishes (its
windows are closed, or it was headless/console-only to begin with), the menu loops back to the
table so you can run another one or type `q` to quit - so a session can run several demos back
to back without re-invoking the shell command each time. The sections below describe what each
menu entry does; select it by the title shown here.

The demo number can also be passed directly as an argument - `. cli demo 3` prints that demo's
description, runs it, and exits immediately, skipping the interactive table/prompt entirely.
This is the override for scripts and LLMs that already know which demo they want (debugging a
specific one, or driving it non-interactively) without having to answer a prompt.

## demo: minimum-disturbance training

    . cli demo

Select **Minimum-disturbance training** from the menu it prints.

Runs `perceptron/demos/demo_minimum_disturbance_training.py`, which replicates `test_training_of_linear_classifier` outside
of pytest: trains a classifier against a random reference classifier and pops up three
windows — the convergence curve on a linear scale, the same curve on a log scale (better for
seeing how fast it converges, since disagreement tends to drop roughly exponentially — see
the cardinality-sweep demo below for more on this), and the decision-boundary chart. Both
convergence charts are smoothed with a trailing moving average. Unlike the test, it isn't
time-boxed — the windows stay open until you close them. It uses `cardinality=4` (four
hyperplanes ANDed together) to also showcase the minimum-disturbance multi-unit learning
rule described in [structure](structure.md); since a higher cardinality shrinks the
reference's positive region, it uses `train.reachable_reference_and_training_data` (also
used by the cardinality-sweep demo) to regenerate the reference rather than risk failing on
one unlucky `randomize()`. A 2D convex region needs at least 3 half-planes to be bounded at
all, so at `cardinality=4` the demo also passes `geometry.is_positive_region_bounded` as that
helper's `is_valid` filter, rejecting (and regenerating) any reference whose positive region
isn't a bounded, closed shape — bounded regions are rare (~10% of random draws at this
cardinality), but rejecting on it is a cheap geometry check with no sampling, so a large
attempt budget is still fast. It also prints the trained student's classification of a fresh
point (never seen during training) alongside the reference's, to show the trained classifier
actually being used to predict, not just compared to the reference by eye on a chart. The
decision-boundary chart's bounds are widened as needed, via `chart.reference_region_bounds`,
so the whole bounded region stays visible instead of being cropped at the training bounds.

## demo: linear-classifier cardinality sweep

    . cli demo

Select **Linear-classifier cardinality sweep** from the menu it prints.

Runs `perceptron/demos/demo_linear_classifier_cardinality_sweep.py`, which trains independent reference/student
pairs at `cardinality = 1, 2, 3, 4` and overlays their disagreement-rate convergence curves
on two charts, each built from the series smoothed with a trailing moving average (to see
the trend through the sampling noise from `class_balanced_disagreement_rate`'s small
per-class sample count) — one linear-scale, one log-scale (better for comparing how fast
each cardinality's disagreement rate drops, since it does so roughly exponentially) — and
printing each cardinality's before/after disagreement and a prediction-agreement check. A
disagreement rate of exactly 0.0 has no position on a log axis, so a fully-converged
cardinality's curve on that chart simply stops once it hits zero. Since a higher cardinality
shrinks the reference's positive region (the intersection of more half-planes), some
randomly generated reference
classifiers make one class unreachable within the given bounds — the demo regenerates the
reference (up to 20 times) rather than failing the whole sweep on one unlucky draw.

## demo: unreachable-class safety guard

    . cli demo

Select **Unreachable-class safety guard** from the menu it prints.

Runs `perceptron/demos/demo_unreachable_class_safety_guard.py`, a headless, console-only demo of
`random_alternating_training_data`'s safety guard: it first generates training data from a
normal, randomly initialised classifier (retrying if an unlucky `randomize()` happens to make
one class unreachable within the bounds — this alone can occasionally happen), then
deliberately constructs a classifier with tiny weights and a large threshold, whose decision
boundary never crosses its bounds, and shows the resulting `RuntimeError` being raised and
caught instead of hanging forever.

## demo: XOR linear-classifier ceiling

    . cli demo

Select **XOR: linear-classifier ceiling** from the menu it prints.

Runs `perceptron/demos/demo_xor_linear_classifier_ceiling.py`. Every other demo's target is a
`LinearClassifierNetwork` of the same architecture the student trains with, so it's always
representable by construction; this one deliberately isn't. It trains students at
`cardinality = 1..4`, swept across AND, OR, and (where distinct) majority `required_active`
gates (see [structure](structure.md)), against an XOR-style target
(`category = (x > 0) != (y > 0)`, two diagonally opposite quadrants) built as a small
adapter object exposing the same `input_bounds`/`classify_state` interface as
`LinearClassifierNetwork` - which is enough for it to drop straight into
`random_alternating_training_data`/`class_balanced_disagreement_rate` unchanged, no new
training-data or metric code needed. Each configuration's line also reports its
`train_linear_classifier_network` result's `.diagnostic` - whether that run converged,
plateaued, or was still improving (see [structure](structure.md)) - and a summary tally
across all nine closes the demo's own claim that none of them are close to converging,
rather than just asserting it. It prints why: the output layer's weights are always `1.0`
per hidden node, so the output can only be a monotonically non-decreasing function of how
many hidden nodes are active - XOR needs the opposite for some units, which no
`required_active` at any cardinality can express. Finishes by plotting the training data
(colored by true label) against the best-performing student's hyperplanes, to show the
mismatch visually.

## demo: XOR backprop convergence

    . cli demo

Select **XOR: backprop convergence** from the menu it prints.

Runs `perceptron/demos/demo_xor_backprop_convergence.py` - the direct counterpart to the demo above, using
the exact same `XORTarget` (imported from `demo_xor_linear_classifier_ceiling.py`, not reimplemented)
and the exact same `random_alternating_training_data`/`train_linear_classifier_network` calls,
but training a `BackpropClassifierNetwork` (see [structure](structure.md#backprop)) instead of a
`LinearClassifierNetwork`. Where every configuration in the previous demo plateaus well short of
convergence, this one reaches a training accuracy well past that ceiling, because its output
layer is trained too - a hidden node here can push the output either way, not just
monotonically increase how likely it is to fire. Prints the same kind of `.diagnostic` summary
as every other training demo, then plots the training data over a probability heatmap
(`chart.plot_classifier_probability_heatmap`) instead of decision lines, since the learned
region isn't a union of half-planes a line can represent.

## demo: backprop stripes architecture sweep

    . cli demo

Select **Backprop stripes architecture sweep** from the menu it prints.

Runs `perceptron/demos/demo_backprop_stripes_architecture_sweep.py`, which mirrors
`demo_linear_classifier_cardinality_sweep.py`'s pattern (train several configurations against the same target,
overlay smoothed convergence curves) but compares `BackpropClassifierNetwork` architectures -
`[4]`, `[8]`, `[4, 4]`, `[8, 8]` - instead of `LinearClassifierNetwork` cardinalities. `[8]` and
`[4, 4]` share the same total node count, as do `[4]` and `[8, 8]`'s first layer vs its second,
so the comparison is at a matched node budget rather than just "more capacity wins". The target
is `StripesTarget`, 4 alternating vertical bands - harder than `demo_xor_backprop_convergence.py`'s 2-region
XOR, so the architectures actually separate instead of all converging easily. Unseeded, like
every sweep demo in this codebase: which architecture comes out ahead varies noticeably between
runs, since plain fixed-learning-rate gradient descent is sensitive to where random
initialization happens to land - this demo shows that variability rather than asserting depth
or width is definitively better.

## demo: backprop circular boundary

    . cli demo

Select **Backprop circular boundary** from the menu it prints.

Runs `perceptron/demos/demo_backprop_circular_boundary.py`. Every other demo's target boundary,
representable or not, is built from straight edges - `LinearClassifierNetwork`'s positive
region is always a polygon (an intersection of half-planes, see
`geometry.reference_positive_region_polygon`), so it can only ever facet a curve with more and
shorter edges, never actually curve. This target is a disk (radius 4, centered on the origin) -
a genuinely curved boundary. Trains a single-hidden-layer `BackpropClassifierNetwork` (measured:
reaches ~0.99 training accuracy) and plots the learned probability heatmap against the training
data, visibly showing a smooth, rounded decision boundary rather than a faceted polygon.

## demo: backprop vs linear parity check

    . cli demo

Select **Backprop vs linear parity check** from the menu it prints.

Runs `perceptron/demos/demo_backprop_linear_parity_check.py`. Every other backprop demo picks a target no
`LinearClassifierNetwork` can represent well (XOR, stripes, a circle); this one is the opposite
check - a single half-plane (`cardinality=1`), the easiest possible target and squarely within
`LinearClassifierNetwork`'s own representational sweet spot. Trains a `LinearClassifierNetwork`
and a `BackpropClassifierNetwork` on the exact same reference and training data (measured: both
reach ~0.995-0.999 training accuracy), then plots the reference's boundary (green), the linear
student's boundary (purple), and the backprop student's boundary as a probability heatmap
underneath, all overlaid - a parity check showing backprop learns just as well here, not only on
the harder targets the other backprop demos focus on.

## demo: UCI digit recognition

    . cli demo

Select **UCI digit recognition** from the menu it prints.

Runs `perceptron/demos/demo_uci_digit_recognition.py` - classic-style handwritten digit recognition,
using `MultiClassBackpropClassifierNetwork` (see [structure](structure.md#multi-class)) instead
of any 2D geometric target. Loads the bundled UCI ML hand-written digits dataset
(`data/digits/digits.csv` via `digits_data.load_digits_dataset`, 8x8 pixel images, 10 classes,
1797 samples), splits 80/20 into train/test, and trains a single-hidden-layer (32-node) network
with `train_linear_classifier_network` - reused completely unmodified, the same as every other
backprop demo. Measured: ~99.5% training accuracy, ~95-97% held-out test accuracy, in under a
minute. Saves the trained model to `data/digits/trained_model.json` (gitignored - a regenerable
build artifact) via the new `save()`/`load()` persistence, printing how to reload it without
retraining. Plots a training-accuracy-by-epoch curve, a confusion matrix
(`chart.new_confusion_matrix_figure`), and a grid of sample test predictions
(`chart.sample_predictions_figure`) colored to flag any incorrect ones.

## demo: vectorized UCI digit recognition

    . cli demo

Select **Vectorized UCI digit recognition (numpy)** from the menu it prints.

Runs `perceptron/demos/demo_vectorized_uci_digit_recognition.py` - trains
`MultiClassBackpropClassifierNetwork` and its numpy-array-backed sibling
`VectorizedMultiClassBackpropClassifierNetwork` (see
[structure](structure.md#vectorized-array-based-classes)) on the same UCI digits split, same seed
and hyperparameters, and reports each one's measured training/test accuracy and wall-clock
training time side by side. Measured: matching accuracy (99.5%/96.1% vs. 99.4%/96.7% train/test)
at 8.57s vs. 77.16s - a 9.00x wall-clock speedup, the same numbers documented in
[vectorized array-based classes](vectorized-array-classes.md#measured-results), checked live
rather than only quoted. Plots both training-accuracy-by-epoch curves overlaid, a
confusion matrix, and sample test predictions for the vectorized network.

## demo: Rust vs vectorized UCI digit recognition

    . cli demo

Select **Rust vs vectorized UCI digit recognition** from the menu it prints.

Runs `perceptron/demos/demo_rust_vs_vectorized_uci_digit_recognition.py` - the demo above's own
scenario extended with a third network: `RustArrayMultiClassBackpropClassifierNetwork` (see
[the production cutover plan](rust-production-cutover.md)), the Rust-array-core-backed sibling
that replaces numpy as the production backend (numpy stays on permanently as the benchmark
comparison, not removed or deprecated). Trains all three at the same seed/hyperparameters and
reports each one's measured accuracy and wall-clock training time side by side. Measured: matching
accuracy (99.5%/96.9% pure-Python, 99.5%/96.9% numpy, 99.7%/97.5% Rust train/test) at 68.89s
(pure-Python) vs. 4.30s (numpy) vs. 1.30s (Rust) - a 3.30x speedup over numpy and 52.89x over
pure-Python, close to [research and
analysis](research-and-analysis.md#phase-2-tier-2-real-per-example-training-is-a-genuine-win-at-both-scales-measured)'s
own recorded 3.40x figure (real training runs vary run to run). Both figures above predate
2026-09-16's later matmul SIMD work (see [research and
analysis](research-and-analysis.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)) -
re-running this demo today would show a faster Rust time and a higher ratio, around 3.6x per that
entry's own measurement, not re-captured here in full per-network detail. Saves the trained Rust model to
`data/digits/trained_model_rust.json`, printing how to reload it without retraining. Plots all
three training-accuracy-by-epoch curves overlaid, a confusion matrix, and sample test predictions
for the Rust network.

## demo: UCI digit capture

    . cli demo

Select **UCI digit capture** from the menu it prints.

Runs `perceptron/demos/demo_uci_digit_capture.py` - an interactive companion to the digit-recognition
demo above. Loads the model that demo trains and saves (`data/digits/trained_model.json` via
`MultiClassBackpropClassifierNetwork.load`; run `. cli demo` and choose **UCI digit recognition** first if that file
doesn't exist yet), then opens two canvases: a 32x32 grid you paint with the mouse (click or
click-and-drag - each stroke stamps a `CAPTURE_BRUSH_RADIUS`-wide square, not a single cell; see
below), and an 8x8 preview next to it showing the result of genuinely reproducing the bundled
training data's own preprocessing on what you drew (see `digit_capture.downsample_to_target_grid`
and docs/structure.md's "multi-class" section) - 32x32 divided into nonoverlapping 4x4 blocks,
each block's "on" pixel count (0-16) becoming one of the 8x8 grid's graded values, exactly like
the real UCI hand-written digits dataset's own extraction from a 32x32 NIST bitmap. Every stroke
updates the preview and reclassifies live, showing the predicted digit and the model's confidence
in it.

The brush matters more than it might look: a single-mouse-cell-wide stroke (what plain
click-drag paints without one) comes out only ~25% as intense as any real training example after
downsampling - measured directly, this was a real bug (almost everything misclassified toward
whichever class happened to catch faint input) before `digit_capture.paint_brush_stroke` was
added to stamp a wider, more realistically pen-stroke-thick mark per event.

Unlike every other demo, this one needs a display and mouse input - it's not part of the
automated test suite (`tests/test_digit_capture.py` covers only the pure grid-flattening/
coordinate-mapping/downsampling/brush logic behind it, not the tkinter UI itself).

## demo: MNIST ensemble recognition

    . cli demo

Select **MNIST ensemble recognition** from the menu it prints.

Runs `perceptron/demos/demo_mnist_ensemble_recognition.py` - the same shape as the digit-recognition demo
above, but on the real, full-scale MNIST dataset (28x28 pixel images, 60000 train / 10000 test)
instead of the small bundled UCI set, and using `EnsembleBackpropClassifierNetwork` (see
[structure](structure.md#multi-class)) instead of `MultiClassBackpropClassifierNetwork`: 10
completely independent `FanInAwareBackpropClassifierNetwork`s (see
[structure](structure.md#backprop-siblings) - a fan-in-aware-init sibling of
`BackpropClassifierNetwork`, opted into via `ensemble_train.py`'s `classifier_cls` parameter),
one per digit, each trained on its own small, class-balanced binary dataset, with no shared
hidden layer and no synchronization between them at all - trained as 10 parallel
`multiprocessing` jobs (`ensemble_train.train_ensemble_parallel_from_indices`), memory-aware
worker count included. The design behind this, and a real memory-exhaustion failure hit (and
genuinely fixed, not just worked around) while building it at full scale, are written up in
[research and analysis](research-and-analysis.md#parallelizing-mnist-training). One-time setup:
converts the supplied `data/mnist/mnist-{train,test}.parquet` files to a flat binary format
(`mnist_data.convert_parquet_to_binary`) the first time it's run, so training itself never needs
`pyarrow`. Measured on this machine: 29.6 minutes wall-clock for the full training run, memory
stable throughout, 96.01% held-out test accuracy - up from 89.4% before switching to the
fan-in-aware sibling (a real, +6.6-point improvement from the init fix alone, at the same
wall-clock cost - see [research and
analysis](research-and-analysis.md#the-ensemblereal-mnist-investigation)). Saves the trained
model to `data/mnist/trained_model.json` (gitignored, same as the
digit-recognition demo's), printing how to reload it without retraining. Plots the same chart set
as the digit-recognition demo: a per-digit training-accuracy-by-epoch curve, a confusion matrix,
and a grid of sample test predictions.

## demo: vectorized MNIST recognition

    . cli demo

Select **Vectorized MNIST recognition (numpy)** from the menu it prints.

Runs `perceptron/demos/demo_vectorized_mnist_recognition.py` - headless, console-only. Trains
`MultiClassBackpropClassifierNetwork` and `VectorizedMultiClassBackpropClassifierNetwork` for one
real epoch each over the full 60000-example MNIST training set (`[30]`-node hidden layer, the
same architecture [research and analysis](research-and-analysis.md#parallelizing-mnist-training)'s
own ~12.5-minutes/epoch pure-Python figure used), reporting test accuracy and wall-clock epoch
time side by side, plus timing `load_mnist_dataset_as_array`'s bulk decode against
`load_mnist_dataset`'s tuple-per-example decode on the same file. Measured: 32.7s vs. 18.45 min
- a 33.87x wall-clock speedup for the epoch, and a 15.44x speedup for the bulk decode alone (see
[structure](structure.md#vectorized-array-based-classes)). Takes on the order of 15-20 minutes to
run, mostly the pure-Python epoch.

## demo: Rust vs vectorized MNIST recognition

    . cli demo

Select **Rust vs vectorized MNIST recognition** from the menu it prints.

Runs `perceptron/demos/demo_rust_vs_vectorized_mnist_recognition.py` - headless, console-only.
The demo above's own scenario extended with a third network:
`RustArrayMultiClassBackpropClassifierNetwork` (see [the production cutover
plan](rust-production-cutover.md)), the Rust-array-core-backed sibling that replaces numpy as the
production backend. Trains all three for one real epoch each over the full 60000-example MNIST
training set (`[30]`-node hidden layer, same architecture as the demo above), reporting test
accuracy and wall-clock epoch time side by side. Measured: matching accuracy (93.4%/93.3% pure-Python,
93.4%/93.2% numpy, 93.0%/92.9% Rust train/test) at 1128.9s (pure-Python) vs. 15.9s (numpy) vs.
11.4s (Rust) - a 1.39x speedup over numpy and 98.64x over pure-Python, close to [research and
analysis](research-and-analysis.md#phase-2-tier-2-real-per-example-training-is-a-genuine-win-at-both-scales-measured)'s
own recorded 1.31x multi-seed figure. Both figures above predate 2026-09-16's later matmul SIMD
work (see [research and
analysis](research-and-analysis.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)) -
re-running this demo today would show a faster Rust time and a higher ratio, around 2.6x per that
entry's own measurement, not re-captured here in full per-network detail. Takes on the order of
15-20 minutes to run, mostly the pure-Python epoch.

## demo: MNIST ensemble capture

    . cli demo

Select **MNIST ensemble capture** from the menu it prints.

Runs `perceptron/demos/demo_mnist_ensemble_capture.py` - an interactive companion to the MNIST-recognition
demo above, the same overall interaction as `demo_uci_digit_capture.py` but against MNIST's own
reference preprocessing instead of the UCI dataset's block-counting downsample. Loads the model
the recognition demo trains and saves (`data/mnist/trained_model.json` via
`EnsembleBackpropClassifierNetwork.load`; run `. cli demo` and choose **MNIST ensemble recognition** first if that file doesn't
exist yet), then opens two canvases: a 64x64 grid you paint with the mouse (click or click-and-
drag - each stroke stamps a `CAPTURE_BRUSH_RADIUS`-wide square), and a 28x28 preview next to it
showing the result of genuinely reproducing MNIST's own three-step preprocessing on what you drew
(see `mnist_capture.preprocess_capture` and [structure](structure.md#multi-class)): crop to the
drawn content's bounding box, aspect-preserving anti-aliased scale so the longer side hits 20
pixels, center-of-mass placement into the 28x28 field. Every stroke updates the preview and
reclassifies live, showing the predicted digit and the model's confidence in it.

`CAPTURE_BRUSH_RADIUS` (4) was chosen empirically against the real trained ensemble, the same way
the UCI capture tool's radius was: tested against hand-simulated "0"/"1"/"7" strokes, confidence
stayed at ~1.00 through radius 2-5, and degraded from radius 6 onward as a "0"'s hole started
filling in (0.94 at radius 6, 0.34 at radius 8, misclassified as "8" by radius 12) - 4 sits
comfortably inside the safe range.

Building this demo surfaced a real bug, found via an end-to-end smoke test (a simulated paint
stroke through the actual tkinter app, not just the pure preprocessing functions in isolation):
`resize_area_weighted`'s area-weighted average can overshoot its mathematically-guaranteed
`[0.0, 1.0]` bound by a tiny floating-point amount (observed directly: `1.0000000000000002`),
which the preview canvas's strict `[0.0, 1.0]` color-mapping assertion then rejected -
`scale_to_fit` now clamps its result to fix this (the general-purpose `resize_area_weighted`
itself is left unclamped, since its valid output range depends on whatever range its caller's
own input happens to be in - not always `[0.0, 1.0]`).

Unlike every other demo, this one needs a display and mouse input - it's not part of the
automated test suite (`tests/test_mnist_capture.py` covers only the pure
crop/scale/center-of-mass/brush logic behind it, not the tkinter UI itself).

## demo: backprop variant comparison

    . cli demo

Select **Backprop variant comparison** from the menu it prints.

Runs `perceptron/demos/demo_backprop_variant_comparison.py`, which reproduces three A/B
comparisons [research and analysis](research-and-analysis.md) documents from one-off,
never-saved investigation scripts - as a permanent, re-runnable demo instead of numbers you can
only read about. Trains all seven variants fresh (no new library code beyond the demo itself -
every network class it uses already exists and is already tested elsewhere), prints each
comparison's measured accuracy, and plots a training-accuracy-by-epoch chart per section:

- **Multi-class loss function** - `MultiClassBackpropClassifierNetwork` (one-vs-rest, MSE) vs
  `SoftmaxMultiClassBackpropClassifierNetwork` (softmax, cross-entropy) on the full UCI digits
  set, matching `demo_uci_digit_recognition.py`'s own architecture and hyperparameters exactly
  (see [research and analysis](research-and-analysis.md#softmaxcross-entropy-re-alignment)).
- **Binary loss function** - `BackpropClassifierNetwork` (quadratic) vs
  `BinaryCrossEntropyBackpropClassifierNetwork` (cross-entropy) on the XOR target, at both the
  shared learning rate and cross-entropy's own retuned rate (see
  [research and analysis](research-and-analysis.md#binary-cross-entropy-for-backpropclassifiernetwork)).
- **Weight-init scheme** - this codebase's fan-in-aware production default vs a Xavier/Glorot
  variant implemented only locally in this demo file (not a real library class, since it was
  measured not worth adopting - see
  [research and analysis](research-and-analysis.md#xavierglorot-init-measured-not-worth-adopting)),
  on UCI digits.

Takes about 4 minutes end to end (measured directly). Unlike the other backprop demos, this one
doesn't train against a single target with one clear answer - each section is a comparison, and
the point is to let the documented findings be checked directly rather than assumed to still
hold.
