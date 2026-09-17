# structure

[← back to README](../README.md)

```
indrajala_ml/
  model/
    base_node.py                    AbstractNode — value() interface; WeightedInputNode — the
                                     weighted-sum/z() machinery AssociationNode and BackpropNode
                                     both build on
    bounds.py                       validate_input_bounds — the input_bounds shape/width check
                                     shared by every network class below
    state_node.py                   sense point holding a scalar input value
    state_layer.py                  a vector of StateNodes (network input layer)
    association_node.py             weighted, thresholded neuron (z, activation, learning rule)
    association_layer.py            a layer of AssociationNodes over a given input layer
    linear_classifier_network.py    input -> hidden (association) -> output (k-of-n) layers
    backprop_node.py                sigmoid neuron trained by gradient descent (z, forward
                                     cache, output/hidden delta, gradient step); sigmoid() itself
                                     guards the OverflowError math.exp(-z) raises for very
                                     negative z
    backprop_layer.py               a layer of BackpropNodes over a given input layer;
                                     _node_cls — the per-node-class override point every sibling
                                     layer below (softmax, ReLU, cross-entropy, momentum, L2)
                                     uses instead of a retrofit
    backprop_network_base.py        BackpropNetworkBase — layer assembly, forward pass, the
                                     hidden-layer half of backprop, snapshot/restore, shared by
                                     backprop_classifier_network.py and
                                     multiclass_backprop_classifier_network.py; hidden_layer_cls/
                                     output_layer_cls — the per-layer-class override points every
                                     sibling network below uses; randomize_fan_in_aware — the
                                     fan-in-aware init scheme, shared by
                                     multiclass_backprop_classifier_network.py and
                                     fan_in_aware_backprop_classifier_network.py
    backprop_classifier_network.py  input -> hidden (backprop, any depth) -> trainable output
    model_io.py                     save_model_json/load_model_json — the JSON envelope shared by
                                     every save()/load()-capable network class below
    multiclass_backprop_classifier_network.py  one-vs-rest multi-class sibling of
                                     backprop_classifier_network.py (class_count-node output,
                                     fan-in-aware init, save()/load() persistence)
    softmax_output_layer.py         SoftmaxOutputNode/SoftmaxOutputLayer — joint softmax
                                     activation across an output layer's nodes (see "multi-class")
    softmax_multiclass_backprop_classifier_network.py  softmax + cross-entropy sibling of
                                     multiclass_backprop_classifier_network.py
    ensemble_backprop_classifier_network.py  class_count completely independent
                                     BackpropClassifierNetworks (no shared hidden layer), argmax
                                     over each one's own predict_probability at inference
    fan_in_aware_backprop_classifier_network.py  fan-in-aware-init sibling of
                                     backprop_classifier_network.py (see "backprop siblings")
    binary_cross_entropy_backprop_classifier_network.py  binary cross-entropy sibling of
                                     backprop_classifier_network.py (see "backprop siblings")
    relu_layer.py                   ReLUNode/ReLULayer — a hidden-layer-only max(0, z)
                                     activation (see "backprop siblings")
    relu_backprop_classifier_network.py  ReLU-hidden-layer sibling of
                                     backprop_classifier_network.py (see "backprop siblings")
    momentum_layer.py               make_momentum_node_cls/make_momentum_layer_cls — factory
                                     functions adding a momentum term to every trainable layer's
                                     weight update (see "backprop siblings")
    momentum_backprop_classifier_network.py  momentum sibling of
                                     backprop_classifier_network.py (see "backprop siblings")
    l2_regularization_layer.py      make_l2_node_cls/make_l2_layer_cls — factory functions
                                     adding an L2 weight-decay penalty to every trainable layer's
                                     weight update, bias excluded (see "backprop siblings")
    l2_regularized_backprop_classifier_network.py  L2-regularized sibling of
                                     backprop_classifier_network.py (see "backprop siblings")
    adam_layer.py                   make_adam_node_cls/make_adam_layer_cls — factory functions
                                     giving every trainable layer's weight update a per-parameter
                                     adaptive learning rate (Kingma & Ba, 2014) - bias-corrected
                                     running mean/variance estimates per weight, not momentum's
                                     single shared velocity term (see "backprop siblings")
    adam_backprop_classifier_network.py  Adam sibling of backprop_classifier_network.py (see
                                     "backprop siblings", docs/adam-optimizer.md)
    conv_kernel.py                  ConvKernel — one output channel's shared, trainable
                                     kernel weights/bias, with its own accumulate_gradient/
                                     apply_accumulated_gradient pair (see "convolutional layer")
    conv_unit.py                    ConvUnit — one output spatial position's forward/backward
                                     math over a receptive-field window into ConvKernel's shared
                                     weights (see "convolutional layer")
    conv_layer.py                   ConvLayer — a channel_count x out_height x out_width grid of
                                     ConvUnits sharing channel_count ConvKernels; the per-layer
                                     accumulate/apply/snapshot/restore hooks BackpropNetworkBase
                                     dispatches through instead of reaching into layer.nodes
                                     directly (see "convolutional layer")
    conv_multiclass_backprop_classifier_network.py  one ConvLayer -> dense hidden/output layers,
                                     a sibling of multiclass_backprop_classifier_network.py (see
                                     "convolutional layer")
    array_layer.py                  ArrayLayer — one layer's weights/activations as whole numpy
                                     arrays (W, b) instead of per-node objects; forward/backward/
                                     gradient formulas are matrix ops, not per-node loops (see
                                     "vectorized array-based classes")
    vectorized_multiclass_backprop_classifier_network.py  VectorizedMultiClassBackpropClassifierNetwork
                                     — a standalone (non-BackpropNetworkBase) network built from
                                     ArrayLayers, sharing only the external learn/classify_state/
                                     snapshot/save contract every sibling network shares (see
                                     "vectorized array-based classes")
  capture_common.py               stamp_brush — the brush-stamping loop shared by
                                   digit_capture.py and mnist_capture.py's own paint_brush_stroke
  graphics/
    chart.py                      matplotlib helpers: figures/axes, decision-boundary and
                                   training-data plotting, and expanding plot bounds to
                                   fit a classifier's bounded positive region
  geometry.py                     matplotlib-free geometry: intersecting a classifier's
                                   hidden-node half-planes to find its positive region, test
                                   whether that region is bounded and get a tight box around
                                   it, plus a symmetric input-bounds constructor
  train.py                        random training-data generation and the training loop
  evaluate.py                     comparing/measuring classifiers: class-balanced sampling
                                   (efficiently, using geometry.py's tight box around the
                                   positive region when available), sampling a point and
                                   classifying it with two networks, a permutation-invariant
                                   class-balanced disagreement metric, and series smoothing
  digits_data.py                  loads/normalizes the bundled 8x8 digits dataset, plus a
                                   train/test split (a fixed finite dataset, unlike every other
                                   target here, which is continuously re-sampleable)
  multiclass_evaluate.py          confusion_matrix and accuracy against a held-out test set -
                                   evaluate.py's functions are two-class- and geometry-specific
                                   and don't generalize to this
  digit_capture.py                pure helpers for the interactive digit-capture tool: maps a
                                   mouse pixel coordinate to a tile, stamps a realistically
                                   pen-thick brush stroke (not a single mouse-cell), genuinely
                                   reproduces the reference work's own 32x32-to-8x8
                                   block-counting downsample, and flattens the resulting 8x8
                                   grid into the same state shape digits_data.py produces
  mnist_data.py                   loads the real MNIST dataset: a one-time convert_parquet_to_binary
                                   conversion (pyarrow imported locally, only for that conversion)
                                   to a flat, header-less binary format, then label-only
                                   (load_mnist_labels) and indexed-record (load_mnist_records_at_indices,
                                   direct seek - decodes only what's asked for) readers that
                                   together let ensemble training avoid ever fully decoding the
                                   dataset in any one process (see research-and-analysis.md), plus
                                   load_mnist_dataset_as_array — a bulk numpy decode straight from
                                   raw bytes for the vectorized classes below (see "vectorized
                                   array-based classes")
  ensemble_train.py                builds each class's balanced binary dataset from label data
                                   alone (select_balanced_indices/build_balanced_binary_dataset),
                                   then trains all class_count classifiers as fully independent
                                   multiprocessing jobs with no synchronization between them -
                                   train_ensemble_parallel (small datasets, fully in memory) and
                                   train_ensemble_parallel_from_indices (large datasets - each
                                   worker loads only its own selected records, itself); a
                                   memory-aware worker count (_select_worker_count) caps the pool
                                   by available memory as well as CPU count
  mnist_capture.py                 pure helpers for the interactive MNIST-capture tool: genuinely
                                   reproduces MNIST's own reference preprocessing (crop to
                                   bounding box -> aspect-preserving anti-aliased scale-to-20 via
                                   area-weighted resampling -> center-of-mass placement into 28x28
                                   - see resize_area_weighted, scale_to_fit, center_of_mass,
                                   place_centered, preprocess_capture), plus a brush-stroke helper
                                   for the 64x64 capture grid (the stamping itself is shared with
                                   digit_capture.py's via capture_common.stamp_brush - only the
                                   grid-size-specific validation differs, see capture_common.py's
                                   own entry above)
  demos/
    menu.py                                      the . cli demo entrypoint: a text REPL that lists every demo
                                                 (from registry.py), lets you pick one by number, prints its
                                                 longer description, runs it, then loops back until you quit -
                                                 or pass a demo number directly (. cli demo 3) to skip the menu
                                                 and run just that one demo
    registry.py                                  the DEMOS list menu.py renders and runs: each entry's module
                                                 path, title, one-line summary and longer description
    capture_app.py                                the shared tkinter capture UI (CaptureApp/CaptureConfig/
                                                 run_capture_demo) demo_uci_digit_capture.py and
                                                 demo_mnist_ensemble_capture.py both configure and call
    demo_minimum_disturbance_training.py         standalone script that trains a classifier and plots the
                                                 result, showcasing the minimum-disturbance multi-unit learning
                                                 rule at cardinality=4
    demo_linear_classifier_cardinality_sweep.py  trains classifiers at several cardinalities and compares
                                                 convergence
    demo_unreachable_class_safety_guard.py       shows random_alternating_training_data's max_attempts guard
                                                 tripping
    demo_xor_linear_classifier_ceiling.py        trains against a target no required_active/cardinality can
                                                 represent
    demo_xor_backprop_convergence.py             trains a BackpropClassifierNetwork on the same target - and
                                                 converges
    demo_backprop_stripes_architecture_sweep.py  compares single- vs two-hidden-layer
                                                 BackpropClassifierNetworks at matched node budgets, on a
                                                 harder-than-XOR striped target
    demo_backprop_circular_boundary.py           trains against a circular target - a genuinely curved
                                                 boundary, unlike any LinearClassifierNetwork's polygon
    demo_backprop_linear_parity_check.py         parity check: both models on the same easy, linearly-separable
                                                 target
    demo_uci_digit_recognition.py                classic-style handwritten digit recognition on the bundled 8x8
                                                 dataset, with confusion-matrix/sample charts and a saved,
                                                 reloadable trained model
    demo_uci_digit_capture.py                    an interactive 32x32 mouse-painted grid, genuinely downsampled
                                                 to 8x8 the same way the reference work's own preprocessing
                                                 did, classified live by the trained model saved above (see the
                                                 "multi-class" section above)
    demo_mnist_ensemble_recognition.py           real MNIST (28x28, 60000/10000) recognition via
                                                 EnsembleBackpropClassifierNetwork - 10 independently,
                                                 parallel-trained one-vs-rest classifiers (see "multi-class"
                                                 below and research-and-analysis.md)
    demo_mnist_ensemble_capture.py               an interactive 64x64 mouse-painted grid, genuinely reproducing
                                                 MNIST's own crop/scale/center-of-mass preprocessing
                                                 (mnist_capture.py), classified live by the ensemble trained
                                                 above
    demo_backprop_variant_comparison.py          reproduces three research-and-analysis.md A/B comparisons
                                                 (one-vs-rest vs softmax, quadratic vs binary cross-entropy,
                                                 fan-in-aware vs Xavier/Glorot init) live, side by side, with
                                                 charts - see "backprop siblings" and demos.md
    demo_vectorized_uci_digit_recognition.py     trains MultiClassBackpropClassifierNetwork and
                                                 VectorizedMultiClassBackpropClassifierNetwork on the same
                                                 UCI digits split, same seed/hyperparameters, reporting
                                                 accuracy and wall-clock side by side (see "vectorized
                                                 array-based classes")
    demo_vectorized_mnist_recognition.py         the same pure-Python-vs-numpy comparison, one real epoch
                                                 over the full 60000-example MNIST training set (see
                                                 "vectorized array-based classes")
    demo_rust_vs_vectorized_uci_digit_recognition.py
                                                 the UCI digits comparison above, extended with
                                                 RustArrayMultiClassBackpropClassifierNetwork, the
                                                 production backend (see "the Rust array core" below)
    demo_rust_vs_vectorized_mnist_recognition.py the real-MNIST comparison above, extended the same way
data/
  digits/
    digits.csv                      bundled 8x8 digits dataset (1797 rows, 64 pixels + a
                                     label), extracted offline from sklearn.datasets.load_digits() -
                                     scikit-learn was never a runtime dependency
tests/                           one file per module under test, plus test_training_pipeline.py for
                                  end-to-end coverage; all headless (matplotlib `Agg` backend, no
                                  windows shown)
  helpers.py                     shared test fixtures (e.g. a classifier with a known, hand-built
                                  bounded positive region)
  test_model.py                  LinearClassifierNetwork construction, learning rule, randomized()
  test_geometry.py               square_bounds, is_positive_region_bounded,
                                  reference_positive_region_polygon, positive_region_bounding_box
  test_train.py                  training-data generation, reachable_reference_and_training_data,
                                  train_linear_classifier_network's pocket-tracking diagnostic
                                  (converged/plateaued/still_improving)
  test_evaluate.py               sample_class_balanced_states, class_balanced_disagreement_rate,
                                  compare_on_random_point, smoothed_series
  test_chart.py                  reference_region_bounds, disagreement_axis_bounds,
                                  plot_training_data, plot_linear_classifier_network
  test_training_pipeline.py      end-to-end training + convergence + decision-boundary plotting
                                  (mirrors what demo_minimum_disturbance_training.py does, minus the windows)
  test_backprop_model.py         BackpropClassifierNetwork construction, hand-computed forward/
                                  backward pass, randomize() symmetry-breaking, snapshot/restore
  test_backprop_training_pipeline.py  proves train_linear_classifier_network drives a
                                       BackpropClassifierNetwork well past the linear ceiling on XOR,
                                       unchanged
  test_digits_data.py             load_digits_dataset, split_train_test
  test_multiclass_backprop_model.py  MultiClassBackpropClassifierNetwork construction,
                                     hand-computed one-vs-rest forward/backward pass, fan-in
                                     init, snapshot/restore, save()/load()
  test_multiclass_evaluate.py     confusion_matrix, accuracy
  test_multiclass_training_pipeline.py  proves train_linear_classifier_network drives
                                     MultiClassBackpropClassifierNetwork on real digit data,
                                     unchanged
  test_digit_capture.py           tile_grid_to_state, pixel_to_tile, downsample_to_target_grid,
                                   paint_brush_stroke, intensity_to_color - the pure logic behind
                                   demo_uci_digit_capture.py; the tkinter mouse/canvas code itself
                                   isn't unit-tested (no headless display in this test suite)
  test_mnist_data.py              load_mnist_dataset/labels/records_at_indices, convert_parquet_to_binary -
                                   needs the real (gitignored, locally-supplied) MNIST data files, see
                                   setup.md
  test_mnist_capture.py           bounding_box, crop, resize_area_weighted, scale_to_fit, center_of_mass,
                                   place_centered, preprocess_capture, paint_brush_stroke - the pure logic
                                   behind demo_mnist_ensemble_capture.py
  test_ensemble_backprop_classifier_network.py  EnsembleBackpropClassifierNetwork construction,
                                   predict_probabilities/classify_state, snapshot/restore, save()/load()
  test_ensemble_train.py          select_balanced_indices/build_balanced_binary_dataset,
                                   train_ensemble_parallel(_from_indices) via real multiprocessing.Pool
                                   runs, _select_worker_count's memory-aware capping
  test_softmax_output_layer.py, test_softmax_multiclass_backprop_model.py,
  test_softmax_multiclass_training_pipeline.py, test_fan_in_aware_backprop_model.py,
  test_binary_cross_entropy_backprop_model.py, test_relu_layer.py, test_relu_backprop_model.py,
  test_momentum_layer.py, test_momentum_backprop_model.py, test_l2_regularization_layer.py,
  test_l2_regularized_backprop_model.py, test_adam_layer.py, test_adam_backprop_model.py
                                   one file per backprop sibling class and its underlying
                                   node/layer machinery (see "backprop siblings"), each with the same
                                   hand-computed-forward/backward-pass pattern as test_backprop_model.py
  test_backprop_layer.py, test_backprop_constructor_validation.py  shared BackpropLayer behavior
                                   and the parametrized constructor-validation checks every
                                   BackpropNetworkBase sibling's __init__ shares
  test_gradient_accumulation.py, test_learn_batch.py, test_train_mini_batch.py  mini-batch gradient
                                   descent: node-level accumulate_gradient/apply_accumulated_gradient
                                   parity against the per-example apply_gradient path, learn_batch on
                                   the classifier networks, and train.train_backprop_network_mini_batch
                                   (see "mini-batch gradient descent")
  test_conv_kernel.py, test_conv_unit.py, test_conv_layer.py,
  test_conv_multiclass_backprop_model.py  ConvKernel weight/gradient arithmetic, ConvUnit forward/
                                   backward against hand-computed values, ConvLayer's receptive-field
                                   wiring and per-layer accumulate/apply/snapshot hooks, and
                                   ConvMultiClassBackpropClassifierNetwork end to end against a UCI
                                   digits baseline (see "convolutional layer")
  test_array_layer.py, test_vectorized_multiclass_backprop_model.py  ArrayLayer's forward/backward/
                                   gradient formulas parity-checked against BackpropNode's per-node
                                   reference, and VectorizedMultiClassBackpropClassifierNetwork end to
                                   end (learn, learn_batch, randomize, snapshot/restore, save/load) -
                                   see "vectorized array-based classes"
cli                                setup / test / clean helper script
```

`LinearClassifierNetwork` composes:

- an **input layer** (`StateLayer`) of raw input values,
- a **hidden layer** of `cardinality` `AssociationNode`s, each fully connected to the input layer,
- a single-node **output layer** that fires once at least `required_active` of the hidden
  layer's nodes are active (via weights of 1 and a threshold of `-(required_active - 1)`).
  `required_active` defaults to `cardinality` (AND: every hidden node must agree); passing 1
  gives OR (any one is enough), and anything in between gives a general k-of-n gate, in the
  spirit of a MADALINE-style committee machine. `geometry.py`'s positive-region functions
  only support the AND case (`required_active == cardinality`) and `dimension == 2` - the
  true positive region under any other gate is a union of intersections, not a single
  intersection, and a higher dimension needs more than just the first two weights it reads -
  and those functions raise rather than silently return a wrong answer if called on one.

Each `AssociationNode` updates via the perceptron learning rule
(`w += learning_rate * (reference - actual) * input`), and `train.py` drives this over a
shuffled, class-balanced set of training examples generated from a reference classifier. On
a misclassification, `LinearClassifierNetwork.learn()` doesn't update every hidden node —
it picks the single node closest to flipping (smallest `|z()|`) among those responsible for
the error, a minimum-disturbance rule in the spirit of Widrow's MADALINE, needed once
`cardinality > 1` so hidden nodes can specialize into different half-planes instead of all
converging to the same one. This selection is combination-gate-agnostic: it only relies on
the output being a monotonically non-decreasing function of how many hidden nodes are
active, which holds for AND, OR, or any k-of-n `required_active` - so `learn()` needed no
changes to support gates other than AND.

Unlike the single-neuron case (see [theory](theory.md)), this
has no convergence guarantee analogous to Rosenblatt's theorem — there's no proof it finds a
matching set of hyperplanes in finite steps, or at all, for an arbitrary target polytope. In
practice it converges well for modest cardinality (see `tests/test_training_pipeline.py`'s
cardinality=2 case), but that's an empirical observation, not a theorem.

Because of that, `train.train_linear_classifier_network` uses a pocket-algorithm-style
"keep the best, not the latest" rule: training accuracy on `training_data` is measured after
every epoch, and the student is left at whichever epoch's weights scored best, not
necessarily the raw last epoch's - a no-op when training does converge (the best epoch is
then the last one), and a real difference when it doesn't (see
[demos](demos.md#demo-xor-linear-classifier-ceiling)). It returns a `ConvergenceSeries` - the
same list of `(iteration, disagreement_rate)` pairs it's always returned, so every existing
use keeps working unchanged, plus a `.diagnostic` (a `TrainingDiagnostic`) that answers,
without needing to eyeball a chart, whether that run `.converged` (every training example
correctly classified), `.plateaued` (the best epoch wasn't the last one - more epochs
already stopped helping), or was `.still_improving` (the last epoch was still the best one
seen, just not there yet).

`LinearClassifierNetwork.randomize()` scales each hidden node's weight range inversely with
its own dimension's `input_bounds` half-width (and leaves the threshold range fixed) so that
a random hidden node has a similar chance of splitting the input space regardless of how
large, small, or asymmetric `input_bounds` is - a fixed weight range would let the threshold
dominate at small bounds scales, making almost every random classifier permanently one
class.

`evaluate.sample_class_balanced_states` (used by both `random_alternating_training_data` and
`class_balanced_disagreement_rate`) draws positive-class points from a tight box around the
classifier's own positive region (`geometry.positive_region_bounding_box`) instead of
rejection-sampling all of `input_bounds`, when that box is computable. A classifier's positive
region can be a tiny fraction of `input_bounds` - measured empirically: often under 1% of the
box's area at cardinality 4 with the bounded-region requirement, as low as 0.02%, shrinking
further as cardinality grows - so naive full-box rejection sampling can need far more attempts
than `max_attempts` allows, wasting the whole budget on candidates a tighter box would have
made trivial. Falls back to `input_bounds` (byte-for-byte the previous behaviour) whenever no
tight box is computable - cardinality 1-2 (never bounded), a non-AND `required_active`, or a
classifier that isn't a `LinearClassifierNetwork` at all (both functions accept anything with
the same `input_bounds`/`classify_state` interface, e.g. `demo_xor_linear_classifier_ceiling.py`'s
`XORTarget`).

## backprop

`BackpropClassifierNetwork` is an additive alternative to `LinearClassifierNetwork`, not a
retrofit of it - `AssociationNode`'s hard step function and discrete minimum-disturbance update
rule are fundamentally different from gradient-based learning. It composes `BackpropNode`s
(sigmoid activation, `a = 1/(1+e^-z)`) into `BackpropLayer`s of arbitrary depth
(`input -> hidden layer(s) -> a trainable single-node output layer`). Unlike
`LinearClassifierNetwork`'s output layer (fixed weights of `1.0` per hidden node, so its output
can only be a monotonically non-decreasing function of how many hidden nodes fire -
see `demo_xor_linear_classifier_ceiling.py`), every layer here is trained, including the output layer,
so a hidden node can push the output either way. That's what lets it represent targets like XOR
that no `required_active`/`cardinality` combination can (see `demo_xor_backprop_convergence.py`).

Learning uses mean-squared-error, back-propagated by hand (no autodiff): at the output node,
`delta = (a - y) * a * (1-a)`; at a hidden node, `delta = (Σ downstream delta * weight) *
a * (1-a)` - the same shape at every layer, one rule applied throughout, in the same
first-principles spirit as [theory](theory.md). Each node caches its activation via an explicit
`forward()` pass (`value()` is a pure cache read) so a downstream node's multiple reads of an
upstream node don't each re-pay for a fresh sigmoid evaluation. `randomize()` must give every
node in a layer independent random weights - unlike `AssociationNode`'s harmless identical
default weights, identical starting weights here would give every node in a layer identical
gradients forever, collapsing it to one effective unit.

`train_linear_classifier_network` (see above) trains a `BackpropClassifierNetwork` completely
unchanged - it only ever calls `.learn()`, `.snapshot()`/`.restore()` (renamed from
`LinearClassifierNetwork`'s original `hidden_layer_snapshot`/`restore_hidden_layer` for exactly
this reason), and `.classify_state()`, all of which both classes implement. The two classes'
`.snapshot()` shapes differ (flat, hidden-layer-only for `LinearClassifierNetwork`; nested,
covering every trainable layer including the output layer for `BackpropClassifierNetwork`) but
the training loop only ever treats the snapshot as opaque, so this is invisible to it.

## backprop siblings

Six additive siblings of `BackpropClassifierNetwork` exist alongside it - each isolating one
axis of variation (init scheme, hidden-layer activation, output loss, optimizer, regularization),
none replacing it or changing any existing demo, and each (except the newest, not yet measured -
see below) backed by a real measurement in [research and analysis](research-and-analysis.md)
rather than assumed to help:

- **`FanInAwareBackpropClassifierNetwork`** swaps in the same fan-in-aware `randomize()` scheme
  `MultiClassBackpropClassifierNetwork` already uses (below) instead of
  `BackpropClassifierNetwork`'s own per-dimension-bounds-width scaling. This is the one adopted
  by default: `EnsembleBackpropClassifierNetwork` (below) is built from this class, not plain
  `BackpropClassifierNetwork` - switching alone, no other change, took the real MNIST ensemble
  from 89.4% to 96.01% held-out test accuracy (see "the ensemble/real-MNIST investigation").
- **`ReLUBackpropClassifierNetwork`** replaces sigmoid with ReLU (`max(0, z)`, see `relu_layer.py`)
  for every hidden layer only - the output layer is untouched, since ReLU is a hidden-layer-only
  convention. At `BackpropClassifierNetwork`'s own tuned learning rate it loses badly (dead ReLU
  units were checked directly as the obvious explanation, and ruled out); retuned to its own
  learning rate, it doesn't just recover, it exceeds the sigmoid baseline (99.27% vs 97.80% mean
  on a fixed XOR scenario - see "ReLU hidden-layer activation").
- **`BinaryCrossEntropyBackpropClassifierNetwork`** replaces quadratic loss with binary
  cross-entropy at the output node (`self.delta = self.value() - reference_value`, no `a(1-a)`
  factor) - the canonical loss for binary classification. Needs its own, substantially lower,
  learning rate to match (not beat) the quadratic baseline's tuned performance - see "binary
  cross-entropy for BackpropClassifierNetwork".
- **`MomentumBackpropClassifierNetwork`** adds the momentum term from Rumelhart, Hinton &
  Williams (1986)'s own generalized delta rule, which `apply_gradient` never had before this.
  Sets both `hidden_layer_cls` and `output_layer_cls` - unlike the three above (each touching
  only one layer or the other), this changes the weight-update rule itself, shared by every
  trainable layer. Unlike the other four, no `momentum` coefficient tested actually beat plain
  SGD - the constructor requires an explicit value rather than defaulting to one, and this class
  was kept mainly for a future mini-batch-gradient investigation, where momentum's own literature
  is more commonly validated (see "momentum").
- **`L2RegularizedBackpropClassifierNetwork`** sets both hooks the same way, for the same
  reason - it adds an `l2_lambda * weight` penalty to every weight's gradient (never bias -
  standard practice). Measured on a small, fixed, finite proxy dataset rather than a toy
  geometric target, since L2's whole purpose is generalization: no coefficient tested improved
  held-out accuracy above the unregularized baseline, though the mechanism itself was confirmed
  directly (a strong enough penalty collapses the network to a constant prediction, weights
  decayed to near-zero - see "L2 weight regularization").
- **`AdamBackpropClassifierNetwork`** sets both hooks the same way too - a per-parameter
  adaptive learning rate (Kingma & Ba, 2014), driven by bias-corrected running estimates of each
  weight's own gradient mean and variance, rather than momentum's single shared velocity term.
  Unlike `Momentum...`/`L2Regularized...`, its `beta1`/`beta2`/`epsilon` default to Kingma & Ba's
  own published values rather than requiring an explicit value - see `adam_layer.py`'s own
  docstring for why (they're closer to fixed algorithmic constants in real-world Adam usage than
  a knob this codebase's measurements have an opinion on, unlike momentum's coefficient).
  **Built and correctness-tested (hand-derived regression fixtures, the same convention as every
  sibling above) but not yet measured against the SGD/momentum baseline** - see
  [Adam optimizer](adam-optimizer.md) for the measurement plan.

Of these six, two (`FanInAware...`, `ReLU...`) are genuine, measured improvements over the
plain sigmoid/quadratic-loss/no-momentum/no-regularization baseline; one
(`BinaryCrossEntropy...`) matches it once retuned; two (`Momentum...`, `L2Regularized...`) are
kept as real, tested capabilities despite measuring as nulls on the scenarios tested, not
because either is recommended for use today; `Adam...` is a real, correctness-tested capability
whose own measurement is still open (see [Adam optimizer](adam-optimizer.md)).
`demo_backprop_variant_comparison.py` reproduces three of these comparisons (one-vs-rest vs
softmax, quadratic vs binary cross-entropy, fan-in-aware vs Xavier/Glorot init - see
[demos](demos.md#demo-backprop-variant-comparison)) live and re-runnably, rather than leaving the
documented numbers only readable.

## multi-class

`MultiClassBackpropClassifierNetwork` is a one-vs-rest multi-class sibling of
`BackpropClassifierNetwork`, built entirely on the same `BackpropNode`/`BackpropLayer` blocks -
a new class rather than a retrofit, because `classify_state()`'s return type changes (a class
index, not a 0.0/1.0 float), and every existing backprop demo depends on the binary contract.
Its output layer has `class_count` nodes instead of one; each is trained independently against
a one-hot target (`BackpropNode.compute_output_delta` needed no changes for this - it was
already a per-node, sibling-independent computation), and the predicted class at inference is
whichever output node has the highest activation.

Its `randomize()` uses **fan-in-aware** initialization (`limit = 1/sqrt(fan_in)`) rather than
`BackpropClassifierNetwork.randomize()`'s per-dimension-bounds-width scaling - that scaling was
tuned for 1-2D geometric problems and produces exploding pre-activation sums (guaranteed sigmoid
saturation at every node) once fan-in reaches the tens or hundreds, as it does for a 64-pixel
digit image. Also new: `save()`/`load()` (plain JSON) - trained-model persistence, so a trained
network can be reused without retraining.

`SoftmaxMultiClassBackpropClassifierNetwork` is an additive sibling of
`MultiClassBackpropClassifierNetwork` using softmax + cross-entropy instead of one-vs-rest
sigmoid + quadratic loss - the canonical treatment for a mutually-exclusive multi-class target
like digit classification (see `docs/research-multiclass-and-loss.md`'s "softmax/cross-entropy
re-alignment" entry for the full derivation and measured comparison). Structurally it's a single
class-attribute override (`output_layer_cls = SoftmaxOutputLayer`, see
`indrajala_ml/model/softmax_output_layer.py`) - softmax's cross-node coupling only touches the
forward pass (each node's activation needs every sibling's pre-activation `z`), so
`BackpropNetworkBase`'s existing backward-pass plumbing needed no changes at all; the
softmax+cross-entropy output delta (`activation - target`) is exactly as per-node-independent
as `MultiClassBackpropClassifierNetwork`'s own one-vs-rest delta. `EnsembleBackpropClassifierNetwork`
(below) is deliberately untouched by this - its own one-vs-rest design is a different,
independently-motivated tradeoff (parallelizability across completely independent processes),
not a literature-alignment gap.

`digits_data.py` bundles a small, classic dataset - the UCI ML hand-written digits set (8x8
pixel images, 10 classes, 1797 samples), extracted once, offline, from
`sklearn.datasets.load_digits()` into `data/digits/digits.csv`. scikit-learn was only ever a
throwaway extraction tool; there's no runtime dependency on it, just a flat CSV and a small
parser. This is a fixed, finite, already-labeled dataset, unlike every other target in this
codebase (which are all continuously re-sampleable geometric regions) - hence `split_train_test`
(new) and `multiclass_evaluate.py`'s `confusion_matrix`/`accuracy` (new; `evaluate.py`'s
functions are two-class- and geometry-specific and don't generalize here).
`train_linear_classifier_network` still needs no changes at all - it only calls `.learn()`,
`.snapshot()`/`.restore()`, and `.classify_state()` (via `_training_accuracy`'s bare `==` check,
which works identically whether `category` is a float or an int).

`demo_uci_digit_capture.py` classifies a live, mouse-painted digit with a trained model loaded from
disk (`MultiClassBackpropClassifierNetwork.load`). Rather than painting an 8x8 grid directly (an
earlier version of this tool did, first with binary tiles, then with an ad-hoc soft-brush
heuristic to fake grading), it genuinely reproduces the reference work behind the bundled
training data: the dataset's own description (`sklearn.datasets.load_digits()`) states "32x32
bitmaps are divided into nonoverlapping blocks of 4x4 and the number of on pixels are counted in
each block" to produce the 8x8, 0-16-graded shape `data/digits/digits.csv` actually contains. So
the tool captures a binary 32x32 bitmap by mouse (mirroring NIST's own thresholded scan) and
`digit_capture.downsample_to_target_grid` genuinely block-counts it down the same way, rather
than approximating grading with a heuristic - shown live in a second preview canvas (rendered via
`digit_capture.intensity_to_color`, a linear grayscale mapping) so what the classifier actually
sees is visible. The result still goes through `digit_capture.tile_grid_to_state` - the same
row-major, `[0.0, 1.0]`-normalized shape `digits_data.py` produces - unchanged by this.

Painting a single 32x32 cell per mouse event isn't enough, though - measured directly: a
mouse-thin (1-cell-wide) stroke block-counts down to only ~25% of full intensity, far fainter
than any real training example (which run ~14-16 out of 16 throughout their stroked regions,
since a real pen stroke is proportionally thick relative to the 32x32 capture resolution, not
single-pixel-thin). This was a real, reported bug - almost every digit misclassified toward
whichever class happened to catch faint, ambiguous input (in practice, nearly always "7") -
fixed by `digit_capture.paint_brush_stroke`, which stamps a `CAPTURE_BRUSH_RADIUS`-wide square
per stroke instead of one cell. That radius (2, a 5x5 stamp) was chosen empirically: wide enough
to reach near-full block intensity, but not so wide it fills in a real digit's negative space -
e.g. an unfilled "0"'s hole, which a radius of 3+ visibly started doing in testing.

`demo_mnist_ensemble_recognition.py` trains on the real, full-scale MNIST dataset (28x28, 60000 train /
10000 test images) rather than the small bundled UCI set, using
`EnsembleBackpropClassifierNetwork` instead of `MultiClassBackpropClassifierNetwork` - 10
completely independent `FanInAwareBackpropClassifierNetwork`s (see "backprop siblings" above; a
`classifier_cls` parameter on `ensemble_train.py`'s training functions, defaulting to plain
`BackpropClassifierNetwork`, is what lets this demo opt into the fan-in-aware sibling without
touching any other caller), one per digit, each trained on its own class-balanced binary dataset
(`ensemble_train.build_balanced_binary_dataset`) with no shared hidden layer and no
synchronization of any kind between them, dispatched as parallel `multiprocessing` jobs
(`ensemble_train.train_ensemble_parallel_from_indices`). This design, and the real
memory-exhaustion bug hit (and fixed) while building it at full scale, is written up in
[research and analysis](research-multiclass-and-loss.md#parallelizing-mnist-training). The switch to
fan-in-aware init - the same fix "backprop siblings" describes - took this demo's own measured
result from 89.4% to 96.01% held-out test accuracy at the same wall-clock cost (~30 minutes,
measured directly at 29.6 minutes for the current configuration - see [research and
analysis](research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)).

`demo_mnist_ensemble_capture.py` classifies a live, mouse-painted digit with `EnsembleBackpropClassifierNetwork.load`
(the model the demo above trains and saves), the same overall interaction as
`demo_uci_digit_capture.py` but against MNIST's own reference preprocessing instead of the UCI
dataset's block-counting downsample: the user paints a binary 64x64 bitmap
(`mnist_capture.CAPTURE_GRID_SIZE`, deliberately higher resolution than the UCI tool's 32x32, so
there's real room for aspect-preserving scaling to do something meaningful), then
`mnist_capture.preprocess_capture` genuinely reproduces MNIST's own three-step pipeline - crop to
the drawn content's bounding box, aspect-preserving anti-aliased scale so the longer side hits 20
pixels, center-of-mass placement into a 28x28 field - shown live in a second preview canvas.
`CAPTURE_BRUSH_RADIUS` (4, a 9x9 stamp) was chosen empirically against the real trained ensemble
the same way `digit_capture.CAPTURE_BRUSH_RADIUS` was: tested against hand-simulated "0"/"1"/"7"
strokes, confidence stayed at ~1.00 through radius 2-5, degraded from radius 6 onward as a "0"'s
hole started filling in (0.94 at 6, 0.34 at 8, misclassified as "8" by 12) - 4 sits comfortably in
the safe range. Building this demo surfaced a real floating-point bug: `resize_area_weighted`'s
area-weighted average can overshoot its mathematically-guaranteed `[0.0, 1.0]` bound by a tiny
amount (observed directly: `1.0000000000000002`) from summing many small overlap contributions,
which `intensity_to_color`'s strict range assertion then rejected - `scale_to_fit` now clamps its
result (not the general-purpose `resize_area_weighted` itself, whose valid output range depends
on its caller's own input range).

## convolutional layer

`ConvMultiClassBackpropClassifierNetwork` places one `ConvLayer` directly after the
(non-trainable) input `StateLayer`, followed by ordinary dense hidden/output layers - a sibling
of `MultiClassBackpropClassifierNetwork`, not a retrofit (its `__init__` doesn't call
`super().__init__()`; it builds `hidden_layers`/`output_layer`/`trainable_layers` directly). It
encodes the one structural fact dense layers throw away for image data: pixels have 2D spatial
relationships, and the same local pattern can appear anywhere in the image. A `ConvLayer` holds
`channel_count` `ConvKernel`s (each owning `kernel_size**2` shared, trainable weights plus a
bias) and `out_height * out_width * channel_count` `ConvUnit`s, one per output spatial position
per channel; each `ConvUnit` connects only to its own `kernel_size x kernel_size` receptive-field
window of the input (single input channel, `'valid'` padding, configurable stride) and computes
against its channel's shared `ConvKernel` weights rather than owning independent weights the way
`BackpropNode` does. `ConvUnit` is composition, not inheritance (`AbstractNode`, not
`BackpropNode`), since a genuinely shared, cross-instance weight list doesn't fit
`BackpropNode.input_node_weights`'s owned-and-rebindable design.

Weight sharing is why `BackpropNetworkBase`'s gradient/persistence methods
(`_accumulate_gradients`, `_apply_accumulated_gradients`, `_apply_gradients`, `snapshot`,
`restore`) dispatch through a per-*layer* hook (`BackpropLayer.accumulate_gradients()` etc.)
instead of reaching into `layer.nodes` directly - calling `apply_accumulated_gradient` once per
spatial position would apply the same kernel's update up to `out_height * out_width` times per
step. `BackpropLayer`'s default implementations are exactly the old per-node loop bodies (every
other sibling layer is unaffected); `ConvLayer` overrides them to iterate `ConvUnit`s for
accumulation (summing every position's contribution into its kernel) but `ConvKernel`s (once
each) for apply/snapshot/restore.

Validated on UCI digits at a comparable trainable-parameter budget (2410 dense vs. 2530 conv, 8
seeds each): mean test accuracy came out statistically indistinguishable (96.69% dense vs. 96.52%
conv) - a clean null, not a win, honestly reported rather than stretched into one (see
[research and analysis](research-backprop-siblings.md#convolutional-layers-on-uci-digits)). Stacked
conv layers, multi-channel input, `'same'` padding, and pooling are all out of scope - a single
conv layer directly after the non-trainable input needs no backprop-through-convolution, since
nothing before it is ever trained.

## vectorized array-based classes

`VectorizedMultiClassBackpropClassifierNetwork` rewrites the forward/backward/gradient math
around whole-layer numpy arrays instead of individual node objects: `ArrayLayer` holds a weight
matrix `W` (shape `(size, input_size)`) and bias vector `b`, not `size` separate node objects,
and every formula that was a per-node Python loop becomes one array operation (`forward`:
`self.a = sigmoid(self.W @ x + self.b)`; `compute_hidden_delta`: one matmul, `next_layer.W.T @
next_layer.delta`, replacing a per-node `sum()`; `accumulate_gradient`: `np.outer(self.delta,
input_layer.a)`). This can't reuse `BackpropNetworkBase`'s per-node orchestration at all -
there's no `own_index` to enumerate - so `VectorizedMultiClassBackpropClassifierNetwork` is a
fully standalone class, sharing only the external contract (`learn`, `learn_batch`,
`classify_state`, `predict_probabilities`, `snapshot`/`restore`, `save`/`load`) every sibling
network already shares, with its own JSON envelope (arrays via `.tolist()`).

Built and parity-checked against the pure-Python reference classes at every stage, then measured
end to end: UCI digits (`[32]` hidden, 30 epochs) trains to matching accuracy (99.5%/96.1% vs.
99.4%/96.7% train/test) in 8.57s vs. 77.16s pure-Python - **9.00x**. One real epoch over the full
60000-example MNIST training set (`[30]` hidden) measures **33.87x** (32.7s vs. 18.45 min), with
`load_mnist_dataset_as_array`'s bulk numpy decode itself **15.44x** faster than the existing
tuple-per-example decode (0.41s vs. 6.32s) - see
[vectorized array-based classes](vectorized-array-classes.md#measured-results) for the full
numbers and parity methodology.

`numpy` is kept permanently here, scoped specifically to this role: a performance-benchmarking
mirror any array backend's speed claim gets measured against, not this codebase's production
array backend. [A hand-built Rust core](rust-array-core.md) (`rust/indrajala_ml_array/`, wrapped
via PyO3) implements the same operation subset ([the numpy interface
subset](numpy-interface-subset.md)) this class needs, is parity-tested against real numpy across
618 tests (525 at this phase's own completion; the growth since is [the Adam array-layer
sibling](adam-array-layer.md)'s own stage 5 fused-op tests, a later addition), and backs
`RustArrayLayer`/`RustArrayMultiClassBackpropClassifierNetwork`
(`indrajala_ml/model/rust_array_layer.py`,
`indrajala_ml/model/rust_array_multiclass_backprop_classifier_network.py`) - the actual production
path, per [the production cutover plan](rust-production-cutover.md)'s now-completed phases 0-2:
a mechanical loop-reorder and fused-per-layer-call fix to the Rust core itself, then a class
mirroring `VectorizedMultiClassBackpropClassifierNetwork`'s exact external contract, tier-1
bit-close parity-checked against the pure-Python reference
(`tests/test_rust_array_multiclass_backprop_model.py`), then measured on real training runs: a
genuine speedup over numpy, first measured at 3.40x at UCI digits scale and 1.31x at real MNIST
scale (see [research and
analysis](research-rust-performance.md#phase-2-tier-2-real-per-example-training-is-a-genuine-win-at-both-scales-measured)),
with statistically indistinguishable accuracy. `demo_rust_vs_vectorized_uci_digit_recognition.py`/
`demo_rust_vs_vectorized_mnist_recognition.py` run all three (pure-Python, numpy, Rust) side by
side.

Two further SIMD optimization passes (both 2026-09-16, see [research and
analysis](research-rust-performance.md#explicit-simd-intrinsics-a-real-further-win-with-fused-multiply-add-kept-consistent-across-every-path)
and
[research and analysis](research-rust-performance.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)
for the full writeups) moved that number since. Neither training path above needs `batch_size >= 32`
mini-batch matmul performance - both use `learn()`'s per-example (`batch_size=1`) shape
exclusively - but that gap against numpy's BLAS was closed anyway: cache-blocking, threading, and
an AVX2+FMA SIMD path on the crate's 2D×2D matmul case moved `batch_size=512`'s Rust/numpy ratio
to 0.84x-1.04x. More consequentially, the `Matrix @ Vector` case (`self.W @ x` - the shape
`batch_size=1` training *actually* uses, initially assumed not worth optimizing at this scale,
then measured directly and found to be ~97% of a fused forward call's cost) got the same AVX2+FMA
treatment, moving the real per-example training-run win to **~3.6x** at UCI digits scale and
**~2.6x** at real MNIST scale - the current numbers, still with statistically indistinguishable
accuracy.

## possible next steps

What's next for this codebase, given the architecture and measured results described above (see
[research and analysis](research-and-analysis.md) for the investigations behind each decision).
Grouped by [goals and strategy](goals-and-strategy.md)'s own three-tier priority order (new
measured primitives, then deepening existing work, then rigor-protecting infrastructure) - see
that document for the reasoning behind the ordering itself; within a tier, items aren't otherwise
ranked.

### new model primitives, measured the existing way

- **Dropout** - `DropoutBackpropClassifierNetwork` is built and correctness-tested (see
  [dropout](dropout.md)'s stage 2), but its actual regularization effect on this codebase's own
  benchmark is a known, flagged gap, not a measured result: the overfitting-gap sweep was
  attempted (a real 50-run measurement, launched against an 8-way `multiprocessing.Pool`) and
  stopped before completion once it was observed running ~5x slower than serial calibration
  predicted, on track for 40+ minutes - the only path this sibling has today is the per-node one
  (no vectorized/Rust-matmul-backed counterpart exists, the same launch-scope precedent every
  sibling except Adam follows). Closing this gap needs either tolerating that wall-clock cost
  outright or building a vectorized dropout sibling first, mirroring [the array-based Adam
  sibling](adam-array-layer.md)'s own precedent - neither committed to here. See
  [dropout](dropout.md#the-measurement-gap---not-run-deliberately-not-silently-dropped) for the
  full reasoning.
- **Batch or layer normalization** - no normalization exists at all; likely the highest-value item
  in this group once hidden layers get deeper than 1-2, but also the biggest lift of anything
  here (running statistics, a train/eval-mode split - real complexity beyond the existing
  siblings' pattern). Not yet started.

**RMSprop** (Adam minus its momentum-like first-moment term), that used to be listed here, is now
closed - [Adam optimizer](adam-optimizer.md)'s stage 7: measured (no new code needed,
`AdamBackpropClassifierNetwork(..., beta1=0.0)` already *is* RMSprop) against Adam and sigmoid on
a fresh real-MNIST-proxy batch-size sweep. RMSprop tracked Adam within seed-to-seed noise at every
batch size tested - the second-moment normalization alone accounts for Adam's batch-size-
robustness win, its first-moment term adding nothing measurable. Not adopted as a separate
capability; Adam remains the recommended choice. See [research and
analysis](research-adam-optimizer.md#rmsprop-the-second-moment-term-alone-accounts-for-adams-batch-size-win-stage-7-of-the-adam-optimizer-workplan)
for the full measurement.

**A learning-rate schedule** (decay/warmup), that used to be listed here, is also now closed -
[a learning-rate schedule](learning-rate-schedule.md)'s all-five-stages workplan: `lr_schedule.
linear_warmup`, wired into both training loops via a widened `learning_rate: float |
Callable[[int], float]`, then retested against the documented `batch_size=128`/`lr=64.0`
divergence ([the learning-rate-vs-batch-size
follow-up](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)).
Warmup fixes the divergence, most cleanly at `momentum=0.9`/`warmup_steps>=25` (closing back to
the documented `batch_size=32` stable band with tight variance) - see [research and
analysis](research-backprop-siblings.md#the-batch_size128-divergence-retested-with-warmup) for the
full measurement, including a flagged asymmetry (momentum benefits from warmup more than plain SGD
does) not chased further. Scoped to warmup only - decay remains out of scope, no equivalently
concrete failure case motivates it yet.

### deepening what's already here

- **A `shape` property on the network base class(es)** - no built-in way currently to introspect a
  network's structure (dimension, per-layer sizes, hidden-vs-output shape, which sibling
  class/factory is in use) as a single, meaningfully-summarized data structure; every existing
  demo/debugging use reaches into `hidden_layers`/`output_layer`/`trainable_layers` by hand. Not
  yet started.
- **A real-MNIST-scale convolutional layers validation**, now that the UCI-digits-scale result
  came back a flat null rather than a clear loss - real MNIST has meaningfully more spatial
  structure (28x28 vs. 8x8) for convolution's own advantages to potentially show up in, but a
  full run costs on the order of 30 minutes per architecture per seed (see
  [research and analysis](research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)), a
  real wall-clock cost this UCI-digits result alone doesn't settle is worth spending. Not yet
  run.
- **Stacking conv layers, and pooling** - `conv_layer.py`'s own docs (see "convolutional layer"
  above) name both as explicitly out of scope for the current single-conv-layer design, along
  with multi-channel input and `'same'` padding - the current design needs no
  backprop-through-convolution since nothing before its one layer is ever trained; stacking would
  change that. Not yet started.
The array-based (Rust-matmul-backed) Adam sibling that used to be listed here - both the
numpy-backed half (`AdamArrayLayer`/`AdamVectorizedMultiClassBackpropClassifierNetwork`) and the
Rust-matmul-backed counterpart (`AdamRustArrayLayer`/`AdamRustArrayMultiClassBackpropClassifierNetwork`)
- is now closed, all five stages done including the wall-clock/robustness measurement: both
array-based backends measured 70x-794x faster per example than the per-node path (widening with
batch size), and Adam's batch-size accuracy-robustness result confirmed to survive the array/Rust
port unchanged. See [an array-based Adam sibling](adam-array-layer.md) for the full plan and
measured results.

### infrastructure that protects the rigor

- **An audit of hand-rolled measurement scripts for reusable infrastructure** - every real-scale
  measurement in this codebase's own history (Adam's batch-size sweeps, the
  learning-rate-vs-batch-size follow-up, RMSprop's ablation, the warmup-step sweep, dropout's
  overfitting-gap sweep, ...) hand-rolls its own uncommitted one-off script per this codebase's
  own established practice - but several pieces recur near-identically every time: building a
  fixed MNIST-digit-N proxy (200/200 balanced, 80/20 split), a fork-based `multiprocessing.Pool`
  sweep over (config, seed) pairs with per-process reseeding, an epoch-count calibration probe,
  and mean/stdev-into-a-markdown-table aggregation. Worth determining which of these are real,
  tested, permanent fixtures worth building - versus which legitimately stay one-off, shaped by
  each measurement's own specifics - rather than assuming either answer up front. Not yet started
  - not even audited yet.

The three items that used to be here - matmul performance (see [research and
analysis](research-rust-performance.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)),
CI (see [setup](setup.md#ci)), and Python packaging/pinning (see
[setup](setup.md#dependencies)) - are all closed; follow those links for the detail.
