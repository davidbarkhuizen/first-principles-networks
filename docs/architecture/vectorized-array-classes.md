# vectorized array-based classes

[← back to vectorization](vectorization.md)

`indrajala_ml/model/array_layer.py` and
`indrajala_ml/model/vectorized_multiclass_backprop_classifier_network.py` rewrite this codebase's
forward/backward/gradient math around whole-layer arrays instead of individual node objects,
built purely additively (no existing class changed) and validated against real `numpy` as the
concrete array backend - see [structure](../project/structure.md#vectorized-array-based-classes) for the
short version and measured headline numbers. This document is the fuller design reference: why
this needed a standalone class, the exact array formulas, and the numerical-parity discipline
that backs the "identical accuracy trajectory, honestly measured" claim.

## why numpy here, now

[Vectorization](vectorization.md)'s array-core plan (the [Rust implementation](rust-array-core.md))
specifically avoids adopting real `numpy` as a permanent production dependency - a deliberate
values choice for this "hand-build everything" codebase, not a technical necessity. But
*designing and proving out* a whole new class of array-based model classes is a separable
question from *which array implementation eventually backs them* - answering it against a
hand-rolled Rust core that doesn't exist yet would mean debugging two unproven things at once
(the new class architecture, and a brand-new array library) with no working reference to check
either against. Building these classes against real numpy first - already installed on this
machine, already proven correct and fast - let the class design and its numerical parity against
the existing pure-Python reference be proven independently, before
[the Rust core](rust-array-core.md) had to reproduce anything. `numpy` is now kept permanently in
this role: a performance-benchmarking mirror, not this codebase's production array backend (see
[vectorization](vectorization.md#decision)).

## why a standalone class, not a `BackpropNetworkBase` sibling

Every sibling class elsewhere in this codebase - including `ConvLayer`
([convolutional layer](../project/structure.md#convolutional-layer)), which shares weights across many
spatial positions - still represents "one trainable unit" as one Python object with a `.value()`/
`.delta` pair, and reuses `BackpropNetworkBase`'s generic per-node/per-layer orchestration
(`_forward_outputs`, `_backward_hidden_layers`, the accumulate/apply/snapshot hooks). That
machinery is built around `compute_hidden_delta(next_layer_nodes, own_index)` - one call per
node, indexing into a downstream node's own per-node `input_node_weights` list.

Array-based vectorization can't reuse that: the entire point is replacing "one Python object, one
method call, per node" with "one array, one matrix operation, for the whole layer." A layer's
weights become one 2D `ndarray` (`shape=(size, input_size)`), not `size` separate `list[float]`s;
a layer's activations become one 1D or batched-2D `ndarray`, not `size` separate node objects
each caching its own activation. There is no `own_index` to enumerate and no per-node
`compute_hidden_delta` to call - the entire backward pass for a layer is one matrix
multiplication (`W_next.T @ delta_next`), computed once, not `size` times.

`VectorizedMultiClassBackpropClassifierNetwork` is therefore a standalone class, not a
`BackpropNetworkBase` subclass and not built from `ArrayLayer`/`BackpropLayer` composition. It
shares only the *external* contract every sibling network in this codebase already shares -
`learn`, `learn_batch`, `classify_state`, `predict_probabilities`, `snapshot`/`restore`,
`save`/`load` - not any internal implementation. `indrajala_ml/model/backprop_node.py`/
`backprop_layer.py`/`backprop_network_base.py` themselves are untouched.

## class design

### `ArrayLayer` - one layer's weights as one array, not `size` node objects

| method | array formula |
|---|---|
| `__init__(size, input_size)` | `self.W = np.zeros((size, input_size))`, `self.b = np.zeros(size)` - shape, not a list of node objects |
| `forward(x)` (single example) | `self.z = self.W @ x + self.b; self.a = sigmoid(self.z)` (an `_activation` override point, one per network variant, not per node) |
| `forward_batch(X)` (`X.shape == (batch_size, input_size)`) | `self.Z = X @ self.W.T + self.b; self.A = sigmoid(self.Z)` - the same formula, batched, via numpy's own broadcasting of `+ self.b` across every row |
| `compute_output_delta(reference)` (quadratic) | `self.delta = (self.a - reference) * self.a * (1 - self.a)` - elementwise, whole vector at once |
| `compute_hidden_delta(next_layer)` | `downstream = next_layer.W.T @ next_layer.delta; self.delta = downstream * self.a * (1 - self.a)` - one matmul replaces `BackpropNode.compute_hidden_delta`'s per-node Python `sum()` entirely |
| `accumulate_gradient()` (called once per example in a batch) | `self._grad_W += np.outer(self.delta, self.input_layer.a); self._grad_b += self.delta` |
| `apply_accumulated_gradient(learning_rate, batch_size)` | `self.W -= learning_rate * self._grad_W / batch_size; self.b -= learning_rate * self._grad_b / batch_size`, then zero the accumulators |

### `VectorizedMultiClassBackpropClassifierNetwork` - the standalone network

| method | design |
|---|---|
| `__init__(layer_sizes, dimension, class_count)` | builds a list of `ArrayLayer`s, matching `MultiClassBackpropClassifierNetwork`'s own shape (hidden layers, then a `class_count`-wide output layer), no `StateLayer`/`BackpropLayer` involved |
| `_forward(state)` | `x = np.array(state, dtype=np.float64)`, then each layer's `forward(x)` in sequence, `x = layer.a` between them; `forward_batch` is the batched analogue `learn_batch` uses |
| `learn(learning_rate, state, category)` | forward, `compute_output_delta` against a one-hot target array, `compute_hidden_delta` for every earlier layer in reverse, `accumulate_gradient` + `apply_accumulated_gradient(batch_size=1)` per layer |
| `learn_batch(learning_rate, batch)` | stacks every example's state into one `X` matrix and every category into a one-hot `Y` matrix, runs `forward_batch`, computes every layer's delta batched, accumulates once per layer instead of once per (layer, example) pair - `grad_W = (delta_batch.T @ X_batch) / batch_size` replaces a per-example accumulation loop with one matrix multiply |
| `randomize()` | fan-in-aware, `limit = 1/sqrt(previous_size)`, `layer.W = np.random.uniform(-limit, limit, size=(size, previous_size))` - one call per layer instead of a nested Python loop over every weight |
| `classify_state(state)` | `np.argmax(self.predict_probabilities(state))` |
| `snapshot()`/`restore()` | `[(layer.W.copy(), layer.b.copy()) for layer in self.layers]` and the inverse |
| `save()`/`load()` | its own JSON envelope (arrays via `.tolist()`/`np.array(...)`) - can't reuse `save_model_json` any more than `ConvMultiClassBackpropClassifierNetwork` could |

### MNIST data loading

`load_mnist_dataset`/`load_mnist_records_at_indices` decode into `tuple[float, ...]` per example -
the "60000 x 784 = 47 million boxed Python float objects" cost
[research and analysis](../research/research-multiclass-and-loss.md#parallelizing-mnist-training) measured and
worked around by lazy per-worker decoding, not fixed at the root. `mnist_data.py`'s
`load_mnist_dataset_as_array(path, limit=None) -> np.ndarray` (shape `(n, 784)`) addresses the
root cause directly: `np.frombuffer(data, dtype=np.uint8).reshape(n, 785)[:, :-1]
.astype(np.float64) / 255.0` decodes the whole file in one bulk numpy call - a new, additive
function alongside the existing one.

## numerical parity validation

Every method above is checked against the *existing pure-Python reference classes*
(`BackpropNode`/`MultiClassBackpropClassifierNetwork`), not assumed from the formulas looking
equivalent:

- **`sigmoid`'s overflow behavior.** `1/(1+math.exp(-z))`'s current `try/except OverflowError:
  return 0.0` vs. numpy's `exp` returning `inf` (and `1/(1+inf) == 0.0` falling out via IEEE 754)
  - checked with a large random-`z` sweep, not assumed.
- **Summation-order rounding.** `np.outer`/`@`'s internal reduction order isn't guaranteed to
  match Python's own `sum()` - the same category of float64 rounding-order risk that got an
  earlier piecewise-sigmoid attempt reverted in this codebase (see
  [structure](../project/structure.md#backprop)).
- **The required regression gate**: for a fixed random seed, weights, and input,
  `VectorizedMultiClassBackpropClassifierNetwork.learn()` lands on the *same* (or provably
  float64-noise-close) weights as `MultiClassBackpropClassifierNetwork.learn()`, across a large
  randomized sweep, not a single hand-picked case.

## measured results

`tests/test_array_layer.py`/`tests/test_vectorized_multiclass_backprop_model.py`/
`tests/test_mnist_data.py`'s parity tests pass first, so these are a faster path to the *same*
trained result, not a different one:

**UCI digits** (`demo_vectorized_uci_digit_recognition.py`, seed 0, `[32]` hidden layer,
`learning_rate=0.5`, 30 epochs): pure-Python 99.4% train / 96.7% test in 77.16s vs. vectorized
99.5% train / 96.1% test in 8.57s - **9.00x** wall-clock speedup. Accuracy trajectories are close
but not bit-identical, as the parity section above anticipates: float64 summation-order rounding
compounds slightly differently across the ~43000 individual SGD steps this run takes.

**Real MNIST** (`demo_vectorized_mnist_recognition.py`, `[30]` hidden layer, `learning_rate=0.5`,
1 epoch, full 60000 train / 10000 test): pure-Python 93.3% train / 93.0% test in 1107.2s
(18.45 min) vs. vectorized 92.5% train / 92.4% test in 32.7s (0.54 min) - **33.87x** wall-clock
speedup for one epoch. `load_mnist_dataset_as_array`'s bulk decode of the full training file
measured at 0.41s vs. `load_mnist_dataset`'s 6.32s - **15.44x**.

Both real-scale speedups land inside [vectorization](vectorization.md#expected-effect)'s
own "15-45x, treat 150.7x as a ceiling" framing - measured at real, practical batch-size-1 SGD,
not a forward-pass-only microbenchmark.

## what stays explicitly out of scope

- **Any change to `indrajala_ml/model/backprop_node.py`/`backprop_layer.py`/
  `backprop_network_base.py`, or any existing sibling class.** Purely additive.
- **Swapping the array backend from `numpy` to the Rust core.** That's
  [the Rust core](rust-array-core.md)'s own eventual follow-on, gated on that core existing and
  passing its own parity checks first - not something these classes do or assume.
- **Convolutional or other sibling variants of the vectorized classes** (a vectorized `ConvLayer`,
  momentum, L2, etc.). This covers the base one-vs-rest/quadratic-loss case only; every other
  variant would be its own, separate follow-on.
