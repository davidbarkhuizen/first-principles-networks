# the numpy interface subset

[← back to vectorization](vectorization.md)

Part 2 of 3 in the [vectorization](vectorization.md) workplan split: the precise, minimal set
of numpy operations [vectorized array-based model classes](vectorized-array-classes.md) actually
calls - derived directly from that document's own class design, not a speculative general
checklist. This is the formal contract [the Rust implementation plan](rust-array-core.md) needs
to satisfy: nothing more (no point building array operations nothing calls) and nothing less (no
silently-assumed numpy behavior the Rust core forgets to replicate).

## why a derived subset, not a general inventory

`vectorization.md`'s original, single-document version of this workplan scoped its own array
core against a broad checklist gathered from every sibling class in this codebase (ReLU,
softmax, momentum, L2, cross-entropy - see that document's own "function-level inventory"). That
was reasonable for a single, all-at-once workplan, but
[vectorized array-based model classes](vectorized-array-classes.md) deliberately narrows its own
scope to the base one-vs-rest/quadratic-loss case only, leaving every other variant (ReLU,
softmax, momentum, L2) as later, separate follow-ons once the base case is proven (see that
document's own "what stays explicitly out of scope"). This document's own subset narrows to
match - a smaller, more precisely justified target than the original inventory, with every
operation traceable to a specific method in the class design that actually needs it. A future
follow-on covering a specific variant (say, a vectorized ReLU sibling) would extend this list
with exactly what *that* variant needs (`np.maximum`, `np.where`), not before.

This table is derived directly from the real, committed code in `array_layer.py` and
`vectorized_multiclass_backprop_classifier_network.py` - including two operations easy to miss
from the class design alone: `.sum(axis=0)` (`accumulate_gradient_batch`'s batched bias gradient)
and the 2D tuple-index write `target_batch[row, category] = 1.0` (`learn_batch`'s batched
one-hot target, a different index shape than the single-example `target[category] = 1.0` case).
If [the vectorized classes](vectorized-array-classes.md)' design changes - a new method, a
reworked formula - this table needs re-deriving to match before
[the Rust core](rust-array-core.md)'s own scope is adjusted.

## dtype and shape

`float64` only, matching Python's own `float` (the same reasoning
[vectorized array-based model classes](vectorized-array-classes.md)'s own numerical-parity
section already requires). Up to 2D: a 1D vector (single-example activations/deltas/biases) or
a 2D matrix (a layer's weights; a batch's stacked examples/activations). No 3D+ arrays anywhere
in the class design - this codebase never batches over more than one axis at once.

## required operations

| operation | numpy call used | semantics required | used by (vectorized-array-classes.md reference) |
|---|---|---|---|
| construct from data | `np.array(data, dtype=np.float64)` | build a 1D array from a Python sequence (a state tuple) | `_forward`'s own state-to-array conversion |
| construct zero-filled | `np.zeros(shape)` | build a 1D or 2D array of zeros at a given shape | `ArrayLayer.__init__`'s `W`/`b`; gradient accumulators; one-hot target construction |
| shape | `.shape` | read an array's dimensions | `learn_batch`'s `X.shape == (batch_size, input_size)` |
| transpose | `.T` | swap a 2D array's two axes | `X @ self.W.T`; `next_layer.W.T @ next_layer.delta`; `delta_batch.T @ X_batch` |
| single-element write | `arr[i] = value` (1D) and `arr[i, j] = value` (2D) | write one element by integer index - either a scalar index into a 1D array or a `(row, col)` tuple index into a 2D array | one-hot target construction, both the single-example (`target[category] = 1.0`) and batched (`target_batch[row, category] = 1.0`, `learn_batch`) cases - two different index shapes, not the same call re-derived |
| matrix multiplication | `@` | matrix-vector product (1D x 2D or 2D x 1D) and matrix-matrix product (2D x 2D), no broadcasting beyond standard matmul rules | every `forward`/`forward_batch`/`compute_hidden_delta` formula in `ArrayLayer` |
| elementwise `+` | `+` | elementwise add, with two scoped broadcasting cases: vector + vector (same shape) and matrix + row-vector (bias add across every row of a batch) | `W @ x + b`; `X @ W.T + b` |
| elementwise `- * /` | `- * /` | elementwise arithmetic, same-shape operands or a scalar operand (learning-rate scaling) | every delta/gradient formula (`(a - reference) * a * (1 - a)`; `W -= lr * grad_W / batch_size`) |
| in-place accumulate | `+=` | elementwise add into an existing array, in place | gradient accumulators (`self._grad_W += ...`) |
| elementwise exp | `np.exp` | elementwise `e^x` over a whole array | `sigmoid`'s own array-wide formula |
| outer product | `np.outer(a, b)` | the full pairwise-product matrix of two 1D vectors | `accumulate_gradient`'s single-example gradient (`np.outer(delta, input_layer.a)`) |
| row-sum reduction | `.sum(axis=0)` | sum a 2D array's rows into a 1D vector - one fixed axis, not a general axis-parameterized reduction | `accumulate_gradient_batch`'s batched bias gradient (`self._grad_b += self.delta_batch.sum(axis=0)`) |
| argmax | `np.argmax(a)` | the index of the largest element in a 1D array | `classify_state` |
| uniform random fill | `np.random.uniform(low, high, size=shape)` | fill an array of the given shape with independent uniform draws | `randomize()`'s fan-in-aware initialization |
| copy | `.copy()` | an independent copy of an array (mutating the copy must not affect the original) | `snapshot()` |
| raw byte decode | `np.frombuffer(data, dtype=np.uint8)` | interpret a raw byte buffer as an array of unsigned 8-bit integers, no copy | `load_mnist_dataset_as_array` |
| reshape | `.reshape(shape)` | reinterpret an array's shape without changing its data or order | `load_mnist_dataset_as_array`'s per-record layout |
| slicing | `arr[:, :-1]` | a contiguous sub-array view along one axis | `load_mnist_dataset_as_array`'s pixel-vs-label split |
| dtype cast | `.astype(np.float64)` | convert an array's element type, preserving values | `load_mnist_dataset_as_array`'s `uint8 -> float64` conversion |
| Python round-trip | `.tolist()` / `np.array(nested_list)` | convert to/from plain nested Python lists, for JSON serialization | `save()`/`load()` |

## explicitly not required

- **General N-dimensional arrays.** Every operation above is 1D or 2D only.
- **General broadcasting.** Only the two specific cases named above (vector+vector,
  matrix+row-vector) - not scalar-to-N-d, not arbitrary shape alignment rules.
- **Fancy or boolean indexing.** Only single-element index writes (`arr[i] = value` or
  `arr[i, j] = value`) and contiguous slices (`arr[:, :-1]`) - no boolean masks, no
  integer-array indexing.
- **General axis-parameterized reductions** (`sum`/`max`/`argmax` with an arbitrary `axis`
  argument), `np.where`, `np.maximum`/`np.minimum`, `np.clip` - with one specific, already-required
  exception: `.sum(axis=0)` on a 2D array (see the required-operations table above), needed by
  `accumulate_gradient_batch`'s batched bias gradient. Everything else in this bullet belongs to
  variants [vectorized array-based model classes](vectorized-array-classes.md) explicitly defers
  (ReLU, softmax, momentum, L2) - real operations `vectorization.md`'s own original inventory
  already identified, just not required by *this* subset's base-case scope. A follow-on covering
  one of those variants would add exactly the operations it needs here, not before.
- **Any linear algebra beyond matmul and outer product** - no inverse, no decomposition, no
  eigenvalues; this codebase's own math never needs them.

## how this feeds the Rust implementation

Every row above becomes exactly one thing to build and parity-check in
[the Rust implementation plan](rust-array-core.md) - no more (nothing here is speculative), no
less (nothing [vectorized array-based model classes](vectorized-array-classes.md) needs is
missing). If that document's own class design changes - a new method, a reworked formula - this
table is what needs re-deriving first, before the Rust workplan's own scope is adjusted to
match; this document is downstream of that one, not independent of it.
