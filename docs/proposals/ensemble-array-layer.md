# an array-based (Rust-matmul-backed) ensemble sibling

[← back to README](../../README.md)

**Status: proposed, not started.** Written up front as a design/measurement plan before any of
it exists, per this repo's own established practice of writing a plan down before implementation,
matching every other array-porting workplan in this codebase's history.

## why this, and why now

`EnsembleBackpropClassifierNetwork` (`ensemble_backprop_classifier_network.py`) has **no
array-based or Rust-matmul-backed counterpart at all** - checked directly, not assumed: neither
`indrajala_ml/model/` nor any workplan doc names one. This matters more than any other gap this
audit found, because the ensemble isn't a toy sibling awaiting its first real measurement - it's
the **best-performing, production-facing capability this codebase has**: 10 independent
`FanInAwareBackpropClassifierNetwork`s, one per digit, is what `demo_mnist_ensemble_recognition.py`
and `demo_mnist_ensemble_capture.py` actually run, and it holds the best documented real-MNIST
result of any architecture in this codebase - 96.01% held-out test accuracy
([the ensemble/real-MNIST
investigation](../research/research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)), ahead of
both one-vs-rest (92.75%) and softmax (89.12%) on the same real-MNIST comparison
([softmax on real full-scale MNIST](../research/research-multiclass-and-loss.md#softmax-on-real-full-scale-mnist)).
Today that flagship result is reached entirely on the per-node path, at ~29.6 minutes wall-clock,
sped up only by `multiprocessing` (10 jobs, 8 parallel workers) - vectorization has never touched
it. Every other array/Rust port in this codebase's history has been a weight-update-rule or
activation variant layered onto an already-array-based network; this is the first case where the
*already-adopted, already-production-facing* capability itself has never been vectorized at all.

## scope: two real, separately-buildable pieces

**Piece 1 - a single-output array-based network, which doesn't exist yet at all.** Every prior
array-ported sibling in this codebase's history (momentum, L2, ReLU, dropout - see [research and
analysis](../research/research-backprop-siblings.md)) had to scope itself onto
`MultiClassBackpropClassifierNetwork`'s existing array line because no single-output
(`BackpropClassifierNetwork`-equivalent) array-based line exists - a gap the array-based Adam
sibling first flagged and left unresolved. The ensemble is exactly the case that gap blocks
directly: each ensemble
sub-network *is* a `BackpropClassifierNetwork` (via its `FanInAwareBackpropClassifierNetwork`
sibling), not a multiclass network. This workplan is where building that single-output array
network finally becomes necessary, not optional - see "design" below.

**Piece 2 - a thin ensemble wrapper around `class_count` of them**, mirroring
`EnsembleBackpropClassifierNetwork`'s own "assemble already-constructed classifiers, don't build
them" composition, not `BackpropNetworkBase.__init__`'s "build from layer_sizes" shape.

Also mirrored, once piece 1 exists as `ArrayBackpropClassifierNetwork`/
`RustArrayBackpropClassifierNetwork`: [binary cross-entropy's own array
port](binary-cross-entropy-array-layer.md), which depends on exactly this piece rather than
duplicating it.

## design

### piece 1: `ArrayBackpropClassifierNetwork` - fan-in-aware only, by design, not by omission

A new standalone class, structurally almost identical to
`VectorizedMultiClassBackpropClassifierNetwork` but with a single-node output layer and no
`class_count` at all:

```python
class ArrayBackpropClassifierNetwork:
    def __init__(self, layer_sizes: list[int], dimension: int) -> None:
        validate_layer_sizes(layer_sizes)
        self.dimension = dimension
        self.layer_sizes = layer_sizes

        self.layers: list[ArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(ArrayLayer(size, previous_size))
            previous_size = size
        self.output_layer = ArrayLayer(1, previous_size)
        self.layers.append(self.output_layer)

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return float(self._forward(state)[0])

    def classify_state(self, state: tuple[float, ...]) -> float:
        return 1.0 if self.predict_probability(state) > 0.5 else 0.0
    # learn/learn_batch/randomize/randomized/snapshot/restore/save/load mirror
    # VectorizedMultiClassBackpropClassifierNetwork's own shape exactly, minus class_count
```

`randomize()` implements **only** the fan-in-aware scheme (`limit = 1/sqrt(fan_in)`) - the same
choice `VectorizedMultiClassBackpropClassifierNetwork.randomize()` already made, skipping
`BackpropClassifierNetwork.randomize()`'s own per-dimension-bounds-width scaling entirely, not as
an oversight but for the same reason that scheme was never ported to the multiclass array line:
it's "tuned for 1-2D geometric problems"
([FanInAwareBackpropClassifierNetwork](../project/structure.md#backprop-siblings)'s own docstring) and this
class exists specifically for the ensemble's 784-dimension MNIST use case, where fan-in-aware
init is the only scheme ever measured to work - measured directly to take the real ensemble from
89.4% to 96.01% test accuracy
([the ensemble/real-MNIST
investigation](../research/research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)). This also
means `FanInAwareBackpropClassifierNetwork` itself gets no separate array-layer workplan of its
own in this round - its only real consumer, at the only fan-in scale that matters, is exactly
this class, so a separate document would just duplicate this section.

Extracting the shared `limit = 1/sqrt(previous_size)` fan-in-aware draw into one function both
`VectorizedMultiClassBackpropClassifierNetwork.randomize()` and this class's `randomize()` call
(mirroring `randomize_fan_in_aware`'s own extraction, once `FanInAwareBackpropClassifierNetwork`
needed the identical per-node scheme `MultiClassBackpropClassifierNetwork` already had) avoids
two independent copies of the same formula at the array level too.

`save()`/`load()` need their own JSON envelope (no `class_count`, no `input_bounds` - the same
"this class has no notion of `input_bounds`, no `StateLayer`" reasoning
`VectorizedMultiClassBackpropClassifierNetwork.save()`'s own comment already gives), not
`save_model_json` (which hardcodes both).

### piece 2: `EnsembleArrayBackpropClassifierNetwork`

```python
class EnsembleArrayBackpropClassifierNetwork:
    def __init__(self, classifiers: list[ArrayBackpropClassifierNetwork]) -> None:
        assert len(classifiers) >= 2
        self.classifiers = classifiers
        self.class_count = len(classifiers)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return [c.predict_probability(state) for c in self.classifiers]

    def classify_state(self, state: tuple[float, ...]) -> int:
        return int(np.argmax(self.predict_probabilities(state)))
    # snapshot/restore/save/load mirror EnsembleBackpropClassifierNetwork's own shape exactly
```

A direct structural mirror of `EnsembleBackpropClassifierNetwork` itself, substituting
`ArrayBackpropClassifierNetwork` for `BackpropClassifierNetwork` throughout - no new design
question here beyond piece 1's.

### the training-path integration gap, found by checking, not assumed to just work

`ensemble_train.py`'s parallel training functions (`train_ensemble_parallel`,
`train_ensemble_parallel_from_indices`) take a `classifier_cls: type[BackpropClassifierNetwork]`
parameter specifically so `demo_mnist_ensemble_recognition.py` can opt into
`FanInAwareBackpropClassifierNetwork` without touching any other caller - checked directly in
`ensemble_train.py`: `classifier_cls.randomized(layer_sizes, dimension, input_bounds)` is the
exact call each worker makes. `ArrayBackpropClassifierNetwork.randomized`, per its own
constructor above, takes `(layer_sizes, dimension)` - **no `input_bounds` parameter at all**,
since array classes have no `StateLayer`/`input_bounds` notion. Dropping
`ArrayBackpropClassifierNetwork` in as `classifier_cls` today would therefore raise a
`TypeError` on the very first call, not silently misbehave - a real, specific, previously
unstated integration gap, not a hypothetical one. Closing it needs a small `ensemble_train.py`
change (an `input_bounds: list[tuple[float, float]] | None` parameter threaded through, only
passed to `classifier_cls.randomized(...)` when the signature expects it, or an equivalent
duck-typing relaxation) - out of scope for this doc (no code under `indrajala_ml/` here), but
flagged as a concrete, named prerequisite for full `ensemble_train.py` reuse, not glossed over as
"just works because it's duck-typed already."

A second, separate integration question: `ensemble_train.py`'s own docstrings require worker
functions to be "plain, module-level" for multiprocessing picklability, and every existing
worker ships plain Python objects (tuples, floats, per-node snapshots) across process
boundaries. Numpy arrays pickle natively, so `ArrayBackpropClassifierNetwork`'s own
snapshot/restore round-trip should carry across a `multiprocessing.Pool` worker boundary without
issue. `indrajala_ml_array.Array` (the PyO3-wrapped Rust type `RustArrayBackpropClassifierNetwork`
would use) has never been pickled across a process boundary anywhere in this codebase, and
nothing in [the Rust array core](../architecture/rust-array-core.md) or [the production cutover
plan](../architecture/rust-production-cutover.md) states whether it supports `pickle` at all - a real, untested
risk for the Rust-backed ensemble specifically, flagged under "risks" below, not assumed to work
because the numpy backend does.

## correctness validation

Same two-tier convention as every array-based sibling:

- `test_array_backprop_model.py`: `ArrayBackpropClassifierNetwork` parity-checked against
  `FanInAwareBackpropClassifierNetwork` (the genuine per-node reference, fan-in-aware `randomize()`
  and all) via a weight-injection helper mirroring `matching_array_backprop_networks`'s own
  pattern - `learn`/`learn_batch`/`predict_probability`/`classify_state`/`snapshot`/`restore`
  compared step by step, the same "matches after every step, not just at the end" discipline
  every prior array-layer workplan in this codebase used.
- `test_ensemble_array_backprop_model.py`: `EnsembleArrayBackpropClassifierNetwork` end to end
  (construction, `predict_probabilities`/`classify_state` argmax behavior, snapshot/restore,
  save/load), mirroring `test_ensemble_backprop_classifier_network.py`'s own test shape exactly,
  substituting the array classifier throughout.
- Once the Rust counterpart exists: the same two tiers against
  `RustArrayBackpropClassifierNetwork`/`EnsembleRustArrayBackpropClassifierNetwork`, plus
  whatever fused Rust ops the single-output network needs (likely none new - it reuses
  `RustArrayLayer`'s existing fused ops unchanged, just assembled with a 1-node output instead of
  `class_count`-node).

## measurement plan

- **Per-classifier wall-clock**: the standard fused-layer benchmark (`dimension=784, hidden=16`,
  numpy vs. Rust, several batch sizes) - per [goals and strategy](../project/goals-and-strategy.md#measurement-discipline-the-per-node-paths-two-jobs-and-the-one-it-doesnt-have),
  no fresh per-node timing run: this round's own five siblings already established per-node is
  30-1000x+ slower independently and consistently, so this benchmark compares the two backends
  actually worth choosing between, not re-confirms what per-node already lost every time.
- **The real question - does vectorizing change how this codebase should train the ensemble at
  all**: today's ~29.6-minute wall-clock comes from `multiprocessing` parallelism across 10
  independent per-node jobs, not from any per-classifier speed - a figure this codebase already
  has on record (the [ensemble/real-MNIST
  investigation](../research/research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)'s own step
  3), not one to re-time fresh. Once each classifier trains 70-800x faster per example (this
  round's own consistent finding), it's a real, open, honestly unresolved question whether
  **training all 10 classifiers serially in one process**, with no multiprocessing at all, beats
  or merely matches that documented parallel-per-node wall-clock - and if it does, that would
  remove real complexity from this codebase (the memory-aware worker-count capping, the
  index-based record loading fix, the whole `multiprocessing.Pool` machinery `ensemble_train.py`
  carries specifically for per-node-scale cost) rather than just making the existing approach
  faster. Measure the new serial array/Rust configuration directly and compare it against that
  existing documented figure, rather than assuming vectorization is strictly additive to the
  current design.
- **Accuracy parity at real-MNIST scale**: reproduce the 96.01% held-out test accuracy result on
  the array/Rust-backed ensemble, same architecture/hyperparameters, checking it lands within
  seed-to-seed noise of the per-node figure - the same [RNG-exception](../architecture/rust-array-core.md#the-rng-exception)
  caveat every array port in this codebase's history already carries (different RNG stream, not
  bit-identical, but qualitatively the same result expected).

## risks and open questions

- **The `ensemble_train.py` `classifier_cls` signature mismatch** (`input_bounds` required by
  the existing call, absent from the array network's own constructor) - a real, specific, found
  gap; see "design" above. Not resolved here (no code changes in this doc's scope), named as a
  concrete prerequisite.
- **Rust-object multiprocessing picklability is untested** - a real, unresolved risk specific to
  the Rust-backed ensemble path (the numpy path is expected to be fine); see "design" above. If
  it turns out `indrajala_ml_array.Array` doesn't pickle, the serial-training question above
  becomes moot for the Rust backend specifically (no parallelism needed if training is already
  fast enough serially) rather than a blocker - itself a reason to measure the serial-vs-parallel
  question first.
- **Whether multiprocessing is still worth keeping at all once vectorization lands** - flagged
  under "measurement plan" above as the single most consequential open question this workplan
  raises, deliberately not prejudged either way.
- **`save()`/`load()` envelope divergence** - `EnsembleBackpropClassifierNetwork.save()` uses
  `save_model_json` (hardcodes `input_bounds`); the array ensemble needs its own envelope, the
  same precedent `VectorizedMultiClassBackpropClassifierNetwork.save()`'s own comment already
  established for an identical problem. Not a blocker, just a stated design difference.

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. `ArrayBackpropClassifierNetwork` (piece 1) + its parity-check tests against
   `FanInAwareBackpropClassifierNetwork`.
3. `EnsembleArrayBackpropClassifierNetwork` (piece 2) + its own parity-check tests.
4. The per-classifier wall-clock measurement, and the serial-vs-parallel training-strategy
   measurement described above - the actual open question this workplan exists to answer.
5. Docs closeout: this document and `structure.md`'s possible-next-steps entry updated to reflect
   the actual result, including whatever the serial-vs-parallel finding turns out to be.
6. **Conditional** on stage 4's finding and on picklability being resolved:
   `RustArrayBackpropClassifierNetwork` + `EnsembleRustArrayBackpropClassifierNetwork` + the
   full real-MNIST accuracy-parity run, plus (if the `ensemble_train.py` integration gap above is
   closed in the same window) an actual `demo_mnist_ensemble_recognition.py`-equivalent run on
   the new backend, not just a standalone measurement script.
