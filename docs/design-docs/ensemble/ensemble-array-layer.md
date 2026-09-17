# an array-based (Rust-matmul-backed) ensemble sibling

[← back to README](../../../README.md)

Design reference for `ArrayBackpropClassifierNetwork`/`RustArrayBackpropClassifierNetwork` and
`EnsembleArrayBackpropClassifierNetwork`/`EnsembleRustArrayBackpropClassifierNetwork` (see
[structure](../../project/structure.md#possible-next-steps)). Built and measured, both backends,
unconditionally; see [research and
analysis](../../research/research-multiclass-and-loss.md#step-4-an-arrayrust-backed-ensemble-and-the-parallel-vs-serial-question)
for the full measurement and decision.

## design

### piece 1: `ArrayBackpropClassifierNetwork`/`RustArrayBackpropClassifierNetwork` - fan-in-aware only, by design, not by omission

A new standalone single-output array-based network line (every prior array-ported sibling in
this codebase's history had only ever needed the existing `class_count`-wide multiclass line),
structurally almost identical to `VectorizedMultiClassBackpropClassifierNetwork`/
`RustArrayMultiClassBackpropClassifierNetwork` but with a single-node output layer and no
`class_count` at all:

```python
class ArrayBackpropClassifierNetwork:
    def __init__(self, layer_sizes, dimension, input_bounds=None) -> None:
        # input_bounds accepted and discarded - see "the training-path integration gap" below
        ...
        self.output_layer = ArrayLayer(1, previous_size)
        self.layers.append(self.output_layer)

    def predict_probability(self, state) -> float:
        return float(self._forward(state)[0])

    def classify_state(self, state) -> float:
        return 1.0 if self.predict_probability(state) > 0.5 else 0.0
    # learn/learn_batch/randomize/randomized/snapshot/restore/save/load mirror
    # VectorizedMultiClassBackpropClassifierNetwork's own shape exactly, minus class_count
```

`randomize()` implements **only** the fan-in-aware scheme (`limit = 1/sqrt(fan_in)`) - the same
choice `VectorizedMultiClassBackpropClassifierNetwork.randomize()` already made, skipping
`BackpropClassifierNetwork.randomize()`'s own per-dimension-bounds-width scaling entirely: it's
"tuned for 1-2D geometric problems"
([FanInAwareBackpropClassifierNetwork](../../project/structure.md#backprop-siblings)'s own
docstring), while this class exists specifically for the ensemble's 784-dimension MNIST use case,
where fan-in-aware init is the only scheme ever measured to work - measured directly to take the
real ensemble from 89.4% to 96.01% test accuracy ([the ensemble/real-MNIST
investigation](../../research/research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)).

The shared `limit = 1/sqrt(previous_size)` fan-in-aware draw is extracted into one function
(`array_layer.fan_in_aware_random_layer`) both `VectorizedMultiClassBackpropClassifierNetwork.randomize()`
and this class's `randomize()` call, avoiding two independent copies of the same formula.

`save()`/`load()` use their own JSON envelope (`save_single_output_array_model_json`/
`load_single_output_array_model_json`, `model_io.py`) - no `class_count`, no `input_bounds`, the
same "this class has no notion of `input_bounds`, no `StateLayer`" reasoning
`VectorizedMultiClassBackpropClassifierNetwork.save()`'s own comment already gives, not
`save_array_model_json` (which hardcodes `class_count`).

The Rust counterpart (`RustArrayBackpropClassifierNetwork`) is a direct structural mirror,
substituting `RustArrayLayer`/`indrajala_ml_array.Array` throughout - no new fused Rust op was
needed, it reuses `RustArrayLayer`'s existing fused ops, just assembled with a 1-node output
instead of `class_count`-node.

### piece 2: `EnsembleArrayBackpropClassifierNetwork`/`EnsembleRustArrayBackpropClassifierNetwork`

A direct structural mirror of `EnsembleBackpropClassifierNetwork` itself, substituting
`ArrayBackpropClassifierNetwork`/`RustArrayBackpropClassifierNetwork` for
`BackpropClassifierNetwork` throughout. Unlike piece 1 (no `class_count` notion at all), this
wrapper does have a real `class_count` (the number of assembled sub-networks), but its own
`save()`/`load()` still can't reuse `save_array_model_json`: that helper's snapshot handling
assumes one network's own flat per-layer `(W, b)` list, not this class's per-classifier nested
snapshot shape - so it uses the bare `save_json`/`load_json` primitives instead (`model_io.py`),
the same "envelope shape doesn't fit the fixed layout" case that module's own docstring names
`ConvMultiClassBackpropClassifierNetwork.save` as an example of.

### the training-path integration gap, and how it was closed

`ensemble_train.py`'s parallel training functions (`train_ensemble_parallel`,
`train_ensemble_parallel_from_indices`) always call
`classifier_cls.randomized(layer_sizes, dimension, input_bounds)` - three arguments, the last of
which the new single-output array classes have no real use for (no `StateLayer`/`input_bounds`
notion). Closed by giving both classes' `__init__`/`randomized()` an accepted-and-discarded
`input_bounds=None` trailing parameter, rather than changing `ensemble_train.py`'s own shared
contract.

A second, separate gap: `indrajala_ml_array.Array` does not support pickling at all (confirmed
directly - `pickle.dumps` raises `TypeError`), which would otherwise block a Rust-backed
classifier_cls from ever returning a trained snapshot across a `multiprocessing.Pool` worker
boundary. Closed by `ensemble_train._picklable_snapshot` (converts a worker's returned snapshot
to plain, always-picklable nested lists before it crosses the process boundary - a no-op for
per-node/numpy snapshots, which are already picklable) paired with
`RustArrayBackpropClassifierNetwork.restore()`'s own tolerance for receiving either plain lists
or `indrajala_ml_array.Array`, the same pattern its own `load()` already used. Once both were in
place, the *existing* `train_ensemble_parallel`/`train_ensemble_parallel_from_indices` trained a
Rust-backed ensemble unchanged - no new public training function was needed for the parallel
case. `train_ensemble_serial_from_indices` (a new, single-process training path with no
`multiprocessing.Pool` at all) was still built, as the direct way to answer the "is
multiprocessing still worth keeping" question - see the research doc for that measurement.

## correctness validation

Same two-tier convention as every array-based sibling: `test_array_backprop_model.py`/
`test_rust_array_backprop_model.py` parity-check the single-output network against the per-node
`FanInAwareBackpropClassifierNetwork` reference (`learn`/`learn_batch`/`predict_probability`/
`classify_state`/`snapshot`/`restore`, matching after every step, not just at the end);
`test_ensemble_array_backprop_model.py`/`test_ensemble_rust_array_backprop_model.py` exercise the
ensemble wrapper end to end, mirroring `test_ensemble_backprop_classifier_network.py`'s own test
shape; `test_ensemble_train.py` proves both backends train correctly through both the parallel
and serial paths.
