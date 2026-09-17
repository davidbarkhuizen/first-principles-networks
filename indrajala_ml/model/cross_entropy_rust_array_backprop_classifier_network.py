from __future__ import annotations

from typing import Sequence

import indrajala_ml_array as pa

from indrajala_ml.model.bounds import validate_batch, validate_layer_sizes
from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.model_io import load_single_output_array_model_json, save_single_output_array_model_json
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class CrossEntropyRustArrayBackpropClassifierNetwork:
    """
    The Rust-array-core-backed sibling of CrossEntropyArrayBackpropClassifierNetwork - see
    docs/architecture/rust-production-cutover.md's phase 1 and
    docs/proposals/binary-cross-entropy-array-layer.md. Built unconditionally alongside every
    other Rust-backed sibling in this codebase, not gated behind a numpy-stage wall-clock verdict -
    per docs/architecture/rust-production-cutover.md's 2026-09-16 clarification. Mirrors
    CrossEntropyArrayBackpropClassifierNetwork's external contract exactly (learn, learn_batch,
    predict_probability, classify_state, randomize/randomized, snapshot/restore, save/load) so
    it's a drop-in swap - only the array backend (indrajala_ml_array.Array via
    CrossEntropyRustArrayLayer, not numpy via CrossEntropyArrayLayer) differs.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        # input_bounds is accepted and discarded - see ArrayBackpropClassifierNetwork's own
        # docstring for why (duck-type compatibility with ensemble_train.py's classifier_cls
        # contract).
        validate_layer_sizes(layer_sizes)
        self.layer_sizes = layer_sizes
        self.dimension = dimension

        self.layers: list[RustArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(RustArrayLayer(size, previous_size))
            previous_size = size

        self.output_layer = CrossEntropyRustArrayLayer(1, previous_size)
        self.layers.append(self.output_layer)

    def _forward(self, state: tuple[float, ...]) -> "pa.Array":
        x = pa.Array(list(state))
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return self._forward(state).tolist()[0]

    def classify_state(self, state: tuple[float, ...]) -> float:
        return 1.0 if self.predict_probability(state) > 0.5 else 0.0

    def learn(self, learning_rate: float, state: tuple[float, ...], category: float) -> None:
        activations = [pa.Array(list(state))]
        x = activations[0]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)

        target = pa.Array([category])
        self.output_layer.compute_output_delta(target)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta(self.layers[i + 1])

        for layer, input_activation in zip(self.layers, activations):
            layer.accumulate_gradient(input_activation)
            layer.apply_accumulated_gradient(learning_rate, batch_size=1)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], float]]) -> None:
        validate_batch(batch)
        batch_size = len(batch)

        activations = [pa.Array([list(state) for state, _category in batch])]
        X = activations[0]
        for layer in self.layers:
            X = layer.forward_batch(X)
            activations.append(X)

        target_batch = pa.Array([[category] for _state, category in batch])
        self.output_layer.compute_output_delta_batch(target_batch)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])

        for layer, input_activation_batch in zip(self.layers, activations):
            layer.accumulate_gradient_batch(input_activation_batch)
            layer.apply_accumulated_gradient(learning_rate, batch_size)

    def randomize(self) -> None:
        # the same fan-in-aware scheme CrossEntropyArrayBackpropClassifierNetwork.randomize uses
        # - limit = 1/sqrt(fan_in) - but drawn from indrajala_ml_array.uniform, not
        # np.random.uniform; see rust-array-core.md's "the RNG exception" for why this can never
        # be seed-reproducible against the numpy-backed sibling's own draws.
        previous_size = self.dimension
        for layer in self.layers:
            limit = 1.0 / (previous_size ** 0.5)
            layer.W = pa.uniform(-limit, limit, (layer.size, previous_size))
            layer.b = pa.uniform(-limit, limit, layer.size)
            previous_size = layer.size

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> "CrossEntropyRustArrayBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple["pa.Array", "pa.Array"]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple["pa.Array", "pa.Array"]]) -> None:
        # tolerates plain nested lists as well as pa.Array (wrapping via pa.Array(...) when
        # needed), the same pattern load() already uses - lets a snapshot cross a
        # multiprocessing.Pool worker boundary as plain, picklable lists (see
        # ensemble_train._picklable_snapshot) and land here without a separate reconstruction
        # step at every call site.
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = W.copy() if isinstance(W, pa.Array) else pa.Array(W)
            layer.b = b.copy() if isinstance(b, pa.Array) else pa.Array(b)

    def save(self, path: str) -> None:
        save_single_output_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            snapshot=self.snapshot(),
        )

    @classmethod
    def load(cls, path: str) -> "CrossEntropyRustArrayBackpropClassifierNetwork":
        state = load_single_output_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"])
        network.restore([(pa.Array(W), pa.Array(b)) for W, b in state["snapshot"]])
        return network
