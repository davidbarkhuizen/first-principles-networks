from __future__ import annotations

from typing import Sequence

import perceptron_array as pa

from perceptron.model.model_io import load_json, save_json
from perceptron.model.rust_array_layer import RustArrayLayer


class RustArrayMultiClassBackpropClassifierNetwork:
    """
    The Rust-array-core-backed sibling of VectorizedMultiClassBackpropClassifierNetwork - see
    docs/rust-production-cutover.md's phase 1. Mirrors that class's external contract exactly
    (learn, learn_batch, classify_state, predict_probabilities, randomize/randomized,
    snapshot/restore, save/load) so perceptron/train.py's duck-typed
    train_linear_classifier_network/train_backprop_network_mini_batch work unchanged - only the
    array backend (`perceptron_array.Array` via `RustArrayLayer`, not numpy via `ArrayLayer`)
    differs.

    Per docs/rust-production-cutover.md's 2026-09-16 clarification, this class is the intended
    production backend unconditionally - not contingent on beating
    VectorizedMultiClassBackpropClassifierNetwork's numpy benchmark, which stays on permanently
    as the comparison point, not a bar this class had to clear first.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int) -> None:

        assert len(layer_sizes) >= 1, "layer_sizes must specify at least one hidden layer"
        assert all(size >= 1 for size in layer_sizes), f"every hidden layer must have at least 1 node; got {layer_sizes}"
        assert class_count >= 2, f"class_count must be at least 2; got {class_count}"

        self.layer_sizes = layer_sizes
        self.dimension = dimension
        self.class_count = class_count

        self.layers: list[RustArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(RustArrayLayer(size, previous_size))
            previous_size = size

        self.output_layer = RustArrayLayer(class_count, previous_size)
        self.layers.append(self.output_layer)

    def _forward(self, state: tuple[float, ...]) -> "pa.Array":
        x = pa.Array(list(state))
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state).tolist()

    def classify_state(self, state: tuple[float, ...]) -> int:
        return pa.argmax(self._forward(state))

    def learn(self, learning_rate: float, state: tuple[float, ...], category: int) -> None:
        activations = [pa.Array(list(state))]
        x = activations[0]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)

        target = pa.Array.zeros(self.class_count)
        target[category] = 1.0
        self.output_layer.compute_output_delta(target)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta(self.layers[i + 1])

        for layer, input_activation in zip(self.layers, activations):
            layer.accumulate_gradient(input_activation)
            layer.apply_accumulated_gradient(learning_rate, batch_size=1)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], int]]) -> None:
        assert len(batch) >= 1, "batch must not be empty"
        batch_size = len(batch)

        activations = [pa.Array([list(state) for state, _category in batch])]
        X = activations[0]
        for layer in self.layers:
            X = layer.forward_batch(X)
            activations.append(X)

        target_batch = pa.Array.zeros((batch_size, self.class_count))
        for row, (_state, category) in enumerate(batch):
            target_batch[row, category] = 1.0
        self.output_layer.compute_output_delta_batch(target_batch)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])

        for layer, input_activation_batch in zip(self.layers, activations):
            layer.accumulate_gradient_batch(input_activation_batch)
            layer.apply_accumulated_gradient(learning_rate, batch_size)

    def randomize(self) -> None:
        # the same fan-in-aware scheme VectorizedMultiClassBackpropClassifierNetwork.randomize
        # uses - limit = 1/sqrt(fan_in) - but drawn from perceptron_array.uniform, not
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
        class_count: int,
    ) -> "RustArrayMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple["pa.Array", "pa.Array"]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple["pa.Array", "pa.Array"]]) -> None:
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = W.copy()
            layer.b = b.copy()

    def save(self, path: str) -> None:
        # own envelope, not save_model_json (model_io.py) - same reasoning
        # VectorizedMultiClassBackpropClassifierNetwork.save already established: no
        # input_bounds/StateLayer notion here.
        save_json(
            path,
            {
                "layer_sizes": self.layer_sizes,
                "dimension": self.dimension,
                "class_count": self.class_count,
                "snapshot": [(W.tolist(), b.tolist()) for W, b in self.snapshot()],
            },
        )

    @classmethod
    def load(cls, path: str) -> "RustArrayMultiClassBackpropClassifierNetwork":
        state = load_json(path)
        network = cls(state["layer_sizes"], state["dimension"], state["class_count"])
        network.restore([(pa.Array(W), pa.Array(b)) for W, b in state["snapshot"]])
        return network
