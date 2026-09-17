from __future__ import annotations

from typing import Sequence

import indrajala_ml_array as pa

from indrajala_ml.model.bounds import validate_batch, validate_class_count, validate_layer_sizes
from indrajala_ml.model.model_io import load_array_model_json, save_array_model_json
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer


class SoftmaxRustArrayMultiClassBackpropClassifierNetwork:
    """
    The Rust-matmul-backed counterpart to SoftmaxVectorizedMultiClassBackpropClassifierNetwork -
    see docs/softmax-array-layer.md's Rust-matmul-backed follow-on. Mirrors that class's external
    contract exactly against SoftmaxRustArrayLayer instead of SoftmaxArrayLayer - the same
    relationship RustArrayMultiClassBackpropClassifierNetwork has to
    VectorizedMultiClassBackpropClassifierNetwork, applied one level further to the softmax line.
    A wholly separate class, not a subclass swapping a layer_cls extension point.

    Hidden layers are built from plain RustArrayLayer (sigmoid); only the output layer is a
    SoftmaxRustArrayLayer - the same split SoftmaxVectorizedMultiClassBackpropClassifierNetwork
    uses. No extra constructor parameter, matching that class's own posture - softmax has no
    tunable coefficient.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int) -> None:

        validate_layer_sizes(layer_sizes)
        validate_class_count(class_count)

        self.layer_sizes = layer_sizes
        self.dimension = dimension
        self.class_count = class_count

        self.layers: list[RustArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(RustArrayLayer(size, previous_size))
            previous_size = size

        self.output_layer = SoftmaxRustArrayLayer(class_count, previous_size)
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
        validate_batch(batch)
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
        # the same fan-in-aware scheme every sibling here uses - limit = 1/sqrt(fan_in) - drawn
        # from indrajala_ml_array.uniform, not np.random.uniform; see rust-array-core.md's "the
        # RNG exception" for why this can never be seed-reproducible against the numpy-backed
        # sibling's own draws.
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
    ) -> "SoftmaxRustArrayMultiClassBackpropClassifierNetwork":
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
        save_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            class_count=self.class_count,
            snapshot=self.snapshot(),
        )

    @classmethod
    def load(cls, path: str) -> "SoftmaxRustArrayMultiClassBackpropClassifierNetwork":
        state = load_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"], state["class_count"])
        network.restore([(pa.Array(W), pa.Array(b)) for W, b in state["snapshot"]])
        return network
