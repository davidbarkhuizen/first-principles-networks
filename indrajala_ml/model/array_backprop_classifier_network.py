from __future__ import annotations

from typing import Sequence

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer, fan_in_aware_random_layer
from indrajala_ml.model.bounds import validate_batch, validate_layer_sizes
from indrajala_ml.model.model_io import load_single_output_array_model_json, save_single_output_array_model_json


class ArrayBackpropClassifierNetwork:
    """
    The single-output numpy-array-backed sibling of FanInAwareBackpropClassifierNetwork - see
    docs/proposals/ensemble-array-layer.md's "piece 1". Structurally almost identical to
    VectorizedMultiClassBackpropClassifierNetwork, but a single-node output layer and no
    class_count at all: this class exists specifically to host
    EnsembleArrayBackpropClassifierNetwork's sub-networks, one independent binary classifier per
    class, not a jointly-trained multiclass network.

    randomize() implements only the fan-in-aware scheme (limit = 1/sqrt(fan_in)) - the same
    choice VectorizedMultiClassBackpropClassifierNetwork.randomize() already made, skipping
    BackpropClassifierNetwork.randomize()'s own per-dimension-bounds-width scaling entirely: that
    scheme is "tuned for 1-2D geometric problems"
    (FanInAwareBackpropClassifierNetwork's own docstring), while this class exists for the
    ensemble's high-dimensional (784-pixel real-MNIST) use case, where fan-in-aware init is the
    only scheme ever measured to work.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        # input_bounds is accepted and discarded - this class has no StateLayer/input_bounds
        # notion, but ensemble_train.py's classifier_cls contract always calls
        # classifier_cls(layer_sizes, dimension, input_bounds) /
        # classifier_cls.randomized(layer_sizes, dimension, input_bounds); accepting it here is
        # the duck-typing relaxation docs/proposals/ensemble-array-layer.md's own "training-path
        # integration gap" section named as the fix, rather than changing that shared contract.
        validate_layer_sizes(layer_sizes)
        self.layer_sizes = layer_sizes
        self.dimension = dimension

        self.layers: list[ArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(ArrayLayer(size, previous_size))
            previous_size = size

        self.output_layer = ArrayLayer(1, previous_size)
        self.layers.append(self.output_layer)

    def _forward(self, state: tuple[float, ...]) -> np.ndarray:
        x = np.array(state, dtype=np.float64)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return float(self._forward(state)[0])

    def classify_state(self, state: tuple[float, ...]) -> float:
        return 1.0 if self.predict_probability(state) > 0.5 else 0.0

    def learn(self, learning_rate: float, state: tuple[float, ...], category: float) -> None:
        activations = [np.array(state, dtype=np.float64)]
        x = activations[0]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)

        target = np.array([category], dtype=np.float64)
        self.output_layer.compute_output_delta(target)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta(self.layers[i + 1])

        for layer, input_activation in zip(self.layers, activations):
            layer.accumulate_gradient(input_activation)
            layer.apply_accumulated_gradient(learning_rate, batch_size=1)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], float]]) -> None:
        validate_batch(batch)
        batch_size = len(batch)

        activations = [np.array([state for state, _category in batch], dtype=np.float64)]
        X = activations[0]
        for layer in self.layers:
            X = layer.forward_batch(X)
            activations.append(X)

        target_batch = np.array([[category] for _state, category in batch], dtype=np.float64)
        self.output_layer.compute_output_delta_batch(target_batch)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])

        for layer, input_activation_batch in zip(self.layers, activations):
            layer.accumulate_gradient_batch(input_activation_batch)
            layer.apply_accumulated_gradient(learning_rate, batch_size)

    def randomize(self) -> None:
        previous_size = self.dimension
        for layer in self.layers:
            layer.W, layer.b = fan_in_aware_random_layer(layer.size, previous_size)
            previous_size = layer.size

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> "ArrayBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = np.array(W, dtype=np.float64).copy()
            layer.b = np.array(b, dtype=np.float64).copy()

    def save(self, path: str) -> None:
        # save_single_output_array_model_json (model_io.py), not save_array_model_json - this
        # class has no class_count notion at all, unlike every multiclass array-backed sibling.
        save_single_output_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            snapshot=self.snapshot(),
        )

    @classmethod
    def load(cls, path: str) -> "ArrayBackpropClassifierNetwork":
        state = load_single_output_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"])
        network.restore([(np.array(W), np.array(b)) for W, b in state["snapshot"]])
        return network
