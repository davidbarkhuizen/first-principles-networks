from __future__ import annotations

from typing import Sequence

import numpy as np

from indrajala_ml.model.bounds import validate_batch, validate_class_count, validate_layer_sizes
from indrajala_ml.model.l2_array_layer import L2ArrayLayer
from indrajala_ml.model.model_io import load_array_model_json, save_array_model_json


class L2VectorizedMultiClassBackpropClassifierNetwork:
    """
    The L2 (weight decay) regularized sibling of VectorizedMultiClassBackpropClassifierNetwork -
    see docs/l2-array-layer.md. Mirrors AdamVectorizedMultiClassBackpropClassifierNetwork's own
    precedent for adding an array-based sibling: a wholly separate class duplicating the same
    external contract (learn/learn_batch/classify_state/predict_probabilities/
    randomize/randomized/snapshot/restore/save/load) against L2ArrayLayer instead of ArrayLayer,
    not a subclass swapping a layer_cls extension point - VectorizedMultiClassBackpropClassifierNetwork.__init__
    hardcodes ArrayLayer construction and has no such extension point to hook.

    l2_lambda is a required constructor parameter, no default - the same posture
    L2RegularizedBackpropClassifierNetwork's per-node counterpart already takes, and both
    hidden layers and the output layer are built from L2ArrayLayer, mirroring
    L2RegularizedBackpropClassifierNetwork's own hidden_layer_cls == output_layer_cls choice.

    snapshot()/restore() intentionally cover only W/b - L2 needs no extra per-parameter state to
    capture in the first place (see L2ArrayLayer's own docstring), so this is not a new gap the
    way it is for momentum/Adam's own array siblings.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        l2_lambda: float,
    ) -> None:

        validate_layer_sizes(layer_sizes)
        validate_class_count(class_count)

        self.layer_sizes = layer_sizes
        self.dimension = dimension
        self.class_count = class_count
        self.l2_lambda = l2_lambda

        self.layers: list[L2ArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(L2ArrayLayer(size, previous_size, l2_lambda))
            previous_size = size

        self.output_layer = L2ArrayLayer(class_count, previous_size, l2_lambda)
        self.layers.append(self.output_layer)

    def _forward(self, state: tuple[float, ...]) -> np.ndarray:
        x = np.array(state, dtype=np.float64)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state).tolist()

    def classify_state(self, state: tuple[float, ...]) -> int:
        return int(np.argmax(self._forward(state)))

    def learn(self, learning_rate: float, state: tuple[float, ...], category: int) -> None:
        activations = [np.array(state, dtype=np.float64)]
        x = activations[0]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)

        target = np.zeros(self.class_count)
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

        activations = [np.array([state for state, _category in batch], dtype=np.float64)]
        X = activations[0]
        for layer in self.layers:
            X = layer.forward_batch(X)
            activations.append(X)

        target_batch = np.zeros((batch_size, self.class_count))
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
        # uses - limit = 1/sqrt(fan_in), one array draw per layer instead of a per-node loop
        previous_size = self.dimension
        for layer in self.layers:
            limit = 1.0 / np.sqrt(previous_size)
            layer.W = np.random.uniform(-limit, limit, size=(layer.size, previous_size))
            layer.b = np.random.uniform(-limit, limit, size=(layer.size,))
            previous_size = layer.size

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        l2_lambda: float,
    ) -> "L2VectorizedMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count, l2_lambda)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = np.array(W, dtype=np.float64).copy()
            layer.b = np.array(b, dtype=np.float64).copy()

    def save(self, path: str) -> None:
        save_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            class_count=self.class_count,
            snapshot=self.snapshot(),
            extra={"l2_lambda": self.l2_lambda},
        )

    @classmethod
    def load(cls, path: str) -> "L2VectorizedMultiClassBackpropClassifierNetwork":
        state = load_array_model_json(path)
        network = cls(
            state["layer_sizes"],
            state["dimension"],
            state["class_count"],
            state["l2_lambda"],
        )
        network.restore([(np.array(W), np.array(b)) for W, b in state["snapshot"]])
        return network
