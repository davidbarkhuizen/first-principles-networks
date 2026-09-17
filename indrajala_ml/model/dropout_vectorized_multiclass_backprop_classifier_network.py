from __future__ import annotations

from typing import Sequence

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.bounds import validate_batch, validate_class_count, validate_layer_sizes
from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.model_io import load_array_model_json, save_array_model_json


class DropoutVectorizedMultiClassBackpropClassifierNetwork:
    """
    The dropout sibling of VectorizedMultiClassBackpropClassifierNetwork - see
    docs/dropout-array-layer.md. Mirrors AdamVectorizedMultiClassBackpropClassifierNetwork's own
    precedent for adding an array-based sibling: a wholly separate class duplicating the same
    external contract, not a subclass swapping a layer_cls extension point -
    VectorizedMultiClassBackpropClassifierNetwork.__init__ hardcodes ArrayLayer construction and
    has no such extension point to hook.

    Hidden layers are built from DropoutArrayLayer; the output layer stays a plain ArrayLayer
    (sigmoid) - the array-level analogue of DropoutBackpropClassifierNetwork's own
    hidden_layer_cls-only override, matching DropoutNode's hidden-layer-only convention.

    drop_probability is a required constructor parameter, no default - the same posture
    DropoutBackpropClassifierNetwork's per-node counterpart already takes.

    learn/learn_batch bracket only the forward pass in a set_training_mode(True)/try/finally,
    not the whole method - matching BackpropClassifierNetwork.learn's own bracket shape (see
    docs/dropout.md's "design"): each DropoutArrayLayer's backward pass reads its own
    forward-time _was_training/_mask snapshot, not the live flag, so a bracket any wider than
    "the forward pass itself" isn't needed, and predict_probabilities/classify_state need no
    bracket at all - training already defaults to False on every fresh DropoutArrayLayer and the
    finally clause guarantees it's never left stuck True.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        drop_probability: float,
    ) -> None:

        validate_layer_sizes(layer_sizes)
        validate_class_count(class_count)

        self.layer_sizes = layer_sizes
        self.dimension = dimension
        self.class_count = class_count
        self.drop_probability = drop_probability

        self.hidden_layers: list[DropoutArrayLayer] = []
        self.layers: list[ArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            layer = DropoutArrayLayer(size, previous_size, drop_probability)
            self.hidden_layers.append(layer)
            self.layers.append(layer)
            previous_size = size

        self.output_layer = ArrayLayer(class_count, previous_size)
        self.layers.append(self.output_layer)

    def _set_training_mode(self, training: bool) -> None:
        for layer in self.hidden_layers:
            layer.set_training_mode(training)

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
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)
        finally:
            self._set_training_mode(False)

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
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                X = layer.forward_batch(X)
                activations.append(X)
        finally:
            self._set_training_mode(False)

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
        drop_probability: float,
    ) -> "DropoutVectorizedMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count, drop_probability)
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
            extra={"drop_probability": self.drop_probability},
        )

    @classmethod
    def load(cls, path: str) -> "DropoutVectorizedMultiClassBackpropClassifierNetwork":
        state = load_array_model_json(path)
        network = cls(
            state["layer_sizes"],
            state["dimension"],
            state["class_count"],
            state["drop_probability"],
        )
        network.restore([(np.array(W), np.array(b)) for W, b in state["snapshot"]])
        return network
