from __future__ import annotations

from typing import Sequence

import numpy as np

from indrajala_ml.model.adam_array_layer import AdamArrayLayer
from indrajala_ml.model.adam_backprop_classifier_network import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON
from indrajala_ml.model.model_io import load_json, save_json


class AdamVectorizedMultiClassBackpropClassifierNetwork:
    """
    The Adam-optimized sibling of VectorizedMultiClassBackpropClassifierNetwork - see
    docs/adam-array-layer.md. Mirrors RustArrayMultiClassBackpropClassifierNetwork's own
    precedent for adding an array-based sibling: a wholly separate class duplicating the same
    external contract (learn/learn_batch/classify_state/predict_probabilities/
    randomize/randomized/snapshot/restore/save/load) against AdamArrayLayer instead of
    ArrayLayer, not a subclass swapping a layer_cls extension point -
    VectorizedMultiClassBackpropClassifierNetwork.__init__ hardcodes ArrayLayer construction and
    has no such extension point to hook.

    snapshot()/restore() intentionally cover only W/b, matching the base array-backed sibling's
    own contract unchanged - the same posture AdamBackpropClassifierNetwork's per-node counterpart
    already has (BackpropNetworkBase.snapshot/restore likewise only ever captured weights/bias,
    never a momentum/Adam node's own velocity/m/v/t), not a new gap this class introduces. A
    resumed-training scenario that needs m/v/t preserved across a snapshot/restore round trip
    would need its own extended envelope; no measurement in this codebase has used one so far.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:

        assert len(layer_sizes) >= 1, "layer_sizes must specify at least one hidden layer"
        assert all(size >= 1 for size in layer_sizes), f"every hidden layer must have at least 1 node; got {layer_sizes}"
        assert class_count >= 2, f"class_count must be at least 2; got {class_count}"

        self.layer_sizes = layer_sizes
        self.dimension = dimension
        self.class_count = class_count
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.layers: list[AdamArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(AdamArrayLayer(size, previous_size, beta1, beta2, epsilon))
            previous_size = size

        self.output_layer = AdamArrayLayer(class_count, previous_size, beta1, beta2, epsilon)
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
        assert len(batch) >= 1, "batch must not be empty"
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
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> "AdamVectorizedMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count, beta1, beta2, epsilon)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = np.array(W, dtype=np.float64).copy()
            layer.b = np.array(b, dtype=np.float64).copy()

    def save(self, path: str) -> None:
        save_json(
            path,
            {
                "layer_sizes": self.layer_sizes,
                "dimension": self.dimension,
                "class_count": self.class_count,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "snapshot": [(W.tolist(), b.tolist()) for W, b in self.snapshot()],
            },
        )

    @classmethod
    def load(cls, path: str) -> "AdamVectorizedMultiClassBackpropClassifierNetwork":
        state = load_json(path)
        network = cls(
            state["layer_sizes"],
            state["dimension"],
            state["class_count"],
            state["beta1"],
            state["beta2"],
            state["epsilon"],
        )
        network.restore([(np.array(W), np.array(b)) for W, b in state["snapshot"]])
        return network
