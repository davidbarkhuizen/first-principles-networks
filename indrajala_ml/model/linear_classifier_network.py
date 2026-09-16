from __future__ import annotations

import random

from indrajala_ml.model.association_layer import AssociationLayer
from indrajala_ml.model.bounds import half_widths as _half_widths
from indrajala_ml.model.bounds import validate_input_bounds
from indrajala_ml.model.state_layer import StateLayer


class LinearClassifierNetwork:
    def __init__(
        self,
        cardinality: int,
        dimension: int,
        input_bounds: list[tuple[float, float]],
        required_active: int | None = None,
    ) -> None:

        assert cardinality >= 1
        self.cardinality = cardinality

        self.dimension = dimension

        validate_input_bounds(dimension, input_bounds)
        self.input_bounds = input_bounds

        # how many of the cardinality hidden nodes must be active for the output to fire -
        # defaults to cardinality (AND, i.e. every hidden node must agree). 1 gives OR (any
        # one hidden node is enough); anything in between gives a general k-of-n gate, in
        # the spirit of a MADALINE-style committee machine.
        required_active = cardinality if required_active is None else required_active
        assert 1 <= required_active <= cardinality
        self.required_active = required_active

        self.input_layer = StateLayer(dimension, input_bounds)

        self.hidden_layer = AssociationLayer(size=cardinality, input_layer=self.input_layer)

        # the output layer is a single neuron, fully connected to the hidden layer, that
        # activates once at least required_active of its input nodes are active
        self.output_layer = AssociationLayer(size=1, input_layer=self.hidden_layer)
        output_node = self.output_layer.nodes[0]
        output_node.update_input_weights([1.0 for _ in self.hidden_layer.nodes])
        output_node.threshold = -float(self.required_active - 1)

    def update_state_layer(self, x_: tuple[float, ...]) -> None:
        self.input_layer.update_state(x_)

    def classify_state(self, state: tuple[float, ...]) -> float:
        self.update_state_layer(state)
        return self.output_layer.nodes[0].value()

    def learn(self, learning_rate: float, state: tuple[float, ...], category: float) -> None:

        self.update_state_layer(state)

        output = self.output_layer.nodes[0].value()
        if output == category:
            return

        if category == 1:
            # false negative: too few hidden nodes are active - flipping any inactive one
            # to active can only increase the count, moving the output toward firing.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 0.0]
        else:
            # false positive: enough hidden nodes are active to fire - flipping any active
            # one to inactive can only decrease the count, moving the output toward silent.
            # This (and the branch above) holds for any required_active, not just AND -
            # the output is a monotonically non-decreasing function of how many hidden
            # nodes are active, regardless of the specific threshold.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 1.0]

        responsible_node = min(candidates, key=lambda node: abs(node.z()))
        responsible_node.learn(learning_rate, category)

    def half_widths(self) -> list[float]:
        return _half_widths(self.input_bounds)

    def randomize(self) -> None:
        # each weight's range scales inversely with its own dimension's half-width, so that
        # w_i * x_i has a similar typical magnitude no matter how large or small that
        # dimension's bounds are (and independently of the other dimensions' bounds, so
        # asymmetric bounds are handled correctly too). The threshold's range doesn't need to
        # scale at all once weights are normalised this way. Without this, a hidden node's
        # weights stayed fixed while x shrank or grew with the bounds, so the threshold ended
        # up dominating w.x whenever the bounds were much smaller than the range was
        # implicitly tuned for, making almost every random node permanently active or
        # permanently inactive (verified: at bounds half-width 0.001, 0/200 random
        # cardinality=1 classifiers had both classes reachable at all). Calibrated so that at
        # half-width 10 - what every existing demo and test uses - this reduces to exactly
        # uniform(-2, 2) per weight, unchanged.
        half_widths = self.half_widths()
        for node in self.hidden_layer.nodes:
            node.update_input_weights(
                [random.uniform(-20.0 / half_width, 20.0 / half_width) for half_width in half_widths]
            )
            node.threshold = random.uniform(-5, 5)

    @classmethod
    def randomized(
        cls,
        cardinality: int,
        dimension: int,
        input_bounds: list[tuple[float, float]],
        required_active: int | None = None,
    ) -> LinearClassifierNetwork:
        network = cls(cardinality, dimension, input_bounds, required_active)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple[list[float], float]]:
        return [(list(node.input_node_weights), node.threshold) for node in self.hidden_layer.nodes]

    def restore(self, snapshot: list[tuple[list[float], float]]) -> None:
        for node, (weights, threshold) in zip(self.hidden_layer.nodes, snapshot):
            node.update_input_weights(weights)
            node.threshold = threshold
