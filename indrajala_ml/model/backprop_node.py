from __future__ import annotations

import math
from typing import Sequence

from indrajala_ml.model.base_node import AbstractNode, WeightedInputNode


def sigmoid(z: float) -> float:
    """
    The standard logistic sigmoid, 1/(1+e^-z) - the textbook formula directly, except caught
    against the OverflowError math.exp(-z) raises for z below about -709 (confirmed directly:
    math.exp(710) raises OverflowError, math.exp(700) doesn't) - a real, reachable value for a
    node whose weighted input sum has drifted large and negative during training, not just a
    theoretical edge case. Deliberately not the usual piecewise e^z/(1+e^z) reformulation for
    z < 0: that avoids the overflow too, but rounds differently from 1/(1+e^-z) across the
    entire negative range, not just the extreme tail - which would silently perturb every
    already-passing seeded/pinned training result in this codebase's tests, for a discrepancy
    that only actually matters in the (unreached, in practice) extreme tail. In that tail,
    1/(1+e^-z) is mathematically indistinguishable from 0.0 in float64 anyway (e^-z is already
    astronomically larger than 1), so catching the overflow and returning that same limiting
    value exactly reproduces what the direct formula would have computed had it not overflowed.
    """

    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0


class BackpropNode(WeightedInputNode):
    """
    Sigmoid-activation neuron trained by gradient descent, in contrast to AssociationNode's
    hard step function and discrete minimum-disturbance update rule. bias plays the same role
    as AssociationNode's threshold (added to the weighted input sum before activation), but is
    named differently to signal it's additive rather than a hard cutoff.
    """

    def __init__(
        self,
        input_nodes: Sequence[AbstractNode],
        input_node_weights: Sequence[float] | None = None,
        bias: float = 0.0,
    ) -> None:
        super().__init__(input_nodes, input_node_weights, offset=bias)

        # populated by forward(); no default - value() must never be called before a forward()
        # pass, that's a caller bug, not something to paper over with a fallback
        self._activation: float

        # populated by compute_output_delta()/compute_hidden_delta() during the backward pass
        self.delta: float

        # accumulated by accumulate_gradient(), consumed and reset by
        # apply_accumulated_gradient() - lets a caller run forward+backward+accumulate over an
        # entire mini-batch before any weight is actually written, instead of apply_gradient's
        # single-example fused compute-and-apply
        self._weight_gradient_accum: list[float] = [0.0 for _ in self.input_nodes]
        self._bias_gradient_accum: float = 0.0

    @property
    def bias(self) -> float:
        return self._offset

    @bias.setter
    def bias(self, value: float) -> None:
        self._offset = value

    def forward(self) -> float:
        # the only place activation is computed - value() is a pure cache read, so downstream
        # nodes reading this one multiple times in a forward pass don't each pay for a fresh
        # sigmoid evaluation
        self._activation = sigmoid(self.z())
        return self._activation

    def value(self) -> float:
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        a = self.value()
        self.delta = (a - reference_value) * a * (1.0 - a)

    def compute_hidden_delta(self, next_layer_nodes: Sequence["BackpropNode"], own_index: int) -> None:
        # relies on every node in next_layer_nodes sharing this node's position (own_index) in
        # its own input_node_weights - true by construction, since every node in a
        # BackpropLayer is built from the same input_layer.nodes list
        a = self.value()
        downstream = sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)
        self.delta = downstream * a * (1.0 - a)

    def accumulate_gradient(self) -> None:
        for i, node in enumerate(self.input_nodes):
            self._weight_gradient_accum[i] += self.delta * node.value()
        self._bias_gradient_accum += self.delta

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.update_input_weights(
            [
                weight - learning_rate * accum / batch_size
                for weight, accum in zip(self.input_node_weights, self._weight_gradient_accum)
            ]
        )
        self.bias = self.bias - learning_rate * self._bias_gradient_accum / batch_size
        self._reset_gradient_accum()

    def _reset_gradient_accum(self) -> None:
        self._weight_gradient_accum = [0.0 for _ in self.input_nodes]
        self._bias_gradient_accum = 0.0

    def apply_gradient(self, learning_rate: float) -> None:
        # the batch_size=1 special case of accumulate + apply, kept as the single-example entry
        # point every existing caller (learn(), every pinned test) already uses - dispatches
        # through accumulate_gradient()/apply_accumulated_gradient() polymorphically, so a
        # subclass overriding either (e.g. MomentumBackpropNode, L2RegularizedBackpropNode) gets
        # its own per-example behavior without needing to override apply_gradient itself
        self.accumulate_gradient()
        self.apply_accumulated_gradient(learning_rate, 1)
