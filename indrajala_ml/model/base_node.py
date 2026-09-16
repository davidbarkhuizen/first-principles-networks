from abc import ABC, abstractmethod
from typing import Sequence


class AbstractNode(ABC):
    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError()


class WeightedInputNode(AbstractNode):
    """
    Shared machinery behind AssociationNode and BackpropNode: both are a node with weighted
    inputs and an additive offset, summed by z() into a single pre-activation value, with an
    independently update-able weight vector. What z() means once computed - a hard step
    (AssociationNode) vs a sigmoid activation (BackpropNode) - is exactly what differs between
    the two, so value() stays abstract here.

    The offset itself is stored as _offset, not exposed directly - each subclass exposes it
    under its own name (threshold vs bias) via a property, since the two names carry real
    meaning (a hard cutoff vs an additive term into a smooth activation), not just cosmetic
    variation.
    """

    def __init__(
        self,
        input_nodes: Sequence[AbstractNode],
        input_node_weights: Sequence[float] | None = None,
        offset: float = 0.0,
        default_input_node_weight: float = 1.0,
    ) -> None:

        self._offset: float = offset

        self.input_nodes: Sequence[AbstractNode] = input_nodes if input_nodes else []

        self.input_node_weights: Sequence[float] = (
            input_node_weights if input_node_weights else [default_input_node_weight for _ in self.input_nodes]
        )

    def update_input_weights(self, weights: list[float]) -> None:
        assert len(weights) == len(self.input_nodes)
        self.input_node_weights = weights

    def z(self) -> float:

        aggregate_input_value: float = sum(
            [self.input_nodes[i].value() * self.input_node_weights[i] for i in range(len(self.input_nodes))]
        )

        return aggregate_input_value + self._offset
