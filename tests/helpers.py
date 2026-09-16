import random
from typing import Callable

import numpy as np
import pytest

from perceptron.model.linear_classifier_network import LinearClassifierNetwork
from perceptron.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork


def assert_save_and_load_round_trip(network, load_fn, tmp_path, filename: str, states):
    """
    Shared core of every save/load round-trip test in this codebase (the *multiclass/softmax/
    ensemble/conv sibling model classes): save network to a temp path, reload it via load_fn,
    and confirm the reloaded network's snapshot and predictions match the original exactly.
    Returns the loaded network so each call site can layer its own class-specific assertions
    (dimension, class_count, input_bounds, conv hyperparameters, ...) on top.
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.snapshot() == network.snapshot()
    for state in states:
        assert loaded.classify_state(state) == network.classify_state(state)
        assert loaded.predict_probabilities(state) == pytest.approx(network.predict_probabilities(state))

    return loaded


def assert_randomize_breaks_symmetry(network) -> None:
    """
    Shared by every *_backprop_model.py's test_randomize_breaks_symmetry_between_nodes_...:
    identical starting weights across nodes in the same layer would receive identical gradients
    forever and the layer would collapse to one effective unit, so randomize() must give each
    node independent random weights, not a shared default.
    """
    weight_sets = [tuple(node.input_node_weights) for node in network.hidden_layers[0].nodes]
    assert len(set(weight_sets)) == len(weight_sets)


def assert_snapshot_restore_round_trip(network, step, times: int = 5) -> None:
    """
    Shared by every *_backprop_model.py's/ensemble's/conv sibling's
    test_snapshot_and_restore_round_trip: snapshot, apply `step` `times` times (confirming the
    network actually moved), restore, and confirm the snapshot matches the pre-training one
    exactly again.
    """
    before = network.snapshot()

    for _ in range(times):
        step()

    assert network.snapshot() != before

    network.restore(before)

    assert network.snapshot() == before


def wire_fixed_single_hidden_node(network) -> None:
    """
    Shared single-hidden-node/single-output-node weight wiring behind every
    *_backprop_model.py's own _fixed_network(): hidden weight=0.5, bias=0.1, output weight=0.8,
    bias=-0.2 - the exact values test_backprop_model.py's own hand-computed forward/backward
    tests derive their expected numbers from, reused by every sibling that needs the identical
    starting point for a side-by-side comparison.
    """
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node.update_input_weights([0.8])
    output_node.bias = -0.2


def matching_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
):
    """
    Shared by test_vectorized_multiclass_backprop_model.py's and
    test_rust_array_multiclass_backprop_model.py's own _matching_networks: builds a
    MultiClassBackpropClassifierNetwork and an array-backed sibling
    (VectorizedMultiClassBackpropClassifierNetwork / RustArrayMultiClassBackpropClassifierNetwork,
    passed as array_network_cls) with identical injected weights. Neither array backend's RNG
    stream is meaningfully comparable to Python's random module (see docs/rust-array-core.md's
    "the RNG exception"), so initial weights are always forced identical explicitly here instead
    of via each network's own randomize(). `wrap` converts a nested Python list of weights (or a
    flat list of biases) into the array backend's own array type - np.array for the numpy
    sibling, pa.Array for the Rust one.
    """
    node_network = MultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


def assert_array_network_weights_match(node_network, array_network, rtol=1e-9, atol=1e-9) -> None:
    """
    Shared by both array-backed siblings' own _assert_networks_match: both numpy ndarrays and
    pa.Array support .tolist(), so the same comparison works against either backend without
    needing a backend-specific read path.
    """
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        expected_W = np.array([node.input_node_weights for node in node_layer.nodes])
        expected_b = np.array([node.bias for node in node_layer.nodes])
        assert np.allclose(array_layer.W.tolist(), expected_W, rtol=rtol, atol=atol)
        assert np.allclose(array_layer.b.tolist(), expected_b, rtol=rtol, atol=atol)


def assert_array_network_snapshot_restore_round_trip(
    array_network_cls, layer_sizes: list[int], dimension: int, class_count: int
) -> None:
    """
    Shared by both array-backed siblings' own test_snapshot_restore_round_trips_weights - same
    .tolist()-based equality reasoning as assert_array_network_weights_match above.
    """
    network = array_network_cls.randomized(layer_sizes, dimension, class_count)
    snapshot = network.snapshot()

    other = array_network_cls(layer_sizes, dimension, class_count)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def assert_array_network_save_load_round_trip(network, load_fn, tmp_path, filename: str, state):
    """
    Array-backed analogue of assert_save_and_load_round_trip above: a numpy/pa.Array snapshot
    element doesn't support a plain == equality check the way a node network's snapshot() does
    (it's elementwise, not a single bool), so this checks the JSON envelope's scalar fields plus
    predict_probabilities via pytest.approx instead of snapshot() equality. Shared by both
    test_vectorized_multiclass_backprop_model.py's and
    test_rust_array_multiclass_backprop_model.py's own test_save_load_round_trips_weights_and_predictions.
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count
    assert loaded.predict_probabilities(state) == pytest.approx(network.predict_probabilities(state))

    return loaded


def classifier_with_bounded_square_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region is exactly the square [-1, 1] x [-1, 1]: x > -1, x < 1, y > -1, y < 1
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 1.0), ([-1.0, 0.0], 1.0), ([0.0, 1.0], 1.0), ([0.0, -1.0], 1.0)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def classifier_with_tiny_bounded_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region is exactly [-0.1, 0.1] x [-0.1, 0.1] - a 0.04 unit^2 square, a tiny
    # fraction of a square_bounds(10.0)-sized (400 unit^2) box (0.01%), for exercising the
    # tight-box positive-region sampling optimisation (see geometry.positive_region_bounding_box)
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 0.1), ([-1.0, 0.0], 0.1), ([0.0, 1.0], 0.1), ([0.0, -1.0], 0.1)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def unreachable_class_classifier(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # tiny weights + a large threshold mean the decision boundary never crosses these bounds,
    # so one class can never be sampled - used to exercise the safety guard against an
    # unreachable-class sampling loop hanging forever (see
    # demo_unreachable_class_safety_guard.py for the same idea as a standalone demo)
    classifier = LinearClassifierNetwork(1, 2, bounds)
    node = classifier.hidden_layer.nodes[0]
    node.update_input_weights([0.01, 0.01])
    node.threshold = -5.0
    return classifier


def network_with_hidden_thresholds(
    dimension: int,
    bounds: list[tuple[float, float]],
    thresholds: list[float],
    required_active: int | None = None,
) -> LinearClassifierNetwork:

    # zero input weights make each node's z() equal to its threshold alone, regardless of
    # state - so the thresholds directly pick each node's active/inactive status
    network = LinearClassifierNetwork(len(thresholds), dimension, bounds, required_active)
    for node, threshold in zip(network.hidden_layer.nodes, thresholds):
        node.update_input_weights([0.0] * dimension)
        node.threshold = threshold
    return network
