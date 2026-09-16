import random

import numpy as np
import pytest

import perceptron_array as pa
from perceptron.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from perceptron.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3


def _matching_networks(rng: random.Random):
    # tier 1 (docs/rust-production-cutover.md's phase 2) - identical fixed weights/inputs
    # injected directly, never randomize(), since perceptron_array.uniform's RNG can never be
    # seed-comparable against Python's random module (see rust-array-core.md's "the RNG
    # exception"). The array-vs-node analogue of
    # tests/test_vectorized_multiclass_backprop_model.py's own _matching_networks.
    node_network = MultiClassBackpropClassifierNetwork(
        LAYER_SIZES, DIMENSION, [(-10.0, 10.0)] * DIMENSION, CLASS_COUNT
    )
    rust_network = RustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)

    previous_size = DIMENSION
    for layer_index, size in enumerate([*LAYER_SIZES, CLASS_COUNT]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        rust_network.layers[layer_index].W = pa.Array(weights)
        rust_network.layers[layer_index].b = pa.Array(biases)

        previous_size = size

    return node_network, rust_network


def _assert_networks_match(node_network, rust_network, rtol=1e-9, atol=1e-9):
    for node_layer, rust_layer in zip(node_network.trainable_layers, rust_network.layers):
        expected_W = np.array([node.input_node_weights for node in node_layer.nodes])
        expected_b = np.array([node.bias for node in node_layer.nodes])
        actual_W = np.array(rust_layer.W.tolist())
        actual_b = np.array(rust_layer.b.tolist())
        assert np.allclose(actual_W, expected_W, rtol=rtol, atol=atol)
        assert np.allclose(actual_b, expected_b, rtol=rtol, atol=atol)


def test_predict_probabilities_matches_across_a_random_sweep():

    rng = random.Random(0)
    node_network, rust_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probabilities(state)
        actual = rust_network.predict_probabilities(state)
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_classify_state_matches_across_a_random_sweep():

    rng = random.Random(1)
    node_network, rust_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert rust_network.classify_state(state) == node_network.classify_state(state)


def test_learn_matches_after_every_step_not_just_at_the_end():

    # one silently-wrong intermediate step should fail loudly rather than being averaged away
    # by many steps - tier 1's own required regression gate (docs/rust-production-cutover.md).
    rng = random.Random(2)
    node_network, rust_network = _matching_networks(rng)
    learning_rate = 0.3

    for step in range(100):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = rng.randrange(CLASS_COUNT)

        node_network.learn(learning_rate, state, category)
        rust_network.learn(learning_rate, state, category)

        _assert_networks_match(node_network, rust_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end():

    rng = random.Random(3)
    node_network, rust_network = _matching_networks(rng)
    learning_rate = 0.3
    batch_size = 8

    for _ in range(20):
        batch = [
            (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), rng.randrange(CLASS_COUNT))
            for _ in range(batch_size)
        ]

        node_network.learn_batch(learning_rate, batch)
        rust_network.learn_batch(learning_rate, batch)

        _assert_networks_match(node_network, rust_network)


def test_randomized_builds_a_usable_network():

    network = RustArrayMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    state = tuple(0.1 * i for i in range(DIMENSION))

    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < CLASS_COUNT


def test_snapshot_restore_round_trips_weights():

    network = RustArrayMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    snapshot = network.snapshot()

    other = RustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def test_save_load_round_trips_weights_and_predictions(tmp_path):

    network = RustArrayMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    path = str(tmp_path / "rust_array_model.json")
    network.save(path)

    loaded = RustArrayMultiClassBackpropClassifierNetwork.load(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count

    state = tuple(0.1 * i for i in range(DIMENSION))
    assert loaded.predict_probabilities(state) == pytest.approx(network.predict_probabilities(state))


def test_construction_rejects_invalid_arguments():

    with pytest.raises(AssertionError):
        RustArrayMultiClassBackpropClassifierNetwork([], DIMENSION, CLASS_COUNT)

    with pytest.raises(AssertionError):
        RustArrayMultiClassBackpropClassifierNetwork([0], DIMENSION, CLASS_COUNT)

    with pytest.raises(AssertionError):
        RustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, class_count=1)


def test_learn_batch_rejects_an_empty_batch():

    network = RustArrayMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
