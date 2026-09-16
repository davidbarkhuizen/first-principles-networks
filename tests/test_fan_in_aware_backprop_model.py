import math

import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.fan_in_aware_backprop_classifier_network import FanInAwareBackpropClassifierNetwork


def test_predict_probability_is_identical_to_the_default_init_sibling():

    # randomize() only changes how weights start out, not the forward pass - hand-set weights
    # must produce exactly the same prediction as BackpropClassifierNetwork's own hand-computed
    # forward-pass test (test_backprop_model.py::test_predict_probability_matches_a_hand_computed_forward_pass)
    network = FanInAwareBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)

    assert network.predict_probability((2.0,)) == pytest.approx(0.5987376536170401)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = FanInAwareBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_randomize_scales_weight_range_with_fan_in():

    # limit = 1/sqrt(fan_in) - a wider first layer (larger fan-in for the output layer) should
    # produce a visibly narrower output-layer weight range than a narrow one - the same property
    # test_multiclass_backprop_model.py checks for MultiClassBackpropClassifierNetwork, which
    # shares this exact scheme via randomize_fan_in_aware
    narrow = FanInAwareBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    wide = FanInAwareBackpropClassifierNetwork.randomized([400], 2, square_bounds(10.0))

    narrow_output_range = max(abs(w) for w in narrow.output_layer.nodes[0].input_node_weights)
    wide_output_range = max(abs(w) for w in wide.output_layer.nodes[0].input_node_weights)

    assert wide_output_range < narrow_output_range


def test_randomize_weight_magnitude_matches_the_fan_in_formula():

    # every weight must lie within [-limit, limit] where limit=1/sqrt(dimension) for the first
    # hidden layer - a direct check of the formula itself, not just its qualitative effect
    dimension = 64
    network = FanInAwareBackpropClassifierNetwork.randomized([8], dimension, square_bounds(10.0, dimension))

    limit = 1.0 / math.sqrt(dimension)
    for node in network.hidden_layers[0].nodes:
        assert all(-limit <= w <= limit for w in node.input_node_weights)
        assert -limit <= node.bias <= limit


def test_snapshot_and_restore_round_trip():

    network = FanInAwareBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
