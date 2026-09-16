import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.l2_regularized_backprop_classifier_network import L2RegularizedBackpropClassifierNetwork


def _fixed_network(l2_lambda: float = 0.1) -> L2RegularizedBackpropClassifierNetwork:
    # same dimension=1, one hidden node, one output node, and same starting weights as
    # test_backprop_model.py's own hand-computed fixture
    network = L2RegularizedBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], l2_lambda)
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_is_identical_to_the_plain_sgd_sibling():

    # L2 only changes apply_gradient - the forward pass is untouched, so this must match
    # test_backprop_model.py's own hand-computed forward-pass test exactly
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == pytest.approx(0.5987376536170401)


def test_learn_matches_the_l2_regularized_update_rule_by_hand():

    # pins the backward-pass arithmetic against independently hand-derived expected values - the
    # direct L2-regularized counterpart of
    # test_backprop_model.py::test_learn_matches_the_backprop_update_rule_by_hand, same starting
    # weights/state/learning_rate/category, so the two update rules' actual numeric divergence is
    # directly comparable. a_h=0.7502601055951177, a_o=0.5987376536170401 (same forward pass as
    # the plain-SGD sibling), delta_o=-0.09640363012729687, delta_h=-0.014450509251916271
    # (same deltas too - L2 doesn't change the backward pass, only the weight update). state
    # x=2.0, category y=1.0, learning_rate=0.1, l2_lambda=0.1 - computed independently (not
    # re-derived from the implementation under test): new_w_h=0.49789010185038324,
    # new_b_h=0.10144505092519163 (identical to the plain-SGD sibling's own bias - never
    # regularized), new_w_o=0.7992327797719059, new_b_o=-0.19035963698727032 (also identical to
    # the plain-SGD sibling's own bias)
    network = _fixed_network(l2_lambda=0.1)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.49789010185038324)
    assert hidden_node.bias == pytest.approx(0.10144505092519163)
    assert output_node.input_node_weights[0] == pytest.approx(0.7992327797719059)
    assert output_node.bias == pytest.approx(-0.19035963698727032)


def test_l2_lambda_zero_matches_the_plain_sgd_sibling_bit_for_bit():

    l2_network = _fixed_network(l2_lambda=0.0)
    plain_network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(plain_network)

    l2_network.learn(0.1, (2.0,), 1.0)
    plain_network.learn(0.1, (2.0,), 1.0)

    l2_hidden = l2_network.hidden_layers[0].nodes[0]
    l2_output = l2_network.output_layer.nodes[0]
    plain_hidden = plain_network.hidden_layers[0].nodes[0]
    plain_output = plain_network.output_layer.nodes[0]
    assert l2_hidden.input_node_weights[0] == pytest.approx(plain_hidden.input_node_weights[0])
    assert l2_hidden.bias == pytest.approx(plain_hidden.bias)
    assert l2_output.input_node_weights[0] == pytest.approx(plain_output.input_node_weights[0])
    assert l2_output.bias == pytest.approx(plain_output.bias)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = L2RegularizedBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), l2_lambda=0.01)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = L2RegularizedBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), l2_lambda=0.01)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
