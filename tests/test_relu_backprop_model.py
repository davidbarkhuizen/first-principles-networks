import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.relu_backprop_classifier_network import ReLUBackpropClassifierNetwork


def _fixed_network(x: float = 2.0) -> ReLUBackpropClassifierNetwork:
    # same dimension=1, one hidden node, one output node, and same starting weights as
    # test_backprop_model.py's own hand-computed fixture - deliberately, so the two hidden-layer
    # activations' forward/backward math can be compared side by side from the exact same
    # starting point
    network = ReLUBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_matches_a_hand_computed_forward_pass():

    # z_h = 0.5*2.0 + 0.1 = 1.1 (positive) -> a_h = relu(1.1) = 1.1 - genuinely different from
    # test_backprop_model.py's own a_h = sigmoid(1.1) = 0.7502601055951177 for this same z_h
    # z_o = 0.8*1.1 - 0.2 = 0.6800000000000002, a_o = sigmoid(z_o) = 0.6637386974043528
    # (independently computed, not re-derived from the implementation under test)
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == pytest.approx(0.6637386974043528)


def test_learn_matches_the_relu_hidden_backprop_update_rule_by_hand():

    # pins the forward+backward arithmetic against independently hand-derived expected values -
    # the direct ReLU-hidden counterpart of
    # test_backprop_model.py::test_learn_matches_the_backprop_update_rule_by_hand, same starting
    # weights/state/learning_rate/category, so the two hidden activations' actual numeric
    # divergence is directly comparable.
    #   a_h = 1.1 (relu, not sigmoid), a_o = sigmoid(0.68) = 0.6637386974043528
    #   delta_o = (a_o - y) * a_o * (1 - a_o) = -0.0750500387266865
    #   delta_h = (delta_o * w_o) * (1.0 if a_h > 0 else 0.0) = -0.060040030981349204 (a_h > 0,
    #   so the ReLU derivative factor is 1.0 - unlike sigmoid's a*(1-a) damping term)
    #   w -= learning_rate * delta * <that weight's input value>; b -= learning_rate * delta
    # state x=2.0, category y=1.0, learning_rate=0.1 - computed independently (not re-derived
    # from the implementation under test): new_w_h=0.5120080061962698,
    # new_b_h=0.10600400309813493, new_w_o=0.8082555042599355, new_b_o=-0.19249499612733137
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5120080061962698)
    assert hidden_node.bias == pytest.approx(0.10600400309813493)
    assert output_node.input_node_weights[0] == pytest.approx(0.8082555042599355)
    assert output_node.bias == pytest.approx(-0.19249499612733137)


def test_learn_leaves_a_dead_units_incoming_weights_unchanged():

    # x=-10.0 makes z_h = 0.5*(-10.0)+0.1 = -4.9 (negative) -> a_h = 0.0, a dead ReLU unit.
    # Hand-derived: delta_h = 0.0 exactly (the ReLU derivative factor is 0 when inactive), so the
    # hidden node's own weight/bias are completely unchanged by learn() - and since a_h=0.0 is
    # what the output node's weight gradient multiplies by, the output weight is also unchanged
    # (0.8 exactly), even though the output *bias* still updates (bias gradient doesn't depend on
    # a_h). z_o = 0.8*0.0 - 0.2 = -0.2, a_o = sigmoid(-0.2) = 0.45016600268752216, delta_o =
    # -0.13609302657524652 - computed independently (not re-derived from the implementation
    # under test): new_b_o = -0.18639069734247535.
    network = _fixed_network(x=-10.0)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    assert network.predict_probability((-10.0,)) == pytest.approx(0.45016600268752216)

    network.learn(0.1, (-10.0,), 1.0)

    assert hidden_node.input_node_weights[0] == 0.5
    assert hidden_node.bias == 0.1
    assert output_node.input_node_weights[0] == 0.8
    assert output_node.bias == pytest.approx(-0.18639069734247535)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = ReLUBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = ReLUBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
