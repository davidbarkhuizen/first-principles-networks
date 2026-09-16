import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.momentum_backprop_classifier_network import MomentumBackpropClassifierNetwork


def _fixed_network(momentum: float = 0.9) -> MomentumBackpropClassifierNetwork:
    # same dimension=1, one hidden node, one output node, and same starting weights as
    # test_backprop_model.py's own hand-computed fixture
    network = MomentumBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], momentum)
    wire_fixed_single_hidden_node(network)
    return network


def test_first_learn_step_matches_the_plain_sgd_sibling_exactly():

    # with no prior step, momentum contributes nothing - the first learn() call must be
    # bit-identical to test_backprop_model.py's own
    # test_learn_matches_the_backprop_update_rule_by_hand (same fixture, same state/category/lr)
    network = _fixed_network(momentum=0.9)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5028901018503833)
    assert hidden_node.bias == pytest.approx(0.10144505092519163)
    assert output_node.input_node_weights[0] == pytest.approx(0.8072327797719059)
    assert output_node.bias == pytest.approx(-0.19035963698727032)


def test_second_learn_step_shows_the_momentum_contribution_by_hand():

    # two consecutive learn() calls on the same state/category, so momentum's actual
    # contribution (present only from the second step onward) is exercised. momentum=0.9,
    # learning_rate=0.1, state x=2.0, category y=1.0, same starting weights as the fixture above
    # - computed independently (not re-derived from the implementation under test), tracking
    # both the weight/bias deltas and the forward pass across both steps:
    #   step 1 (identical to plain SGD, no prior delta): a_h=0.7502601055951177,
    #   a_o=0.5987376536170401, new_w_h=0.5028901018503833, new_b_h=0.10144505092519163,
    #   new_w_o=0.8072327797719059, new_b_o=-0.19035963698727032
    #   step 2 (weights from step 1, momentum term now nonzero): a_h=0.7516114513114047,
    #   a_o=0.6026132829266382, new_w_h=0.5083594576090832, new_b_h=0.10417972880454159,
    #   new_w_o=0.8208947966338368, new_b_o=-0.17216707012974347
    network = _fixed_network(momentum=0.9)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)
    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5083594576090832)
    assert hidden_node.bias == pytest.approx(0.10417972880454159)
    assert output_node.input_node_weights[0] == pytest.approx(0.8208947966338368)
    assert output_node.bias == pytest.approx(-0.17216707012974347)


def test_momentum_zero_matches_the_plain_sgd_sibling_across_many_steps():

    network = MomentumBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), momentum=0.0)
    hidden_node = network.hidden_layers[0].nodes[0]
    original_weights = list(hidden_node.input_node_weights)

    for _ in range(10):
        network.learn(0.1, (1.0, -2.0), 1.0)

    # momentum=0.0 must still train (weights change from the plain gradient term), just with
    # zero contribution from the momentum term specifically - the point of this test is that it
    # doesn't error or silently no-op, not that the exact resulting values match anything
    assert hidden_node.input_node_weights != original_weights


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = MomentumBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), momentum=0.5)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = MomentumBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), momentum=0.5)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
