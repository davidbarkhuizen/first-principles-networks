import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork


def _fixed_network() -> BackpropClassifierNetwork:
    # dimension=1, one hidden node, one output node - small enough to state the whole
    # forward/backward pass exactly; every *_backprop_model.py sibling wires the identical
    # starting weights (wire_fixed_single_hidden_node) for a side-by-side comparison
    network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_matches_a_hand_computed_forward_pass():

    # a minimal 1D-input, single-hidden-node, single-output-node network, small enough to
    # state the whole forward pass exactly: z_h = w_h*x + b_h, a_h = sigmoid(z_h),
    # z_o = w_o*a_h + b_o, a_o = sigmoid(z_o). With w_h=0.5, b_h=0.1, w_o=0.8, b_o=-0.2,
    # x=2.0: z_h=1.1, a_h=sigmoid(1.1)=0.7502601055951177, z_o=0.8*a_h-0.2=0.4002080844760941,
    # a_o=sigmoid(z_o)=0.5987376536170401 (independently computed, not re-derived from the
    # implementation under test).
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == pytest.approx(0.5987376536170401)


def test_classify_state_thresholds_strictly_above_half():

    network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.0])
    hidden_node.bias = 0.0
    output_node.update_input_weights([0.0])
    output_node.bias = 0.0

    # z_o = 0.0 -> a_o = sigmoid(0) = 0.5 exactly - must not classify as active
    assert network.predict_probability((0.0,)) == pytest.approx(0.5)
    assert network.classify_state((0.0,)) == 0.0


def test_learn_matches_the_backprop_update_rule_by_hand():

    # pins the forward+backward arithmetic against independently hand-derived expected
    # values (same minimal network as the forward-pass test above):
    #   delta_o = (a_o - y) * a_o * (1 - a_o)
    #   delta_h = (delta_o * w_o) * a_h * (1 - a_h)
    #   w -= learning_rate * delta * <that weight's input value>;  b -= learning_rate * delta
    # starting weights w_h=0.5, b_h=0.1, w_o=0.8, b_o=-0.2; state x=2.0, category y=1.0,
    # learning_rate=0.1 - computed independently (not re-derived from the implementation under
    # test): delta_o=-0.09640363012729687, delta_h=-0.014450509251916271, giving
    # new_w_h=0.5028901018503833, new_b_h=0.10144505092519163, new_w_o=0.8072327797719059,
    # new_b_o=-0.19035963698727032.
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5028901018503833)
    assert hidden_node.bias == pytest.approx(0.10144505092519163)
    assert output_node.input_node_weights[0] == pytest.approx(0.8072327797719059)
    assert output_node.bias == pytest.approx(-0.19035963698727032)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = BackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
