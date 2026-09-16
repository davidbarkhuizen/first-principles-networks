import pytest

from helpers import assert_randomize_breaks_symmetry, assert_save_and_load_round_trip, assert_snapshot_restore_round_trip
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)


def _fixed_network() -> SoftmaxMultiClassBackpropClassifierNetwork:
    # same dimension=1, one hidden node, 2 output classes, and same starting weights as
    # test_multiclass_backprop_model.py's own _fixed_network - deliberately, so the two
    # classes' forward/backward math can be compared side by side from the exact same starting
    # point (see the comment on the learn() test below)
    network = SoftmaxMultiClassBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], 2)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node_0, output_node_1 = network.output_layer.nodes
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node_0.update_input_weights([0.8])
    output_node_0.bias = -0.2
    output_node_1.update_input_weights([-0.3])
    output_node_1.bias = 0.4
    return network


def test_predict_probabilities_matches_a_hand_computed_softmax_forward_pass():

    # z_h = 0.5*2.0 + 0.1 = 1.1, a_h = sigmoid(1.1) = 0.7502601055951177 (unchanged - the
    # hidden layer stays an ordinary sigmoid; only the output layer's activation changes)
    # z_o0 = 0.8*a_h - 0.2 = 0.4002080844760942, z_o1 = -0.3*a_h + 0.4 = 0.17492196832146473
    # softmax(z_o0, z_o1) = (0.5560845207454033, 0.4439154792545967), which sums to 1.0 - unlike
    # test_multiclass_backprop_model.py's own a_o0=0.5987376536170401/a_o1=0.5436193278499907
    # for the *same* z values, which don't (independently computed, not re-derived from the
    # implementation under test)
    network = _fixed_network()

    probabilities = network.predict_probabilities((2.0,))

    assert probabilities[0] == pytest.approx(0.5560845207454033)
    assert probabilities[1] == pytest.approx(0.4439154792545967)
    assert sum(probabilities) == pytest.approx(1.0)


def test_classify_state_returns_the_argmax_class_index():

    network = _fixed_network()

    # output node 0's activation (0.556) > output node 1's (0.444) at this state
    assert network.classify_state((2.0,)) == 0


def test_learn_matches_the_softmax_cross_entropy_update_rule_by_hand():

    # pins the forward+backward arithmetic against independently hand-derived expected values -
    # the direct softmax/cross-entropy counterpart of
    # test_multiclass_backprop_model.py::test_learn_matches_the_one_vs_rest_update_rule_by_hand,
    # same starting weights/state/learning_rate/category, so the two update rules' actual
    # numeric divergence is directly comparable.
    #   a_o0, a_o1 = softmax(z_o0, z_o1) = (0.5560845207454033, 0.4439154792545967)
    #   one-hot target for category=1: output node 0's reference is 0.0, output node 1's is 1.0
    #   delta_o0 = a_o0 - 0.0 = 0.5560845207454033
    #   delta_o1 = a_o1 - 1.0 = -0.5560845207454033
    #   delta_h = (delta_o0*w_o0 + delta_o1*w_o1) * a_h * (1 - a_h) = 0.1146128386373376
    #   w -= learning_rate * delta * <that weight's input value>; b -= learning_rate * delta
    # state x=2.0, category=1, learning_rate=0.1 - computed independently (not re-derived from
    # the implementation under test): new_w_h=0.47707743227253246, new_b_h=0.08853871613626624,
    # new_w_o0=0.7582791968745743, new_b_o0=-0.25560845207454036,
    # new_w_o1=-0.25827919687457435, new_b_o1=0.45560845207454037
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node_0, output_node_1 = network.output_layer.nodes

    network.learn(0.1, (2.0,), 1)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.47707743227253246)
    assert hidden_node.bias == pytest.approx(0.08853871613626624)
    assert output_node_0.input_node_weights[0] == pytest.approx(0.7582791968745743)
    assert output_node_0.bias == pytest.approx(-0.25560845207454036)
    assert output_node_1.input_node_weights[0] == pytest.approx(-0.25827919687457435)
    assert output_node_1.bias == pytest.approx(0.45560845207454037)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = SoftmaxMultiClassBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), 3)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = SoftmaxMultiClassBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), 4)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 2))


def test_save_and_load_round_trip(tmp_path):

    network = SoftmaxMultiClassBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), 4)
    for _ in range(5):
        network.learn(0.1, (1.0, -2.0), 2)

    loaded = assert_save_and_load_round_trip(
        network, SoftmaxMultiClassBackpropClassifierNetwork.load, tmp_path, "model.json", [(1.0, -2.0)]
    )

    assert isinstance(loaded, SoftmaxMultiClassBackpropClassifierNetwork)
    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count
    assert loaded.input_bounds == network.input_bounds

    # the loaded network's output layer must still be a genuine softmax layer, not a plain
    # BackpropLayer - load() reconstructs via cls(...), so this would only fail if that
    # polymorphism were somehow broken
    assert sum(loaded.predict_probabilities((1.0, -2.0))) == pytest.approx(1.0)
