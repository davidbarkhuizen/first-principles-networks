import pytest

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls, make_momentum_node_cls
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.state_node import StateNode


def _momentum_node(momentum: float, weight: float, bias: float):
    node_cls = make_momentum_node_cls(momentum)
    x = StateNode(1.0)
    node = node_cls(input_nodes=[x])
    node.update_input_weights([weight])
    node.bias = bias
    return node


def test_first_apply_gradient_matches_plain_sgd_since_there_is_no_prior_delta():

    # with no previous step, the momentum term (momentum * 0.0) contributes nothing - the very
    # first update must be bit-identical to BackpropNode.apply_gradient's own plain SGD step
    momentum_node = _momentum_node(0.9, weight=0.5, bias=0.1)
    plain_node = BackpropNode(input_nodes=[StateNode(1.0)])
    plain_node.update_input_weights([0.5])
    plain_node.bias = 0.1

    momentum_node.delta = 0.2
    plain_node.delta = 0.2
    momentum_node.apply_gradient(0.1)
    plain_node.apply_gradient(0.1)

    assert momentum_node.input_node_weights[0] == pytest.approx(plain_node.input_node_weights[0])
    assert momentum_node.bias == pytest.approx(plain_node.bias)


def test_second_apply_gradient_adds_the_momentum_term_by_hand():

    # two consecutive steps with the same delta, so the momentum term's actual contribution is
    # exercised (the first step alone can't distinguish momentum from plain SGD - see the test
    # above). weight=0.5, bias=0.1, x=1.0, delta=0.2 (held constant across both steps),
    # learning_rate=0.1, momentum=0.9 - computed independently (not re-derived from the
    # implementation under test):
    #   step 1: delta_w = 0.1*0.2*1.0 + 0.9*0.0 = 0.02 -> weight = 0.5 - 0.02 = 0.48
    #           bias_delta = 0.1*0.2 + 0.9*0.0 = 0.02 -> bias = 0.1 - 0.02 = 0.08
    #   step 2: delta_w = 0.1*0.2*1.0 + 0.9*0.02 = 0.038 -> weight = 0.48 - 0.038 = 0.442
    #           bias_delta = 0.1*0.2 + 0.9*0.02 = 0.038 -> bias = 0.08 - 0.038 = 0.042
    node = _momentum_node(0.9, weight=0.5, bias=0.1)

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert node.input_node_weights[0] == pytest.approx(0.48)
    assert node.bias == pytest.approx(0.08)

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert node.input_node_weights[0] == pytest.approx(0.442)
    assert node.bias == pytest.approx(0.042)


def test_zero_momentum_is_bit_identical_to_plain_sgd_across_many_steps():

    # momentum=0.0 should behave exactly like BackpropNode regardless of how many steps run,
    # since the momentum term always contributes momentum * anything = 0.0
    momentum_node = _momentum_node(0.0, weight=0.5, bias=0.1)
    plain_node = BackpropNode(input_nodes=[StateNode(1.0)])
    plain_node.update_input_weights([0.5])
    plain_node.bias = 0.1

    for delta in [0.2, -0.1, 0.05, 0.3, -0.4]:
        momentum_node.delta = delta
        plain_node.delta = delta
        momentum_node.apply_gradient(0.1)
        plain_node.apply_gradient(0.1)

    assert momentum_node.input_node_weights[0] == pytest.approx(plain_node.input_node_weights[0])
    assert momentum_node.bias == pytest.approx(plain_node.bias)


def test_make_momentum_layer_cls_builds_nodes_of_the_configured_momentum_node_class():

    input_layer = StateLayer(2, [(-10.0, 10.0), (-10.0, 10.0)])
    layer_cls = make_momentum_layer_cls(0.5)
    layer = layer_cls(size=3, input_layer=input_layer)

    node_cls = make_momentum_node_cls(0.5)
    assert all(type(node).__name__ == node_cls.__name__ for node in layer.nodes)
    assert all(hasattr(node, "_prev_weight_deltas") for node in layer.nodes)
