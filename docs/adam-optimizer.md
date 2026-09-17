# Adam: a per-parameter adaptive-learning-rate optimizer

[← back to README](../README.md)

Design reference for `AdamBackpropClassifierNetwork` (see
[structure](structure.md#backprop-siblings)). Built and measured; see [research and
analysis](research-adam-optimizer.md) for the full measurement and decision.

## design: fits the existing accumulate/apply seam, no new mechanism needed

Same factory pattern as `momentum_layer.py`/`l2_regularization_layer.py`:

- `indrajala_ml/model/adam_layer.py`: `make_adam_node_cls(beta1, beta2, epsilon)` -> a
  `BackpropNode` subclass overriding only `apply_accumulated_gradient` (not
  `accumulate_gradient` - the existing batch-gradient-accumulation mechanism is already generic
  and correct as-is, the same override point momentum uses).
- `indrajala_ml/model/adam_backprop_classifier_network.py`: `AdamBackpropClassifierNetwork`,
  setting `hidden_layer_cls`/`output_layer_cls` to the Adam-configured layer class in `__init__`
  before `super().__init__()` runs - structurally identical to `MomentumBackpropClassifierNetwork`.

Per-parameter state, mirroring momentum's `_prev_weight_deltas`/`_prev_bias_delta`:

- `_weight_m`, `_weight_v` (lists, zero-initialized, one entry per input weight)
- `_bias_m`, `_bias_v` (scalars)
- `_t` (an integer step counter, zero-initialized, incremented at the top of every
  `apply_accumulated_gradient` call)

Update, using `g = accum / batch_size` exactly where momentum/plain SGD plug in the averaged
batch gradient:

```
t += 1
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
w -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

applied independently to every weight and to the bias.

**Why a per-node local `t` needs no shared/global counter**: every trainable node gets exactly
one `apply_accumulated_gradient` call per `learn()`/`learn_batch()` invocation - the network
never calls it for some nodes and not others on a given iteration - so a per-node counter stays
numerically identical to a hypothetical global one, the same reasoning that already lets
momentum's velocity live per-node with no shared state between nodes or layers.

## hyperparameters

`learning_rate` stays the normal, explicit, per-call training-time argument, unchanged from every
other sibling. `beta1`/`beta2`/`epsilon` default to Kingma & Ba's own published values (`0.9`,
`0.999`, `1e-8`) rather than requiring them explicitly the way `momentum`/`l2_lambda` are - those
two were made required specifically because no coefficient this codebase measured was safe to
recommend as a default, whereas beta/epsilon are closer to fixed algorithmic constants in
virtually all real-world Adam usage, not tunable knobs this project has an opinion on.
