# convolutional layer, from scratch

[← back to README](../../README.md)

`ConvKernel`/`ConvUnit`/`ConvLayer`/`ConvMultiClassBackpropClassifierNetwork` add a from-scratch
convolutional layer to this codebase - see [structure](../project/structure.md#convolutional-layer) for the
short version and current architecture. This document is the fuller design reference: why a
conv layer needed a genuinely new abstraction (not just a new per-node formula, unlike every
prior sibling class), and the measured result that answered whether it's worth using.

## why this is next

This codebase's only two real datasets - the bundled UCI digits (8x8, `digits_data.py`) and real
MNIST (28x28, `mnist_data.py`) - are both images, classified with dense layers alone: every node
in a `BackpropLayer` connects to *every* node in `input_layer`, with fully independent weights
per node. That throws away the one structural fact these two datasets have that generic tabular
data doesn't: pixels have 2D spatial relationships, and the same local pattern (an edge, a
stroke) can appear anywhere in the image. A convolutional layer encodes that directly - local
receptive fields instead of full connectivity, and one shared, position-independent weight
kernel per feature instead of one independent weight set per node.

## the architectural break: weight sharing

Every prior sibling layer in this codebase - `ReLULayer`, `SoftmaxOutputLayer`,
`CrossEntropyOutputLayer`, `MomentumLayer`/`L2Layer` - changes only a per-node *formula*, while
leaving two things fixed: every node in a layer connects to the *same* full `input_layer.nodes`
list, and every node owns its *own*, independent `input_node_weights`. Convolution breaks both:

- **Local receptive fields.** A convolutional unit at output position `(row, col)` connects only
  to a `kernel_size x kernel_size` window of the input, not the whole layer.
- **Weight sharing.** Every spatial position within one output channel uses the *identical*
  kernel weights and bias - the literal same trainable parameters, updated once per step from
  gradient contributions summed across every position that used them.

Weight sharing is the deeper break. `BackpropNode`/`WeightedInputNode` conflate "one spatial
computation" with "one independently-owned weight set" - `update_input_weights` rebinds
`self.input_node_weights` to a new list on that one instance, which can't express "this
position's weights are literally the same object every other position in this channel reads and
updates." The fix: split "one spatial position's forward/backward math" (`ConvUnit`) from "one
channel's shared, trainable kernel" (`ConvKernel`), where many `ConvUnit`s reference one
`ConvKernel` and only the kernel accumulates and applies gradient.

That reintroduces, in a new place, the same accumulate/apply seam
[mini-batch gradient descent](mini-batch-gradient-descent.md) built for a different reason - a
kernel's gradient is the *sum* of `delta * input_value` across every spatial position that used
it this step. But `BackpropNetworkBase`'s network-level orchestration (`_accumulate_gradients`,
`_apply_accumulated_gradients`, `_apply_gradients`, `snapshot`, `restore`) all iterated
`for node in layer.nodes` directly, one apply call per node - for a `ConvLayer`, `layer.nodes` is
every spatial position, so calling `apply_accumulated_gradient` once per *position* would apply
the same kernel's update up to `out_height * out_width` times per step. These five methods now
delegate through a per-*layer* hook instead: `BackpropLayer` gained default implementations that
are the exact loop bodies these methods used to have (a pure extraction, zero behavior change for
every existing layer type), and `ConvLayer` overrides them to iterate `ConvUnit`s for accumulate
(summing every position's contribution into its kernel is the desired behavior) but `ConvKernel`s
(once each) for apply/snapshot/restore. This was the one real
`BackpropNetworkBase`/`BackpropLayer` change this addition needed - every previous sibling needed
zero.

## scope: one conv layer, directly after the input

Stacking multiple convolutional layers needs backprop-through-convolution (computing a gradient
with respect to the *input* of a conv layer, not just its weights) - real, standard CNN math with
no existing analogue anywhere in this codebase. A single conv layer placed directly after the
(non-trainable) input `StateLayer` needs none of that: nothing before it is ever trained, so
there's nothing to backpropagate *into*. `_backward_hidden_layers`'s existing formula -
`compute_hidden_delta(next_layer_nodes, own_index)` - already works unchanged for a conv layer's
own delta, computed from a downstream *dense* layer's existing hidden-delta formula, because a
dense layer downstream of a (flattened) conv layer is just an ordinary fully-connected layer as
far as that formula is concerned. Scope is deliberately narrowed to exactly this case: one
`ConvLayer`, single input channel, `'valid'` padding (no synthetic zero-padding nodes), a
configurable stride, followed by ordinary dense hidden and output layers.

## design

| concept | design |
|---|---|
| `ConvKernel` (one per output channel) | owns `weights: list[float]` (flat, length `kernel_size**2`) and `bias: float`, plus its own `accumulate_gradient(delta, receptive_field_values)`/`apply_accumulated_gradient(learning_rate, batch_size)` - the same shape as `BackpropNode`'s own pair, just invoked once per contributing spatial position and applied once per kernel. Fan-in-aware init: `limit = 1/sqrt(kernel_size**2 * in_channels)` - fan-in for a kernel is exactly its own receptive field size. |
| `ConvUnit` (one per output spatial position per channel) | composition, not inheritance (`AbstractNode`, not `BackpropNode`) - a genuinely shared, cross-instance weight list doesn't fit `BackpropNode.input_node_weights`'s owned-and-rebindable design. `forward()`: `z = sum(input_nodes[i].value() * kernel.weights[i] ...) + kernel.bias`, then ReLU. |
| receptive-field wiring | `ConvLayer.__init__` computes, for each output `(row, col)`, the flat input indices `[(row*stride + kr) * input_width + (col*stride + kc) for kr in range(kernel_size) for kc in range(kernel_size)]` into `input_layer.nodes` - row-major flat indexing, matching how `mnist_data.py`/`digits_data.py` already decode pixels. |
| `ConvLayer` | holds `channel_count` `ConvKernel`s and `out_height * out_width * channel_count` `ConvUnit`s as its `.nodes` (channel-major order); `accumulate_gradients()` loops every `ConvUnit`; `apply_accumulated_gradients()`/`snapshot()`/`restore()` loop every `ConvKernel` once each. |
| `ConvMultiClassBackpropClassifierNetwork` | a sibling of `MultiClassBackpropClassifierNetwork` (not calling `super().__init__()` - builds `hidden_layers` = `[conv_layer] + dense_layers`, `output_layer`, `trainable_layers` directly). `learn`/`learn_batch`/`_backward`/`classify_state`/`predict_probabilities`/`randomized` are inherited unchanged. Its own `save`/`load` envelope - `save_model_json`'s `layer_sizes: list[int]` has no way to express conv hyperparameters. |

## numerical and behavioral risks

- **Gradient correctness needs numerical gradient checking, not just a hand-derived example**:
  perturbing one kernel weight by a small `epsilon` and comparing the resulting loss change to
  `accumulate_gradient`'s analytic value - the rigorous check for a genuinely new backward-pass
  formula, beyond a hand-computed sanity case.
- **Row-major flat-index assumption**, tested against a small hand-constructed image with a
  single hot pixel at a known `(row, col)` - true for both datasets' own conventions, but worth
  confirming directly rather than assumed.
- **Whether conv actually helps is an open empirical question**, not assumed - UCI digits at 8x8
  with the existing dense baseline already reaches 99.5%/96.9% train/test, a small, easy task
  that may already be close to saturated, leaving little room for conv to show a clear win at
  this scale.

## measured result: a clean null at UCI-digits scale

At a comparable trainable-parameter budget (2410 dense vs. 2530 conv), 8 seeds each, mean test
accuracy came out statistically indistinguishable (96.69% dense vs. 96.52% conv, well within each
other's stdev) - no measured win for convolution at this scale, honestly reported rather than
stretched into one. See
[research and analysis](../research/research-backprop-siblings.md#convolutional-layers-on-uci-digits) for the full
writeup. `ConvMultiClassBackpropClassifierNetwork` remains a real, correct, tested capability;
whether a real-MNIST-scale run (more spatial structure for convolution's advantages to
potentially show up in, at real wall-clock cost) is worth running is an open item - see
[structure](../project/structure.md#possible-next-steps).

## what stays explicitly out of scope

- **Stacked convolutional layers.** Needs backprop-through-convolution - deliberately avoided by
  scoping to exactly one conv layer, directly after the non-trainable input.
- **Multi-channel input.** Only matters once conv layers stack - deferred alongside stacking.
- **`'same'` padding.** Needs either synthetic zero-padding nodes or special-cased edge-position
  indexing; `'valid'` padding is simpler and sufficient to prove the core mechanism.
- **Pooling (max or average).** Not needed to prove the core mechanism - a `stride > 1` conv
  achieves comparable downsampling without a new layer type or a max-routing backward pass.
- **Any change to `LinearClassifierNetwork`/`AssociationNode`, or to any existing `BackpropLayer`
  sibling's behavior**, beyond the five per-layer hook methods described above.
