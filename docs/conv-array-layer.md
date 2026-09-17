# an array-based (Rust-matmul-backed) convolutional layer

[← back to README](../README.md)

**Status: proposed, not started.** Written up front as a design/measurement plan before any of
it exists, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based
Adam sibling](adam-array-layer.md) for precedent).

## why this, and why now

`ConvMultiClassBackpropClassifierNetwork` (`conv_layer.py`/`conv_kernel.py`/`conv_unit.py`) has
**no array-based or Rust-matmul-backed counterpart at all** - checked directly: no
`array`/`vectorized`/`rust` variant of any conv class exists anywhere in `indrajala_ml/model/` or
this codebase's docs. This is exactly the case [structure](structure.md#possible-next-steps)
already names, and this task's own background context calls out by name: infrastructure that's
inherently expensive and has no vectorized path at all. The UCI-digits-scale validation came back
a flat, honestly-reported null (96.69% dense vs. 96.52% conv, statistically indistinguishable -
[convolutional layers on UCI
digits](research-backprop-siblings.md#convolutional-layers-on-uci-digits)), and the obvious
follow-up - does real MNIST's larger spatial structure (28x28 vs. 8x8) separate the two
differently - has never been run, because a full run costs ~30 minutes per architecture per seed
on the per-node path
([structure](structure.md#possible-next-steps)'s own "possible next steps" entry). That per-run
cost is precisely what every other workplan in this round exists to remove elsewhere in this
codebase; convolution is the one place it was never even attempted.

## scope

`ArrayConvLayer` (numpy) and `RustArrayConvLayer` (Rust-matmul-backed), the same
numpy-first-then-Rust precedent [an array-based Adam sibling](adam-array-layer.md) established,
paired with a new `ArrayConvMultiClassBackpropClassifierNetwork` mirroring
`ConvMultiClassBackpropClassifierNetwork`'s own composition (one conv layer directly after the
input, followed by ordinary dense hidden/output layers built from `ArrayLayer`). Same v1
constraints [convolutional layers](convolutional-layers.md) already scoped for the per-node
version: single input channel, `'valid'` padding only, no stacking - unchanged here, not
reopened.

## design: the actual new work is a windowing primitive, not a matmul port

Every other sibling in this round ([momentum](momentum-array-layer.md),
[L2](l2-array-layer.md), [ReLU](relu-array-layer.md), [softmax](softmax-array-layer.md),
[dropout](dropout-array-layer.md)) is a variant *layered onto* the array world's existing
matmul-shaped pipeline - a different formula applied to the same `W @ x + b` shape `ArrayLayer`
already computes. Convolution is different: its core operation (each output position reading its
own local receptive-field window of the input) has never been expressed as a matmul in this
codebase at all, on any path. The standard vectorization technique ("im2col") makes it one: stack
every receptive-field window into a matrix, so convolution becomes one matmul against the
stacked kernel weights.

```python
def _extract_patches(x: np.ndarray, input_height, input_width, kernel_size, stride) -> np.ndarray:
    # shape (out_height * out_width, kernel_size * kernel_size) - one row per output spatial
    # position, row-major (row * out_width + col), matching ConvLayer._receptive_field_nodes'
    # own row-major flat-index convention exactly
    grid = x.reshape(input_height, input_width)
    out_height = (input_height - kernel_size) // stride + 1
    out_width = (input_width - kernel_size) // stride + 1
    rows = []
    for row in range(out_height):
        for col in range(out_width):
            window = grid[row * stride : row * stride + kernel_size, col * stride : col * stride + kernel_size]
            rows.append(window.reshape(-1))
    return np.stack(rows)   # (out_height*out_width, kernel_size**2)

class ArrayConvLayer:
    def __init__(self, input_height, input_width, kernel_size, channel_count, stride=1) -> None:
        self.input_height, self.input_width = input_height, input_width
        self.kernel_size, self.channel_count, self.stride = kernel_size, channel_count, stride
        self.out_height = (input_height - kernel_size) // stride + 1
        self.out_width = (input_width - kernel_size) // stride + 1
        # K: (channel_count, kernel_size**2) - the array-world analogue of channel_count
        # separate ConvKernel.weights lists, but as one shared array instead of channel_count
        # owned Python lists
        self.K = np.zeros((channel_count, kernel_size * kernel_size))
        self.b = np.zeros(channel_count)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._patches = _extract_patches(x, self.input_height, self.input_width, self.kernel_size, self.stride)
        Z = self._patches @ self.K.T + self.b        # (num_positions, channel_count)
        self.a = np.maximum(0.0, Z)                    # ReLU, the same convention ConvUnit uses
        # channel-major flatten (Z.T, not Z) to match ConvLayer.nodes' own documented ordering:
        # "every (row, col) position for kernel 0, then kernel 1, and so on"
        return self.a.T.reshape(-1)

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        downstream = (next_layer.W.T @ next_layer.delta).reshape(self.channel_count, -1).T
        self.delta = downstream * (self.a > 0.0)      # (num_positions, channel_count)

    def accumulate_gradient(self) -> None:
        # sums every contributing spatial position into the shared kernel gradient in one
        # matmul - the array-world analogue of ConvKernel.accumulate_gradient's own per-position
        # Python-loop sum, composing correctly with mini-batch averaging in
        # apply_accumulated_gradient exactly the way that class's own docstring already requires
        self._grad_K += self.delta.T @ self._patches   # (channel_count, kernel_size**2)
        self._grad_b += self.delta.sum(axis=0)
```

**A real architectural simplification, worth stating plainly**: `ConvUnit`'s own docstring gives
the reason it can't be a `BackpropNode` subclass as "`BackpropNode`'s `input_node_weights` is an
owned, rebindable instance attribute, which fights a weight list many instances need to read and
update identically." That problem is specific to the per-node object graph - `ArrayLayer.W` is
already one shared array read via matmul, not `size` owned-and-rebindable per-node lists, so the
array world has no analogous fight to route around. `ArrayConvLayer` needs no
composition-over-inheritance workaround the way `ConvUnit`/`ConvKernel` did; the weight-sharing
problem that drove that whole design choice simply doesn't arise once weights are already a
shared array.

**The genuinely new primitive**: `_extract_patches` above is written as a Python loop building a
list of slices, deliberately not `np.lib.stride_tricks.sliding_window_view` or fancy/integer-array
indexing - both are explicitly out of [the numpy interface subset](numpy-interface-subset.md#explicitly-not-required)'s
documented scope ("no boolean masks, no integer-array indexing"). This makes the numpy stage's
own "vectorization" partial: the matmul itself is fully vectorized, but window extraction is
still `out_height * out_width` Python-level slice operations per forward pass - a real, honest
limitation worth measuring rather than assuming away (see "measurement plan" below). Closing it
properly (a true single-call windowing op) is a larger primitive than any other sibling in this
round needs, and `RustArrayConvLayer` needs the Rust core's own equivalent (a strided-window
extraction op, `array_conv_extract_patches`-shaped) built from scratch - [the Rust array
core](rust-array-core.md) has no windowing/gather primitive of any kind today.

## correctness validation

Fully deterministic (fan-in-aware kernel init is a random draw, but the forward/backward math
itself has no RNG):

- `test_conv_array_layer.py`: `_extract_patches`/`forward`/`compute_hidden_delta`/
  `accumulate_gradient` checked directly against `ConvUnit`/`ConvKernel`'s own per-node/per-kernel
  formulas across a random sweep of small inputs (matching `test_conv_unit.py`/
  `test_conv_kernel.py`'s own hand-computed fixtures), at the same `kernel_size=3,
  channel_count=4` shape [convolutional layers on UCI
  digits](research-backprop-siblings.md#convolutional-layers-on-uci-digits) measured, plus a
  distinct `stride>1` case (`ConvUnit`'s own stride handling is exercised by
  `test_conv_layer.py` already - this port must match it, not just the `stride=1` default).
- `test_array_conv_multiclass_backprop_model.py`: whole-network parity against
  `ConvMultiClassBackpropClassifierNetwork` itself directly (a genuine per-node reference already
  exists), via a weight-injection helper mirroring `matching_array_backprop_networks`'s own
  pattern, extended to seed both the conv layer's kernel weights and the dense tail's weights
  identically on both sides.
- Once the Rust primitive exists: `rust/indrajala_ml_array/tests/test_array_conv_extract_patches.py`
  parity-tested against the numpy `_extract_patches` reference directly, then the same two
  model-level tiers against `RustArrayConvLayer`.

## measurement plan

- **Does the numpy stage's partial vectorization actually help, given the Python-loop window
  extraction** - measured directly before assuming it does: one training step's wall-clock,
  per-node vs. numpy (partial-vectorization) vs. Rust (once built), at UCI-digits shape
  (`kernel_size=3, channel_count=4` on 8x8 input) and real-MNIST shape (28x28 input, a
  comparable kernel/channel budget). If the numpy stage's own win is small (plausible, given the
  window-extraction loop), that's real information about whether the Rust primitive is worth
  building at all, not assumed necessary up front. This is a deliberate exception to [goals and
  strategy](goals-and-strategy.md#measurement-discipline-the-per-node-paths-two-jobs-and-the-one-it-doesnt-have)'s
  own "don't re-time per-node, the order of magnitude is already established" default: that
  default holds for every other sibling because a whole-array numpy port's win over per-node is
  already known and reconfirmed five times over, but conv's numpy stage is only *partially*
  vectorized (the window-extraction loop stays Python-level), so whether it beats per-node at all
  is a genuinely open question here, not a foregone one - no existing figure to cite instead.
- **Reproduce the UCI-digits null, honestly**: confirm the array/Rust port's own dense-vs-conv
  comparison at the same comparable-parameter-budget setup lands within seed noise of the
  existing 96.69%/96.52% result, before trusting any new-scale finding built on top of it.
- **The real-MNIST-scale validation this null result explicitly left open**: once wall-clock is
  confirmed practical, run the same dense-vs-conv comparison
  [structure](structure.md#possible-next-steps) already named as the natural next step, at real
  MNIST's 28x28 scale - the actual point of this whole workplan.

## risks and open questions

- **The numpy stage may not be a large win** - flagged under "design"/"measurement plan" above;
  unlike every other sibling in this round, this one's numpy stage is only partially vectorized,
  and that's measured, not assumed away.
- **A genuinely new Rust primitive, larger in scope than any other sibling's** - a windowing/gather
  operation, not an elementwise or reduction extension; may be a substantial enough lift that the
  Rust stage is only worth doing once the real-MNIST measurement (achievable on the numpy stage
  alone, if its own win is real even if partial) has already shown convolution is worth
  investing further in - the same conditional-escalation discipline [dropout's own
  workplan](dropout.md#measurement-plan) used for its stage 2.
- **A reconfirmed null at real-MNIST scale is a legitimate outcome** - this workplan exists to
  finally get the answer affordably, not to guarantee convolution wins at the larger scale.
- **Stacked conv layers, multi-channel input, pooling** - out of scope here exactly as they are
  for the per-node version ([structure](structure.md#convolutional-layer)); not reopened.

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. `ArrayConvLayer` (numpy, Python-loop window extraction) +
   `ArrayConvMultiClassBackpropClassifierNetwork` + the parity-check tests above.
3. The wall-clock measurement (per-node vs. numpy, both scales) - deciding whether the Rust
   primitive is worth building at all (see "risks" above).
4. The UCI-digits reproduction and the real-MNIST-scale dense-vs-conv validation - the actual
   point of this workplan.
5. Docs closeout: this document and `structure.md`'s possible-next-steps entry updated to reflect
   the actual result.
6. **Conditional** on stage 3's finding: the Rust core's windowing primitive +
   `RustArrayConvLayer` + `RustArrayConvMultiClassBackpropClassifierNetwork` + parity-check tests
   at every tier.
