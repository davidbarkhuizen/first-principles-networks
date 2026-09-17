# an array-based (Rust-matmul-backed) dropout sibling

[← back to README](../README.md)

Design reference for `DropoutArrayLayer`/`DropoutVectorizedMultiClassBackpropClassifierNetwork`
(numpy) and `DropoutRustArrayLayer`/`DropoutRustArrayMultiClassBackpropClassifierNetwork`
(Rust-matmul-backed) - array siblings of `DropoutBackpropClassifierNetwork` (see
[dropout](dropout.md)). Built and measured; see [research and
analysis](research-backprop-siblings.md#an-array-based-dropout-sibling-another-reconfirmed-null-now-at-real-power)
for the wall-clock/overfitting-headroom measurement and decision.

## design: the one real exception to "purely additive" carries over, in array form

[Dropout's own per-node design](dropout.md#design-a-real-minimal-exception-to-purely-additive-not-glossed-over)
already flagged that dropout needs a call-scoped train/eval-mode distinction no prior sibling
needed - that requirement doesn't go away at the array level, and the same two subtleties that
design document found by testing, not by inspection, needed re-deriving here rather than assumed to
carry over unchanged.

`DropoutArrayLayer(ArrayLayer)` adds a `training: bool` attribute (default `False`, the same
safe-failure-mode default the per-node design chose) and overrides `forward`/`forward_batch` to
draw and apply an inverted-dropout mask, plus `compute_hidden_delta`/`compute_hidden_delta_batch`
to apply that same mask to the backward pass:

```python
class DropoutArrayLayer(ArrayLayer):
    def __init__(self, size: int, input_size: int, drop_probability: float) -> None:
        super().__init__(size, input_size)
        assert 0.0 <= drop_probability < 1.0
        self._drop_probability = drop_probability
        self._keep_probability = 1.0 - drop_probability
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        base = sigmoid(self.z)
        if self.training:
            self._mask = (np.random.random(self.size) >= self._drop_probability).astype(np.float64)
            self.a = base * self._mask / self._keep_probability
        else:
            self._mask = np.ones(self.size)
            self.a = base
        self._base_activation = base           # needed by compute_hidden_delta, mirroring
        self._was_training = self.training     # DropoutNode's own two forward-time snapshots
        return self.a

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.W.T @ next_layer.delta
        sigmoid_derivative = self._base_activation * (1.0 - self._base_activation)
        scale = (self._mask / self._keep_probability) if self._was_training else 1.0
        self.delta = downstream * sigmoid_derivative * scale
```

`forward_batch`/`compute_hidden_delta_batch` mirror this with one independent mask row per
example (`np.random.random((batch_size, self.size))`), matching the per-node design's own
per-example stochastic draw exactly rather than one shared mask for the whole batch - dropout's
whole point is a fresh, independent draw per forward pass, and a batched forward pass is still
`batch_size` independent forward passes from dropout's perspective.

Both subtleties [the per-node design](dropout.md#design-a-real-minimal-exception-to-purely-additive-not-glossed-over)
found by testing carry over unchanged: the backward pass must use `_base_activation` (pre-mask
sigmoid), not `self.a` (post-mask), for the derivative term, and it must read a forward-time
snapshot of `training`/the mask (`_was_training`, `_mask`), not the live attribute, since
`set_training_mode`-equivalent brackets on the array network toggle `training` back to `False`
before backward runs. A `DropoutArrayLayer.set_training_mode(training)` method mirrors
`DropoutLayer`'s own, called from a `VectorizedMultiClassBackpropClassifierNetwork.learn`/
`learn_batch`-level try/finally bracket the same shape the per-node `learn()` uses.

`DropoutRustArrayLayer(RustArrayLayer)` needed a source of per-call randomness in Rust, not just a
fused arithmetic op - a genuinely new category of primitive for [the Rust array
core](rust-array-core.md), which had no RNG surface at all before this (`randomize()`'s own random
draws happen in Python, via `np.random.uniform`, before ever touching the Rust core - see [the
numpy interface subset](numpy-interface-subset.md)'s own `uniform random fill` row): `random.rs`'s
`draw_bernoulli_mask`/`bernoulli_mask` (a hand-rolled xorshift128+ draw, the same generator
`uniform()` already uses, reused rather than a second PRNG built from scratch).

## correctness validation

The one array-ported sibling in this codebase with a genuine RNG-parity caveat, the same category
the array-based Adam sibling already accepted for its own batch-size-robustness measurement, but
sharper here since dropout's mask *is* the mechanism under test, not incidental to it:

- `test_dropout_array_layer.py`: `forward`/`compute_hidden_delta` at `training=False` checked
  directly against `ArrayLayer`'s own plain sigmoid (must match exactly, no rescale) - the array
  analogue of [the per-node design](dropout.md#correctness-validation)'s own inference-mode
  check. At `training=True`, force a known mask via `unittest.mock.patch("numpy.random.random",
  ...)` (the array-world equivalent of the per-node test's own `random.random` patch), then check
  both the forced-kept and forced-dropped cases against the hand-derived chain-rule formula -
  never against a live, unforced numpy RNG draw, since matching an *exact* mask across numpy's and
  Python's own `random` module RNG streams is neither possible nor the property being tested.
- `test_dropout_vectorized_multiclass_backprop_model.py`: whole-network tests via the shared
  `tests/helpers.py` convention - snapshot/restore round-trip (no new snapshot/restore surface:
  `_mask`/`_base_activation`/`training` are per-forward-pass-scoped, the same category
  `_activation`/`delta` already are), plus the same exactly-reproducible-at-inference test
  [dropout's own per-node suite](dropout.md#correctness-validation) added. Not symmetry-breaking:
  no array-backed sibling test file in this codebase has that test (`assert_randomize_breaks_symmetry`
  is shaped around per-node `.nodes`, which array-backed layers don't have) - built instead was a
  genuine cross-implementation parity check no other array sibling's test suite needed to add
  separately, `matching_dropout_array_backprop_networks` (`tests/helpers.py`): dropout is a
  deterministic no-op at eval mode on both sides (training defaults to `False`, and
  `predict_probabilities`/`classify_state` never toggle it), so `predict_probabilities`/
  `classify_state` are checked against a per-node reference
  (`DropoutMultiClassBackpropClassifierNetwork`) across a random sweep, the same shape every
  other array sibling's own eval-mode parity test uses - a `learn()`-step parity check isn't
  attempted, for the same independent-RNG-streams reason noted above.
- `test_dropout_rust_array_layer.py` + `rust/indrajala_ml_array/tests/test_dropout_rng.py` +
  `rust/indrajala_ml_array/tests/test_dropout_fused_layer_ops.py`: the Rust core's RNG primitive,
  parity-tested against numpy's own mask distribution statistically (mean keep-rate over many
  draws within tolerance of `1 - drop_probability`, not a literal per-draw match) - a genuinely
  different validation shape than every other Rust parity test in this codebase, which are all
  bit-close deterministic checks; flagged explicitly rather than silently reusing a
  deterministic-parity test template that doesn't fit.
