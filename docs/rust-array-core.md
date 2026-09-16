# the Rust array core

[← back to vectorization](vectorization.md)

`rust/perceptron_array/` is a standalone PyO3 crate (`pyo3` as the only dependency, ~740 lines
across `array.rs`/`ops.rs`/`linalg.rs`/`ufuncs.rs`/`random.rs`/`mnist.rs`/`lib.rs`) implementing
every operation in [the numpy interface subset](numpy-interface-subset.md)'s table: `Array`
construction/shape/`.T`/slicing/`.copy()`/`.reshape()`/`.tolist()`, single-element read/write at
both the 1D scalar-index and 2D tuple-index shapes, elementwise `+ - * /` with both scoped
broadcasting cases and scalar operands, `__iadd__`/`__isub__`, `exp`, `__matmul__` (all three
shape combinations), `outer`, `sum_axis0`, `argmax`, a hand-rolled xorshift128+ `uniform`, and
`decode_mnist_pixels`. It's parity-tested against real numpy (and, where a pure-Python reference
exists independent of numpy, against that too - a three-way match) across 525 tests in
`rust/perceptron_array/tests/` (254 covering the operation subset itself, plus 271 covering
`fused.rs`'s per-layer functions below). The one documented exception is `uniform`: a hand-rolled
PRNG can never reproduce numpy's Mersenne Twister bit-for-bit, so its tests check statistical
plausibility (range, mean, variance), not per-draw equality - see `random.rs`'s own doc comment.

Also includes `fused.rs`: one Rust function per `ArrayLayer` method (`layer_forward`,
`layer_output_delta`, `layer_accumulate_gradient_batch`, etc. - see
[the production cutover plan](rust-production-cutover.md#0b-fuse-each-layer-operation-into-one-rust-call)
for the full list), doing an entire layer computation in a single Python-to-Rust call instead of
composing it from several `Array` operator calls - added specifically to cut per-call FFI
overhead once that turned out to matter (see "status" below).

**Status: built, parity-tested, and wired into production.** This core backs
`RustArrayLayer`/`RustArrayMultiClassBackpropClassifierNetwork`
(`perceptron/model/rust_array_layer.py`,
`perceptron/model/rust_array_multiclass_backprop_classifier_network.py`) - see
[structure](structure.md#vectorized-array-based-classes) for the numpy-backed classes it replaced
as production (numpy stays on permanently as the benchmarking mirror, per
[vectorization](vectorization.md#decision)). [The production cutover plan](rust-production-cutover.md)
covers the full path this took: an initial naive per-op composition really was 65-335x slower
than numpy at this codebase's real layer sizes - but that number turned out to be measured against
an accidental **debug build** (`./cli build-rust` was missing `--release`); on a release build,
the real gap was already only ~3-10x, closed further by a matmul loop reorder and the fused
functions above to ~2-7x for a synthetic forward pass. On this codebase's actual training paths
(`learn()`'s per-example, `batch_size=1` shape - both existing production demos use this
exclusively, not `learn_batch`), the fused Rust core measures as a genuine, real-training-run win:
**3.40x faster at UCI digits scale, 1.31x faster at real MNIST scale** (see [research and
analysis](research-and-analysis.md#phase-2-tier-2-real-per-example-training-is-a-genuine-win-at-both-scales-measured)),
with statistically indistinguishable accuracy. The crate's matmul is still a naive triple loop, so
it still loses to numpy's BLAS at larger mini-batch sizes (`batch_size >= 32`) - tracked as
follow-on optimization work (SIMD, blocking/tiling, threading), not a blocker for the production
paths that exist today.

## why not just keep using real NumPy

Installing `numpy` would obviously work, and would be the pragmatic choice for anyone whose only
goal is faster training. This crate exists because this repo's own identity treats even a "just
fast math" dependency as a deliberate choice, not a default - the same posture that's kept
scikit-learn as a one-time, offline, non-runtime extraction tool (`digits_data.py`) rather than a
real dependency, and pyarrow as a one-time conversion step
(`mnist_data.convert_parquet_to_binary`) rather than something training itself ever imports.

## why Rust (via PyO3/maturin), not C

Two implementation languages were compared for a from-scratch array core covering exactly
[the numpy interface subset](numpy-interface-subset.md)'s own table (deliberately not
general-purpose NumPy): C wrapped via `ctypes` or the raw CPython C-API, and Rust wrapped via
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs). Rust came out ahead primarily on the
Python-binding layer, not the numerical code itself (the actual math - matmul, elementwise ops,
reductions - is the same algorithmic work in either language): PyO3's
`#[pyclass]`/`#[pyfunction]`/`#[pymodule]` macros generate the reference-counting and
GIL-handling boilerplate automatically, eliminating the single riskiest, hardest-to-debug part of
a raw CPython C-API extension (manual reference-counting bugs), and `maturin` absorbs the
packaging/build step a C extension needs a separate `setup.py`/Makefile for. Real,
production-proven precedent exists for exactly this pattern (`polars`, `ruff`, `orjson`,
`cryptography`'s core are all Rust-behind-Python-bindings, not experiments).

`Cargo.toml` deliberately depends on `pyo3` alone - no `ndarray`, no `rand` crate, matching this
repo's own "hand-build everything" posture: pulling in Rust crates for convenience would just
move the same "adopt vs. hand-build" tension NumPy itself raises down one level, not resolve it.

## scope

Fixed `f64` dtype, up to 2D (matrix) + 1D (vector), row-major contiguous storage - exactly
[the numpy interface subset](numpy-interface-subset.md)'s own table, nothing more. Not targeting
BLAS-competitive matmul performance (naive triple loop, no SIMD/blocking) - that's decades of
industry tuning, out of scope regardless of language; the realistic bar is "meaningfully faster
than pure Python," not matching OpenBLAS.

| file | contents |
|---|---|
| `src/array.rs` | the core type: a flat `Vec<f64>` buffer plus a `shape: (usize, usize)`, with `.copy()`/`.reshape()`/slicing/transpose and single-element read/write - `__setitem__` dispatches on *both* a bare integer index (1D) and a `(row, col)` tuple index (2D) |
| `src/ops.rs` | elementwise `+ - * /`, restricted to exactly the two broadcasting cases the interface subset names (vector+vector, matrix+row-vector), plus scalar operands on either side |
| `src/linalg.rs` | matmul (dispatching on the three shape combinations `ArrayLayer`'s formulas actually use: 1D×2D, 2D×1D, 2D×2D) and `outer` |
| `src/ufuncs.rs` | `exp`, `argmax` (1D only, numpy's own first-occurrence tie-break), `sum_axis0` (the one fixed-axis reduction the interface subset requires) |
| `src/random.rs` | a hand-rolled xorshift128+ PRNG plus `uniform(low, high, shape)` - see "the RNG exception" below |
| `src/mnist.rs` | raw `u8` buffer -> `f64` array decode (the `/255.0` rescale, reshape, and pixel/label slice), matching `load_mnist_dataset_as_array`'s output |
| `src/lib.rs` | the `#[pymodule]` entry point, exposing an `Array` `#[pyclass]` with `#[pymethods]` for the operations above, plus free `#[pyfunction]`s for the standalone ones |

## the RNG exception

`uniform` is the one operation in the whole subset where "parity" cannot mean bit-identical
output against numpy: a hand-rolled generator can never reproduce numpy's Mersenne Twister
bit-for-bit, seeded or not. That has a real, load-bearing consequence:
[the vectorized classes](structure.md#vectorized-array-based-classes)' own "identical accuracy
trajectory" claim, and every other numerical-parity gate in this codebase (mini-batch's
batch-size-1 parity check, this crate's own three-way parity tests), depend on bit-identical
*initial* weights, which `randomize()`'s RNG call supplies. Swapping the array backend to this
core, once wired in, would still change the exact trained weights from a fixed seed even if every
other operation matches numpy exactly - not a bug, but a fact worth stating before anyone is
surprised by it.

## numerical parity validation

For every operation in [the numpy interface subset](numpy-interface-subset.md)'s table: a
randomized-input test comparing the Rust result against **both** real `numpy`'s own result and
the existing pure-Python reference implementation
(`BackpropNode.z()`/`sigmoid()`/[the vectorized classes](structure.md#vectorized-array-based-classes)'
own already-numpy-parity-checked methods) across a large random-input sweep - checking against
numpy transitively re-validates against the pure-Python reference those classes were already
checked against, so a three-way match is strictly stronger evidence than a two-way one. Two risk
areas got particular attention: whether Rust's `f64::exp` saturates to infinity for large
arguments the same way numpy's `np.exp` does (both should per IEEE 754, but checked directly
rather than assumed), and float64 summation-order rounding in `matmul`/`outer` (the same category
of issue that got an earlier piecewise-sigmoid change reverted in this codebase - see
[structure](structure.md#backprop)). The uniform RNG fill is the one named exception - see above.

## expected vs. measured performance

A throwaway numpy benchmark (numpy happens to be installed on this machine; not a repo dependency
and not proposed as one here - used purely as a real proxy for "what a compiled, vectorized
implementation could achieve") at this codebase's actual architecture (`dimension=784, hidden=16,
output=10`) measured 150.7x forward-pass speedup over pure Python - treated as a ceiling, not a
target, since this crate's naive (no-SIMD, no-blocking) matmul should land meaningfully below it.
[Vectorization](vectorization.md#expected-effect) records the reasoning for treating
15-45x as a realistic, still practically significant win, and the real (not extrapolated) 9.00x/
33.87x numpy measurements this core's own eventual production numbers would be compared against.

## what was out of scope for this crate's own build (now resolved elsewhere)

Two items this crate's own PR-staged build deliberately left alone, since resolved by
[the production cutover plan](rust-production-cutover.md) as separate, later work rather than by
changing this crate's own scope: **retargeting production code paths to this core** (done -
`perceptron/model/rust_array_layer.py`/`rust_array_multiclass_backprop_classifier_network.py` are
a new call path alongside the existing numpy one, not an import swap that removes it) and **any
change to `perceptron/model/`** (the fused per-layer functions above did require crate changes,
but nothing in the existing pure-Python or numpy-backed classes was touched).

## what stays explicitly out of scope

- **BLAS-competitive matmul performance.** Naive-but-correct; any tuning (SIMD, blocking/tiling,
  threading) is separately-measured follow-up work - see
  [structure](structure.md#possible-next-steps).
- **Any operation outside [the numpy interface subset](numpy-interface-subset.md)'s own table.**
  No axis-parameterized reductions, no `where`/`maximum`/`clip`, no general N-d arrays or
  broadcasting - see that document's own "explicitly not required" for what a future
  ReLU/softmax/momentum/L2 vectorized variant would need to add here first.
- **Any change to the pure-Python or numpy-backed model classes.** `MultiClassBackpropClassifierNetwork`,
  `ArrayLayer`, and `VectorizedMultiClassBackpropClassifierNetwork` stay exactly as they are,
  permanently - the Rust-backed classes are new siblings, not replacements or retrofits.
