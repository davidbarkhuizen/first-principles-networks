# implementing the required subset in Python-wrapped Rust

[← back to vectorization](vectorization.md)

## status: built and parity-tested; not yet wired into production

PR 0 through PR 9 of the workplan below are built: `rust/perceptron_array/` (a standalone PyO3
crate, `pyo3` as the only dependency, ~740 lines across `array.rs`/`ops.rs`/`linalg.rs`/
`ufuncs.rs`/`random.rs`/`mnist.rs`/`lib.rs`) implements every operation in
[the numpy interface subset](numpy-interface-subset.md)'s table - `Array` construction/shape/
`.T`/slicing/`.copy()`/`.reshape()`/`.tolist()`, single-element read/write at both the 1D
scalar-index and 2D tuple-index shapes, elementwise `+ - * /` with both scoped broadcasting cases
and scalar operands, `__iadd__`/`__isub__`, `exp`, `__matmul__` (all three shape combinations),
`outer`, `sum_axis0`, `argmax`, a hand-rolled xorshift128+ `uniform`, and `decode_mnist_pixels` -
across 10 merged PRs (#174-#183), one per workplan stage, each with its own parity tests before
merging. 254 tests in `rust/perceptron_array/tests/`, checked against real numpy (and, where a
pure-Python reference exists independent of numpy itself, against that too - a three-way match)
at every stage, plus PR 9's own consolidated sweep covering the whole subset in one file. The one
documented exception is `uniform`: a hand-rolled PRNG can never reproduce numpy's Mersenne
Twister bit-for-bit, so its own tests check statistical plausibility (range, mean, variance), not
per-draw equality - see `random.rs`'s own doc comment. All 311 pre-existing pure-Python tests
still pass unchanged throughout - purely additive, nothing in `perceptron/` touched.

**What isn't done**: per "what stays explicitly out of scope" below, retargeting
[vectorized array-based model classes](vectorized-array-classes.md)'s own classes from `numpy` to
this core - the actual production cutover - is a separate, later step this workplan deliberately
doesn't include. This core exists and is proven correct; nothing yet calls it from
`perceptron/model/`. The rest of this document is kept as written before any of this was built -
**only this status section reflects present-tense reality**.

Part 3 of 3 in the [vectorization](vectorization.md) workplan split: a plan for implementing
[the numpy interface subset](numpy-interface-subset.md)'s exact operation list as a hand-built,
tightly-scoped array core in Rust, wrapped for Python via PyO3/maturin - this codebase's decided
production array backend going forward, per [vectorization](vectorization.md)'s own "decision",
kept alongside real `numpy`'s own now-permanent role as a performance-benchmarking mirror rather
than replacing it outright. Still a workplan, not an implementation - nothing in `perceptron/`
changes as a result of this document itself; the PR-staged plan below is what building it
actually looks like.

## what this replaces, and when

[vectorized array-based model classes](vectorized-array-classes.md) is built and validated
against real `numpy` first, deliberately - see that document's own "why numpy here, now". This
document's array core becomes this codebase's production array backend, per the decision recorded
in [vectorization](vectorization.md#decision-numpy-stays-permanently-as-a-benchmark-mirror-the-rust-core-becomes-production) -
not a replacement for the numpy-backed classes, a parallel implementation kept permanently
alongside them. Once built and proven correct against
[the numpy interface subset](numpy-interface-subset.md)'s exact contract, production code paths
(training runs, demos) route through this core instead of `numpy`;
`vectorized-array-classes.md`'s numpy-backed classes stay in the codebase indefinitely as the
benchmarking mirror this core's own future performance claims get measured against, not
deprecated once this core exists - see "what stays explicitly out of scope" below for why the
actual production cutover still isn't part of *this* document's own workplan.

## why not just keep using real NumPy

Installing `numpy` would obviously work, and would be the pragmatic choice for anyone whose only
goal is faster training. This document exists because this repo's own identity treats even a
"just fast math" dependency as a deliberate choice, not a default - the same posture that's kept
scikit-learn as a one-time, offline, non-runtime extraction tool (`digits_data.py`) rather than a
real dependency, and pyarrow as a one-time conversion step
(`mnist_data.convert_parquet_to_binary`) rather than something training itself ever imports. As of
[vectorization](vectorization.md)'s own recorded decision, plain `numpy` staying permanently
*isn't* the call made for production - `vectorized-array-classes.md`'s classes stay on it
permanently, but scoped to a benchmarking-mirror role, not production; this document's core is
what becomes production instead. This section's original framing (moot if a future maintainer
picks plain numpy for everything) no longer applies to *that* choice, though it would still apply
if a future maintainer later decided to walk back the production half of that decision.

## decision: a hand-built Rust core (via PyO3/maturin), not C, not real NumPy

Two implementation languages were compared for a from-scratch array core covering exactly
[the numpy interface subset](numpy-interface-subset.md)'s own table (deliberately not
general-purpose NumPy): C wrapped via `ctypes` or the raw CPython C-API, and Rust wrapped via
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs). Rust came out ahead primarily on the
Python-binding layer, not the numerical code itself (the actual math - matmul, elementwise ops,
reductions - is the same algorithmic work in either language): PyO3's
`#[pyclass]`/`#[pyfunction]`/`#[pymodule]` macros generate the reference-counting and
GIL-handling boilerplate automatically, eliminating the single riskiest, hardest-to-debug part of
a raw CPython C-API extension (manual reference-counting bugs - dangling references,
double-frees, refcount leaks), and `maturin` absorbs the packaging/build step a C extension needs
a separate `setup.py`/Makefile for. Real, production-proven precedent exists for exactly this
pattern (`polars`, `ruff`, `orjson`, `cryptography`'s core are all Rust-behind-Python-bindings,
not experiments).

PyO3/Rust is the toolchain this workplan targets; real NumPy remains the pragmatic default this
document is deliberately not recommending over it.

## expected performance

Measured, not assumed - a throwaway benchmark (`numpy` happens to be installed on this machine;
not a repo dependency and not proposed as one here - used purely as a real proxy for "what a
compiled, vectorized implementation could achieve," since it's the closest thing to this
workplan's target already available to measure against) at this codebase's actual architecture
(`dimension=784`, `hidden=16`, `output=10`, matching
`demo_mnist_ensemble_recognition.py`/the softmax investigation). Correctness was checked before
trusting the timing: the pure-Python and vectorized forward passes were run on identical
weights/input and compared (max difference 5.27e-16 - floating-point noise, not a real
discrepancy).

| | time |
|---|---|
| pure Python, per-example forward pass | 1328.1 us |
| vectorized (numpy), per-example forward pass | 8.8 us |
| **speedup** | **150.7x** |
| vectorized, batched (batch_size=8) | 4.74 us/example (amortized) |
| vectorized, batched (batch_size=32) | 3.57 us/example (amortized) |
| vectorized, batched (batch_size=128) | 2.44 us/example (amortized) |

This measures only the forward pass - backward-pass delta computation and gradient application
weren't separately benchmarked here, but are algorithmically the same shape of work
(matrix-vector products, one outer product), so should see a comparable-order speedup; worth its
own direct measurement once/if this work is pursued, not assumed to transfer unchecked.

### relative: numpy is a ceiling, not the actual target

`numpy`'s array operations are backed by OpenBLAS - SIMD instructions and cache-blocked matrix
multiplication, decades of industry tuning. The workplan below deliberately starts with a
**naive** Rust matmul (no SIMD, no blocking) - see "what stays explicitly out of scope." A naive
implementation should be expected to land meaningfully below this 150x ceiling, not match it;
even capturing a fraction of it - 15-45x, say, at 10-30% of the measured numpy speedup - would
still be a large, practically significant win over the current pure-Python baseline. A real PyO3
extension also carries its own Python<->Rust call-marshaling overhead per call, not present in
this pure-Python-vs-numpy proxy, which would eat into the measured ceiling further - another
reason to treat 150x as an upper bound to measure against once a real implementation exists, not
a number to plan around as delivered.

### absolute: what this would mean for this codebase's real, already-measured workloads

Extrapolating the per-example numpy figure above against this session's own real measured
full-scale baseline (the isolated 4.03ms/iteration full `learn()` call - forward + backward +
gradient application - measured for `MultiClassBackpropClassifierNetwork` at this exact
architecture, see [research and analysis](research-and-analysis.md)): a 60000-example epoch
currently takes ~4.0 minutes of pure-Python compute. Even a conservative fraction of the measured
150x forward-pass ceiling (say, 20x, allowing generously for FFI overhead and the naive-vs-BLAS
gap) would put the same epoch under 15 seconds - the kind of change that doesn't just make
existing runs faster, but changes what's practical to run at all: the
[MNIST ensemble](demos.md#demo-mnist-ensemble-recognition)'s ~30-minute real training run,
several-seed statistical sweeps like the ones behind every "measured, not worth adopting" finding
in [research and analysis](research-and-analysis.md), and a proper learning-rate-vs-batch-size
sweep like the one the mini-batch momentum retest's own confound surfaced (see
[structure](structure.md#possible-next-steps)), would each plausibly drop from a real logistical
constraint (worth scheduling, worth running in the background while doing other work - precisely
how every real-MNIST measurement in this session's own history was run) to something fast enough
to iterate on interactively instead.

## workplan

### scope

Fixed `f64` dtype, up to 2D (matrix) + 1D (vector), row-major contiguous storage - exactly
[the numpy interface subset](numpy-interface-subset.md)'s own table, nothing more: array
construction (from data, zero-filled), shape/transpose, single-element write (both the 1D
scalar-index and 2D tuple-index shapes - see that document's own "re-checked against the built
implementation" note), matmul, elementwise `+ - * /` with the two scoped broadcasting cases,
in-place accumulate, `exp`, `outer`, a fixed-axis (`axis=0`) row-sum reduction, `argmax`, a
uniform RNG, `.copy()`, raw-byte decode/reshape/slice/cast, and a Python-list round-trip for
serialization.

### 1. crate structure

A new top-level Rust crate (e.g. `rust/perceptron_array/`), kept separate from `perceptron/` and
standalone-buildable until proven correct, not dropped into the existing package tree
prematurely.

- `Cargo.toml`: `pyo3` (with the `extension-module` feature) as the only external crate -
  deliberately no `ndarray`, no `rand` crate, matching this repo's own "hand-build everything"
  posture and answering the same dependency-purity question consistently at this layer too
  (pulling in Rust crates for convenience would just move the same "adopt vs. hand-build" tension
  NumPy itself raises down one level, not resolve it).
- `src/array.rs` - the core type: a plain struct wrapping a flat `Vec<f64>` buffer plus a
  `shape: (usize, usize)` (or an enum distinguishing the 1D/2D cases actually used - no reason to
  build general N-dimensional machinery [the numpy interface subset](numpy-interface-subset.md)
  never needs), with `.copy()`/`.reshape()`/slicing/transpose and single-element read/write -
  `__setitem__` needs to accept *both* a bare integer index (1D) and a `(row, col)` tuple index
  (2D, `target_batch[row, category] = 1.0`) - two argument shapes to dispatch on, not one.
- `src/ops.rs` - elementwise `+ - * /`, restricted to exactly the two broadcasting cases the
  interface subset names (vector+vector, matrix+row-vector), plus scalar operands on either side
  (`learning_rate * grad_W`, `grad_W / batch_size`) - not general N-d broadcasting.
- `src/linalg.rs` - matmul (naive triple loop first; a cache-friendlier loop order as a later,
  separately-measured refinement, not bundled into "get it correct" - matches this session's own
  "measure before assuming" posture, applied to performance instead of ML results) and `outer`.
  Not targeting BLAS-competitive performance - that's decades of industry SIMD/cache-tuning work,
  out of scope regardless of language; the realistic target is "meaningfully faster than pure
  Python."
- `src/ufuncs.rs` - `exp`, `argmax` (no axis parameter needed - see
  [the numpy interface subset](numpy-interface-subset.md)'s own "explicitly not required"), and
  `sum_axis0` (a 2D array's rows summed into a 1D vector - the one fixed-axis reduction that
  document's table does require, added after cross-checking `array_layer.py`'s actual
  `accumulate_gradient_batch`, not assumed from its own original design-doc-only derivation).
- `src/random.rs` - a hand-rolled PRNG (a small, well-known public-domain algorithm - e.g.
  xorshift128+ or PCG32 - implemented directly, not pulled from the `rand` crate) plus a
  uniform-range array-fill function.
- `src/mnist.rs` - raw `u8` buffer -> `f64` array decode (the `/255.0` rescale, reshape, and
  pixel/label slice), a direct port of what `load_mnist_dataset_as_array`
  ([vectorized array-based model classes](vectorized-array-classes.md)) needs.
- `src/lib.rs` - the `#[pymodule]` entry point, exposing an array `#[pyclass]` (a project-specific
  name, avoiding collision with real NumPy's own `PyArray`) with `#[pymethods]` for the operations
  above, plus free `#[pyfunction]`s for the standalone ones (RNG, MNIST decode).

### 2. build integration into this repo's existing tooling

- `cli setup` gains a step installing `rustc`/`cargo` (matching its existing
  `apt install python3-tk` pattern) and `maturin` (via pip, into the same `.venv`).
- `cli setup` (or a new `cli build-rust`) runs `maturin develop` to build the extension into the
  active venv - the Rust-side analogue of `pip install -r requirements.txt`.
- `.gitignore` gains the Rust build output (`rust/*/target/`), matching the existing pattern for
  other regenerable artifacts (`trained_model.json`, `__pycache__`).
- `requirements.txt` stays Python-only; `Cargo.toml` is the Rust-side equivalent, not folded into
  it.

### 3. build order

Each stage independently testable before the next starts - no stage depends on an unvalidated
earlier one - and, per this repo's own "finish one unit of work, PR it, merge it" practice
(see every prior workplan's own PR-per-stage history: mini-batch's #139-#141, conv layers' five
PRs), each stage below is sized to be its own PR, not batched with its neighbors. Tests for a
crate this document's own "crate structure" section keeps standalone-buildable, not yet dropped
into `perceptron/`, live under the crate's own `rust/perceptron_array/tests/` (pytest against the
built extension via `maturin develop`), not the top-level `tests/` directory - that move is part
of the eventual, explicitly out-of-scope "swap the backend" follow-on, not this workplan.

**PR 0 - crate scaffolding and build integration.** `rust/perceptron_array/` with `Cargo.toml`
(`pyo3` only), an empty `#[pymodule]` in `src/lib.rs` that imports successfully, `maturin develop`
wired into `cli setup`/a new `cli build-rust`, and the `.gitignore`/`requirements.txt` changes
from "build integration" above. No array type yet - this stage's only claim is "the toolchain
works end to end," proven by one trivial `#[pyfunction]` (e.g. a `ping() -> str`) importable and
callable from a Python REPL. Nothing to parity-check yet.

**PR 1 - array construction, shape, and single-element read/write.** `src/array.rs`: a
`#[pyclass]` (working name `RustArray`, avoiding collision with real numpy's own `PyArray`) with
`#[new]` from a nested Python list, `#[staticmethod] zeros(shape: (usize, usize))` (a 1D
`zeros(n)` and a 2D `zeros((n, m))` overload - or two named constructors, `zeros_vector`/
`zeros_matrix`, if PyO3 overload dispatch turns out awkward - resolved during implementation, not
before), a `shape` getter, `__getitem__`/`__setitem__` accepting *both* argument shapes flagged
above (bare int for 1D, `(row, col)` tuple for 2D - two branches in one method, not two methods,
matching how Python's own `arr[i]` vs. `arr[i, j]` dispatch through the same `__getitem__`/
`__setitem__` slot), `.copy()`, and `.reshape(shape)`. Test
(`rust/perceptron_array/tests/test_array_basics.py`): construct from a Python list/nested list,
read every element back, write and re-read at both index shapes, copy-then-mutate showing
independence, reshape then re-read - the Python<->Rust round-trip proven before any arithmetic
exists to get wrong.

**PR 2 - `.T`, slicing, and dtype cast.** `.T` (a no-op on a 1D array, a real transpose on 2D -
both cases exercised, not just the 2D one), `arr[:, :-1]` (contiguous slice along one axis -
`start:stop` step-1 slices only, no general Python slice semantics), and `.astype`-equivalent
(trivial for a fixed-`f64`-dtype core except where MNIST's `uint8` source needs the cast - see PR
7). Test: transpose-of-transpose round-trips to the original on both 1D and 2D; slicing against a
hand-constructed small 2D array with a known, distinctive pattern (mirroring
[convolutional layers](convolutional-layers.md#numerical-and-behavioral-risks)'s own
hot-pixel-test discipline for exactly this kind of "is the indexing convention right" risk).

**PR 3 - elementwise `+ - * /`, both broadcasting cases, plus scalar operands.** `src/ops.rs`:
vector+vector and matrix+row-vector for `+`, same-shape and scalar-operand for `- * /`
(`learning_rate * grad_W`, `grad_W / batch_size`), and `__iadd__` for in-place accumulate
(`self._grad_W += ...`). `__isub__` is *not* required to be a true in-place op for correctness -
Python's `a -= b` falls back to `a = a - b` when `__isub__` isn't defined - but implementing it
anyway avoids an unnecessary allocation on every `apply_accumulated_gradient` call, worth doing
here rather than deferred. Test (`test_array_ops.py`): a randomized-input sweep against real numpy
for every operator and both broadcasting cases, including the scalar-operand cases explicitly
(not just same-shape elementwise).

**PR 4 - `exp`.** `src/ufuncs.rs::exp`. The one operation with a documented numerical-parity
trap already flagged twice over (`vectorized-array-classes.md`'s "numerical parity validation",
`array_layer.py`'s own `sigmoid` docstring): does Rust's `f64::exp` saturate to `f64::INFINITY`
for large arguments the same way numpy's `np.exp` does (both should, per IEEE 754, but "should"
is exactly the kind of claim this codebase's own convention is to check, not assume - see
"numerical parity validation" below). Test (`test_ufuncs_exp.py`): the same large-`z`
overflow-boundary sweep `tests/test_array_layer.py` already runs for `sigmoid`, run here directly
against `exp` before `sigmoid` itself is ever built on top of it.

**PR 5 - matmul, `outer`, and `sum_axis0`.** `src/linalg.rs::matmul` (dispatching on operand
shapes - 1D×2D, 2D×1D, 2D×2D - the three cases `ArrayLayer`'s formulas actually use, not a fully
general BLAS-style routine) and `::outer`; `src/ufuncs.rs::sum_axis0` (the corrected addition -
see [the numpy interface subset](numpy-interface-subset.md)'s "re-checked against the built
implementation" note). Test (`test_linalg.py`): a three-way match (Rust core, real numpy, and the
pure-Python `BackpropNode` reference formulas directly) on randomized weights/inputs - strictly
stronger evidence than a two-way Rust-vs-numpy check alone, per "numerical parity validation"
below.

**PR 6 - `argmax`.** `src/ufuncs.rs::argmax` (1D only, ties broken the same way numpy's own
first-occurrence rule does - a real behavioral detail to match, not assume). Test: randomized
sweep plus an explicit tied-maximum case.

**PR 7 - RNG and MNIST byte decode.** Grouped into one PR since both are self-contained,
no-dependency additions on top of everything above, not because they're related to each other.
`src/random.rs` - a hand-rolled PRNG (xorshift128+ or PCG32) plus `uniform(low, high, shape)`.
**This is the one operation in the whole subset where "parity" cannot mean bit-identical output
against numpy**, unlike every other stage here: a hand-rolled generator can never reproduce
numpy's Mersenne Twister bit-for-bit, seeded or not. That has a real, load-bearing consequence
flagged nowhere else in this workplan - [vectorized array-based model classes](vectorized-array-classes.md)'s
own "identical accuracy trajectory" claim and every other workplan's "same seed -> same weights"
regression gate (mini-batch's batch-size-1 parity check, this document's own "numerical parity
validation" below) depend on bit-identical *initial* weights, which `randomize()`'s RNG call
supplies. Swapping the array backend to this core, once it exists, would still change the exact
trained weights from a fixed seed even if every other operation matches numpy exactly - not a
bug, but a fact worth stating before anyone is surprised by it. This stage's own test therefore
checks range bounds and statistical properties (mean/variance within tolerance across a large
`N`) against numpy's `np.random.uniform`, not per-draw equality - a different, weaker bar than
every other stage's three-way exact-match test, called out explicitly rather than silently
applying the wrong bar. `src/mnist.rs::decode` - the `u8` buffer decode/reshape/slice/`/255.0`
cast, tested against `load_mnist_dataset_as_array`'s real output on a small real MNIST sample,
exact match expected here (integer-to-float conversion and division have no RNG-style
irreproducibility).

**PR 8 - the Python-list round-trip.** `.tolist()`-equivalent (1D -> flat list, 2D -> nested
list of lists - two shapes, matching the single-element write split from PR 1) and
construct-from-nested-list (extending PR 1's flat-list constructor to accept nesting). Test:
round-trip a 1D and a 2D array through list-and-back and confirm bit-identical recovery -
straightforward, but still exercised explicitly rather than assumed to fall out of PR 1's
constructor plus `tolist`'s obvious inverse.

**PR 9 - the consolidated numerical parity suite.** Not new functionality - a single test module
(`test_numerical_parity.py`) running every operation above through one large randomized sweep in
one place, the "stage 5" gate the original workplan named but scoped generically; broken out here
as its own PR so it exists as one auditable artifact (all operations, one file) rather than
scattered per-stage checks someone has to reassemble by hand later to answer "has the whole
subset actually been checked."

### 4. numerical parity validation

For every operation in [the numpy interface subset](numpy-interface-subset.md)'s table: a
randomized-input test comparing the Rust result against **both** real `numpy`'s own result and
the existing pure-Python reference implementation
(`BackpropNode.z()`/`sigmoid()`/[vectorized array-based model classes](vectorized-array-classes.md)'s
own already-numpy-parity-checked methods) across a large random-input sweep - the same discipline
as the 200k-random-z bit-identical sigmoid check already precedented in this codebase, extended to
every operation in the subset rather than assumed to transfer. The two known risk areas
(summation order, sigmoid-overflow behavior) already flagged in
[vectorized array-based model classes](vectorized-array-classes.md)'s own "numerical parity
validation" section need this check explicitly and early, not as an afterthought - checking
against numpy transitively re-validates against the pure-Python reference that document's own
classes were already checked against, so a three-way match (Rust, numpy, pure Python) is strictly
stronger evidence than a two-way one. The uniform RNG fill is the one named exception to "same
discipline" - see PR 7 above for why bit-identical parity isn't the achievable bar there, and what
gets checked instead.

### 5. what stays explicitly out of scope for this workplan

- **Retargeting production code paths to this core.** [vectorization](vectorization.md#decision-numpy-stays-permanently-as-a-benchmark-mirror-the-rust-core-becomes-production)
  records *that* this core becomes production and `vectorized-array-classes.md`'s numpy-backed
  classes stay on as a permanent benchmark mirror - but the actual cutover (updating whichever
  training/demo entrypoints should route through this core once it exists and passes its own
  parity checks) is a small, mechanical follow-on of its own (a new call path alongside the
  existing numpy one, not an import swap that removes it - the numpy version is being kept, not
  retired), not something this workplan does or assumes as part of "building the core."
- **BLAS-competitive matmul performance.** Naive-but-correct first; any tuning is a
  separately-measured follow-up, not bundled into "does it work."
- **Any operation outside [the numpy interface subset](numpy-interface-subset.md)'s own table.**
  No axis-parameterized reductions, no `where`/`maximum`/`clip`, no general N-d arrays or
  broadcasting - see that document's own "explicitly not required" for what a future
  ReLU/softmax/momentum/L2 vectorized variant would need to add here first, before this core
  could support it.
- **Any change to `perceptron/model/`.** This document builds and validates a fully standalone
  Rust extension - nothing in the existing pure-Python codebase changes as a result of it.

## what this document is not

As of PR #183 (the consolidated parity suite), no longer purely an analysis and a workplan - see
"status" at the top: the crate exists, every operation in the interface subset is implemented and
parity-tested. Still not a migration plan - `perceptron/model/`'s existing classes are completely
untouched, and [vectorized array-based model classes](vectorized-array-classes.md)'s own
numpy-backed classes remain the permanent benchmarking mirror they were decided to be (see
[vectorization](vectorization.md#decision-numpy-stays-permanently-as-a-benchmark-mirror-the-rust-core-becomes-production)),
not superseded by this core's existence. What's still open: the actual production cutover -
retargeting those classes (or building new ones) to call this core instead of `numpy` - which
this document's own "what stays explicitly out of scope" section deliberately left for a separate
follow-on, not something building the core did automatically.
