# setup

[← back to README](../README.md)

## requirements

- Python 3.10+
- `python3-tk` (for the interactive matplotlib `TkAgg` backend used by `. cli demo`; the
  tests themselves are headless and don't need it)
- `cargo`/`rustc` and `maturin` (for building [the Rust array core](rust-array-core.md),
  `rust/perceptron_array/`)

## install

    . cli setup

This installs `python3-tk`/`cargo` via `apt`, creates a `.venv`, installs the Python
dependencies (`matplotlib`, `pytest`, `pyarrow`, `numpy` - see
[vectorized array-based classes](vectorized-array-classes.md) for `numpy`'s scoped,
benchmark-mirror-only role) from `requirements.txt`, installs `maturin` into that venv, and
builds the Rust array core into it (`maturin develop`, run from `rust/perceptron_array/`).
`. cli build-rust` re-runs just that last step (rebuild after changing Rust source, without
redoing the rest of setup).

## test

    . cli test

Runs everything under `tests/` with pytest, grouped one file per module under test (see
[structure](structure.md)), plus `test_training_pipeline.py` for end-to-end coverage - it
builds the same convergence-curve and decision-boundary charts as `. cli demo`, using
matplotlib's `Agg` backend, but never opens a window — so all of it runs unattended (no display
needed).

This does **not** include [the Rust array core](rust-array-core.md)'s own tests
(`rust/perceptron_array/tests/`, 254 tests) - those are pytest tests against the built extension,
kept alongside the standalone crate rather than under the top-level `tests/` directory. Run them
directly, after `. cli setup`/`. cli build-rust` has built the extension into the venv:

    source .venv/bin/activate
    python -m pytest rust/perceptron_array/tests/

One exception: `tests/test_mnist_data.py`'s tests need the real MNIST parquet/binary files
(`data/mnist/*.parquet`, `data/mnist/*.bin` - see `mnist_data.py`). These are deliberately not
committed (large binary data) and have no scripted fetch step anywhere in this codebase - the
`.parquet` files are supplied locally by whoever set up this checkout, and
`mnist_data.convert_parquet_to_binary` (which `. cli demo`'s **MNIST ensemble recognition** demo
calls automatically the first time it runs - see [demos](demos.md)) only converts an
already-present `.parquet` file to the flat binary format training actually reads; it doesn't
fetch the `.parquet` file itself from anywhere. Every other test file runs against bundled
(`data/digits/digits.csv`) or synthetic data and needs nothing external. On a fresh checkout
without the MNIST `.parquet` files, both `test_mnist_data.py` and the MNIST demo will fail until
that data is supplied.

`. cli clean` removes `__pycache__`/`.pytest_cache` directories (used automatically before
`. cli test`).
