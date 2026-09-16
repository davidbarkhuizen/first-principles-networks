# setup

[← back to README](../README.md)

## requirements

- Python 3.10+
- `python3-tk` (for the interactive matplotlib `TkAgg` backend used by `. cli demo`; the
  tests themselves are headless and don't need it)
- `cargo`/`rustc` and `maturin` (for building [the Rust array core](rust-array-core.md),
  `rust/indrajala_ml_array/`)

## install

    . cli setup

This installs `python3-tk`/`cargo` via `apt`, creates a `.venv`, installs the pinned Python
dependencies (`matplotlib`, `pytest`, `pyarrow`, `numpy`, plus this package itself - see
"dependencies" below - and see [vectorized array-based classes](vectorized-array-classes.md) for
`numpy`'s scoped, benchmark-mirror-only role) from `requirements.txt`, installs `maturin` into
that venv, builds the Rust array core into it (`maturin develop --release`, run from
`rust/indrajala_ml_array/` - see [the production cutover plan](rust-production-cutover.md) for why
a debug build isn't good enough here), and fetches the real MNIST dataset (see "MNIST data"
below).
`. cli build-rust` re-runs just the Rust build step, and `. cli fetch-data` re-runs just the
dataset fetch - each independently, without redoing the rest of setup.

## dependencies

`pyproject.toml` declares this package (`indrajala-ml`, versioned, installable via `pip install -e .`
- previously checkout-and-run only, relying on `python -m`'s implicit cwd-on-`sys.path` behavior)
and its direct dependencies, unpinned. `requirements.txt` is the actual **pinned lock** `. cli
setup` installs from (`pip install -r requirements.txt`, which includes `-e .`): every direct and
transitive dependency pinned to the exact version this codebase's test suite is validated against,
so a fresh checkout can't silently get a different, untested dependency version than the one this
repo's own results were measured on.

To deliberately bump a dependency: edit `requirements.in` (or `pyproject.toml`'s dependency list),
regenerate with `pip install pip-tools && pip-compile requirements.in`, then **re-run the full test
suite before committing the new `requirements.txt`** - a lock file is only trustworthy if it's
pinned to versions that were actually validated, not just whatever pip-compile resolved to freshest
that day.

## test

    . cli test

Runs everything under `tests/` with pytest, grouped one file per module under test (see
[structure](structure.md)), plus `test_training_pipeline.py` for end-to-end coverage - it
builds the same convergence-curve and decision-boundary charts as `. cli demo`, using
matplotlib's `Agg` backend, but never opens a window — so all of it runs unattended (no display
needed).

This does **not** include [the Rust array core](rust-array-core.md)'s own tests
(`rust/indrajala_ml_array/tests/`, 525 tests) - those are pytest tests against the built extension,
kept alongside the standalone crate rather than under the top-level `tests/` directory. Run them
directly, after `. cli setup`/`. cli build-rust` has built the extension into the venv:

    source .venv/bin/activate
    python -m pytest rust/indrajala_ml_array/tests/

One exception: `tests/test_mnist_data.py`'s tests need the real MNIST parquet/binary files
(`data/mnist/*.parquet`, `data/mnist/*.bin` - see `mnist_data.py`). These are deliberately not
committed (large binary data), gitignored, and regenerated/refetched locally rather than checked
in - see "MNIST data" below for how a fresh checkout gets them. Every other test file runs against
bundled (`data/digits/digits.csv`) or synthetic data and needs nothing external.

`. cli clean` removes `__pycache__`/`.pytest_cache` directories (used automatically before
`. cli test`).

## MNIST data

    . cli fetch-data

`scripts/fetch_datasets.py` (run automatically as part of `. cli setup`, or standalone via the
command above) ensures `data/mnist/mnist-train.parquet`/`mnist-test.parquet` are present and
SHA-256-verified, fetching from the pinned
[`indrajala-datasets-mnist`](https://github.com/davidbarkhuizen/indrajala-datasets-mnist)`@v2026-09-16`
tag only when a file is missing or doesn't match - see [the dataset sourcing
proposal](dataset-sourcing-proposal.md) for the full design and rationale. It then regenerates
`mnist-train.bin`/`mnist-test.bin` from the parquet (via `mnist_data.convert_parquet_to_binary`) if
those are missing too. Both checks are presence-first: a repeated `. cli setup`/`. cli fetch-data`
against an already-populated checkout makes no network calls and does no reconversion.

UCI digits (`data/digits/digits.csv`) needs none of this - it's small enough (260 KB) to stay
committed directly in this repo; its own
[`indrajala-datasets-uci-digits`](https://github.com/davidbarkhuizen/indrajala-datasets-uci-digits)
packaging exists for metadata consistency with MNIST, not because this repo needs to fetch it.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs `. cli setup` + `. cli test` on
every push to `main` and on every pull request - the same commands described above, so there's
nothing CI-specific to keep in sync by hand. `data/mnist` is cached between runs
(`actions/cache`, keyed on `scripts/fetch_datasets.py`'s own content, so bumping the pinned tag or
a checksum automatically invalidates the cache) to skip the ~18 MB download and `.bin` conversion
on every run.
