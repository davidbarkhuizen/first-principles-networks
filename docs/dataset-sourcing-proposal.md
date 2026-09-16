# dataset sourcing: the `indrajala-datasets-*` proposal

[← back to README](../README.md)

A design proposal, not yet built - per this repo's own practice of writing a plan down before
implementation (see [the production cutover plan](rust-production-cutover.md) for the precedent).
Nothing in this document has been created yet: no GitHub repo, no fetch script, no CI workflow.

## the problem this solves

[Structure](structure.md#possible-next-steps)'s CI item is blocked on one unresolved question:
where does a fresh checkout get the real MNIST data from? `tests/test_mnist_data.py` needs the
real `.parquet` files, which are gitignored (large binary data) with no scripted fetch step
anywhere in this codebase - supplied locally, by hand, whenever a dev checkout needs them. CI
can't run against a file that only exists because someone once copied it in manually.

## datasets in scope

Exactly two exist in this codebase today (checked directly, not assumed):

| dataset | current state | size | blocking anything? |
|---|---|---|---|
| **UCI digits** (`data/digits/digits.csv`) | committed directly to `perceptron`, works today | 260 KB | No - included here for consistency, not urgency |
| **Real MNIST** (`data/mnist/mnist-train.parquet`, `mnist-test.parquet`) | gitignored, supplied locally by hand | ~18 MB total | Yes - the actual CI blocker |

UCI digits' provenance (per [structure](structure.md#multi-class)): a one-time, offline extraction
from `sklearn.datasets.load_digits()` (1797 rows, 8x8 pixel images, 64 pixels + a label per row) -
scikit-learn was never a runtime dependency, just a throwaway extraction tool. MNIST's provenance
(per `mnist_data.py`) is the standard 60000-train/10000-test handwritten digit database, currently
held as Parquet (the format `mnist_data.py`'s `convert_parquet_to_binary` already expects as
input).

## proposal: one `indrajala-datasets-<name>` repo per dataset, under the personal account

Per the structural decision already made: repos under `github.com/davidbarkhuizen/`, not a new
org - `indrajala-datasets-mnist`, `indrajala-datasets-uci-digits`. Each repo stores its dataset in
the canonical format the consuming project (`perceptron`) already reads, not a re-derived format -
the dataset repo is a documented *source*, not a preprocessed cache. This keeps `perceptron`'s own
loader code (`mnist_data.py`, `digits_data.py`) unchanged: they still read Parquet/CSV, just from a
fetched path instead of a manually-supplied one.

### repo layout

```
indrajala-datasets-mnist/
├── README.md                          — human overview, quick start
├── LICENSE                            — license for this repo's own packaging/metadata
├── metadata.json                      — repo-level metadata
└── data/
    ├── mnist-train.parquet
    ├── mnist-train.parquet.metadata.json
    ├── mnist-test.parquet
    └── mnist-test.parquet.metadata.json
```

`indrajala-datasets-uci-digits` mirrors this exactly, with `data/digits.csv` +
`data/digits.csv.metadata.json` and no train/test split (UCI digits is one file,
`perceptron/digits_data.py`'s own `split_train_test` does the split at load time, not the source
data).

### repo-level `metadata.json`

```json
{
  "name": "mnist",
  "description": "The MNIST handwritten digit database - 60000 train / 10000 test 28x28 grayscale images, packaged as Parquet.",
  "source": {
    "name": "Yann LeCun / Corinna Cortes' MNIST database",
    "url": "TODO - fill in and verify the canonical source URL",
    "license": "TODO - verify and state the dataset's own usage terms explicitly, not assumed"
  },
  "version": "2026-09-16",
  "maintainer": "davidbarkhuizen",
  "format": "Apache Parquet, one row per image",
  "files": ["data/mnist-train.parquet", "data/mnist-test.parquet"]
}
```

`indrajala-datasets-uci-digits`'s own repo-level `metadata.json` would state its real, known
provenance directly rather than a `TODO` - `sklearn.datasets.load_digits()`, itself sourced from
the UCI ML repository's optical handwritten digits dataset - since that chain is already
documented in this codebase and doesn't need re-verifying.

### per-file `<filename>.metadata.json`

```json
{
  "filename": "mnist-train.parquet",
  "split": "train",
  "format": "parquet",
  "size_bytes": 15561616,
  "sha256": "TODO - computed at publish time",
  "record_count": 60000,
  "schema": {
    "columns": ["image (28x28 uint8)", "label (uint8, 0-9)"]
  },
  "notes": "Well-known real-MNIST first-5 training labels: 5,0,4,1,9 - a useful load-order sanity check, already relied on by perceptron's own test suite (tests/test_mnist_data.py)."
}
```

The `sha256` field is load-bearing, not decorative: it's what lets a fetch be verified, not just
downloaded (see below).

## how `perceptron` would consume it

### the fetch mechanism, and why it only fetches once locally

A new script (e.g. `scripts/fetch_datasets.py`, wired into `./cli setup` and a standalone
`./cli fetch-data`) runs, per file, this check - **presence and checksum first, network only as a
last resort**:

```
def ensure_dataset_file(local_path, expected_sha256, fetch_url):
    if os.path.exists(local_path) and sha256_of(local_path) == expected_sha256:
        return  # already present and verified - no network call at all
    fetch(fetch_url, local_path)          # only reached if missing or corrupted
    actual = sha256_of(local_path)
    assert actual == expected_sha256, f"downloaded {local_path} but checksum didn't match"
```

**This directly answers the "only fetch once" requirement**: the checksum comparison is a local,
in-memory operation (a 15 MB file's SHA-256 takes a small fraction of a second) with zero network
cost, so running `./cli setup`/`./cli test` repeatedly against an unchanged local checkout performs
exactly one network fetch per file, ever - the first time the file is missing. Every subsequent
run short-circuits at the presence+checksum check before reaching `fetch()` at all. This is the
same effective behavior as today's "supplied locally, once, by hand" workflow - just automated and
verified instead of manual and unverified.

**In CI**, this is a different context, not a violation of the same guarantee: an ephemeral
GitHub Actions runner starts with no cached filesystem, so the check always misses and one fetch
happens - once per CI run (not once per test inside that run, since the fetch step runs once
before `pytest` starts, exactly like `./cli setup` does locally). A later optimization (GitHub
Actions' own dependency-cache action, keyed on the file's expected sha256) could turn "once per
run" into "once per cache key," but that's a genuine future refinement, not part of this proposal.

### pinning

Fetches a specific tag/commit of the dataset repo, not `main`/`latest` - a future change to
`indrajala-datasets-mnist` shouldn't silently change what `perceptron`'s CI tests against between
one PR and the next. The pinned reference lives in `perceptron`'s own config (e.g. a constant in
the fetch script), bumped deliberately when there's a reason to.

### integration points

- `./cli setup` calls the fetch step for every dataset after `install_python_modules`, so a fresh
  checkout ends up with the same local layout (`data/mnist/*.parquet`, `data/digits/digits.csv`)
  it has today - `mnist_data.py`/`digits_data.py`/every existing test path needs zero changes.
- A standalone `./cli fetch-data` for re-running just this step (mirroring `./cli build-rust`'s
  relationship to `./cli setup`).
- CI's workflow calls the same script (or `./cli setup` wholesale) before running `pytest`.

### what changes for UCI digits, if anything

Genuinely optional, and not required to unblock CI (which only needs MNIST): `data/digits/digits.csv`
already works, committed directly, no problem to solve. `indrajala-datasets-uci-digits` would
exist for consistency (one place with proper metadata for every dataset this codebase uses) but
`digits.csv` could stay committed directly in `perceptron` regardless - the fetch mechanism doesn't
require removing the committed copy, and given the file is 260 KB with zero blocker attached to
it, there's no urgency to change what already works. This is a call worth making explicitly, not
assuming either way.

## risks and open questions

- **License/provenance for MNIST is a real "verify, don't assume" item** - deliberately left as
  `TODO` in the metadata template above rather than asserted, consistent with this codebase's own
  standard of checking rather than guessing at facts it can't independently confirm.
- **Single point of failure, but one you control**: `indrajala-datasets-mnist` being unavailable
  would block `perceptron`'s CI the same way a third-party mirror would - the difference from the
  "public mirror" option considered earlier is that you own its availability, not a stranger's, but
  it's still a real external dependency at fetch time, not eliminated entirely.
- **Checksum computed once, at publish time** - if the dataset repo's own file is ever
  regenerated/re-exported, its `sha256` must be recomputed and the metadata updated, or every
  consumer's fetch starts failing the integrity check (the correct failure mode - loud, not
  silent - but worth knowing it'll happen if the source file changes without a metadata bump).
- **Whether to also migrate `digits.csv`** - see above; a genuine open question, not a decision
  this document makes.

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. Compute real checksums + write real metadata for the existing local MNIST parquet files (and
   UCI digits, if that migration is in scope) - a local, no-GitHub-action-required step.
3. Create `indrajala-datasets-mnist` (and optionally `indrajala-datasets-uci-digits`) on GitHub,
   push the data + metadata - a real, visible, public action requiring explicit go-ahead, not
   bundled into an automated step.
4. Add the fetch script + `./cli` wiring in `perceptron`.
5. Add the CI workflow itself (GitHub Actions running `pytest` on push/PR), now unblocked.
6. Update `docs/setup.md`/`docs/structure.md` to describe the shipped result, closing out the CI
   item in [structure](structure.md#possible-next-steps).
