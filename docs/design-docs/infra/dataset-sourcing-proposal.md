# dataset sourcing: the `indrajala-datasets-*` proposal

[← back to README](../../../README.md)

Design reference for `scripts/fetch_datasets.py` and the CI dataset-caching setup - see
[setup](../../project/setup.md#mnist-data) for the current, shipped state (pinned tags, what runs when).

## the fetch mechanism, and why it only fetches once locally

`scripts/fetch_datasets.py` (wired into `./cli setup` and a standalone `./cli fetch-data`) runs,
per file, this check - **presence and checksum first, network only as a last resort**:

```
def ensure_dataset_file(local_path, expected_sha256, fetch_url):
    if os.path.exists(local_path) and sha256_of(local_path) == expected_sha256:
        return  # already present and verified - no network call at all
    fetch(fetch_url, local_path)          # only reached if missing or corrupted
    actual = sha256_of(local_path)
    assert actual == expected_sha256, f"downloaded {local_path} but checksum didn't match"
```

The checksum comparison is a local, in-memory operation (a 15 MB file's SHA-256 takes a small
fraction of a second) with zero network cost, so running `./cli setup`/`./cli test` repeatedly
against an unchanged local checkout performs exactly one network fetch per file, ever - the first
time the file is missing. Every subsequent run short-circuits at the presence+checksum check
before reaching `fetch()` at all.

**In CI**, this is a different context, not a violation of the same guarantee: an ephemeral
GitHub Actions runner starts with no cached filesystem, so the check always misses and one fetch
happens - once per CI run (not once per test inside that run, since the fetch step runs once
before `pytest` starts, exactly like `./cli setup` does locally), *unless* CI-side caching (below)
is added.

Fetches a specific tag/commit of the dataset repo, not `main`/`latest` - a future change to
`indrajala-datasets-mnist` shouldn't silently change what `indrajala-ml`'s CI tests against between
one PR and the next. The pinned reference lives in `indrajala-ml`'s own config (a constant in the
fetch script), bumped deliberately when there's a reason to.

## CI-side caching

Implemented via GitHub Actions' first-party `actions/cache`, keyed on the pinned
`indrajala-datasets-mnist` reference *plus* the file's expected `sha256` from its metadata (not a
static key) - so bumping the pinned commit/tag automatically invalidates the cache and forces a
fresh fetch+verify, rather than risking a stale cache silently serving old data forever:

```yaml
- uses: actions/cache@v4
  with:
    path: data/mnist
    key: mnist-${{ env.PINNED_REF }}-${{ env.EXPECTED_SHA256 }}
- run: ./cli fetch-data   # short-circuits per-file if the cache already restored a verified copy -
                          # the same presence+checksum check above covers this case for free
```

No new logic needed beyond what `ensure_dataset_file` above already does: a cache hit just means
the presence+checksum check succeeds without `fetch()` ever running, exactly like the local-dev
case. `indrajala-ml` is a **public** GitHub repo, so this isn't a cost-saving measure in any dollar
sense today - what it buys is faster CI feedback and less load on `indrajala-datasets-mnist`
itself. The one scenario where this stops being purely a speed optimization: if `indrajala-ml` or
`indrajala-datasets-*` ever go private, at which point skipping the download/conversion step would
directly reduce billed Actions minutes.
