"""
Ensures this checkout's dataset source files are present and checksum-verified, fetching from
their pinned `indrajala-datasets-*` repo (see docs/dataset-sourcing-proposal.md) only when a file
is missing or doesn't match its expected SHA-256.

Presence+checksum is checked first, network only as a last resort: a repeated `./cli setup`/
`./cli fetch-data` against an unchanged local checkout performs zero network calls after the
first successful fetch. Only MNIST is fetched here - UCI digits' digits.csv stays committed
directly in `perceptron` (see the proposal's "what changes for UCI digits" section), with its own
`indrajala-datasets-uci-digits` packaging existing for metadata consistency, not because
perceptron needs to fetch it.
"""

import hashlib
import os
import urllib.request

PINNED_REF = "v2026-09-16"
_RAW_BASE = f"https://raw.githubusercontent.com/davidbarkhuizen/indrajala-datasets-mnist/{PINNED_REF}/data"

DATASETS = [
    {
        "local_path": "data/mnist/mnist-train.parquet",
        "sha256": "f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf",
        "url": f"{_RAW_BASE}/mnist-train.parquet",
    },
    {
        "local_path": "data/mnist/mnist-test.parquet",
        "sha256": "d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b",
        "url": f"{_RAW_BASE}/mnist-test.parquet",
    },
]


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset_file(local_path: str, expected_sha256: str, fetch_url: str) -> None:
    if os.path.exists(local_path) and sha256_of(local_path) == expected_sha256:
        print(f"{local_path}: present and verified, skipping fetch")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"{local_path}: fetching from {fetch_url} ...")
    urllib.request.urlretrieve(fetch_url, local_path)

    actual_sha256 = sha256_of(local_path)
    assert actual_sha256 == expected_sha256, (
        f"{local_path}: downloaded but checksum didn't match "
        f"(expected {expected_sha256}, got {actual_sha256})"
    )
    print(f"{local_path}: fetched and verified")


def main() -> None:
    for dataset in DATASETS:
        ensure_dataset_file(dataset["local_path"], dataset["sha256"], dataset["url"])


if __name__ == "__main__":
    main()
