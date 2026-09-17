# benchmarking infrastructure

[← back to README](../README.md)

**Status: proposed, not started.** Written up front as a design/workplan doc before any of it
exists, per this repo's own practice (see [Adam optimizer](adam-optimizer.md), [an array-based
Adam sibling](adam-array-layer.md) for precedent).

## why this, and why now

[Structure](structure.md#possible-next-steps)'s "infrastructure that protects the rigor" tier
flagged an audit item: every real-scale measurement this codebase has run - Adam's batch-size
sweeps, the learning-rate-vs-batch-size follow-up, RMSprop's ablation, the warmup-step sweep, L2's
and dropout's array-stage sweeps (see [research: Adam
optimizer](research-adam-optimizer.md), [research: backprop
siblings](research-backprop-siblings.md), [an array-based L2
sibling](l2-array-layer.md#measurement-plan-and-result-stage-3), [an array-based dropout
sibling](dropout-array-layer.md#measurement-plan-and-result-stage-3)) - hand-rolled its own
uncommitted one-off script, per this codebase's own established practice. None of those scripts
are in the repo; only the prose describing what each one did survives.

An audit of that prose (not the code, which doesn't exist to audit) found four pieces recurring
nearly identically every time:

1. **A fixed MNIST digit-N proxy dataset** - a balanced binary (digit K vs. every other digit,
   stratified) or, less often, multiclass subset, built once and reused across every seed in a
   sweep, never resampled per seed. Sizes vary (320 examples in the earlier momentum/Adam sweeps,
   400 in L2's and dropout's), but the shape is the same every time.
2. **A `multiprocessing.Pool` sweep runner** over (config, seed) pairs, each job handed its own
   fixed seed from a caller-supplied seed list so results are reproducible from that list.
3. **A wall-clock calibration probe**, run once before committing to the full sweep, to project
   total cost and decide up front whether it's affordable - done by hand, informally, in every
   sweep so far.
4. **Mean/stdev-into-markdown-table aggregation**, in the identical `mean% ± stdev%` per-config
   format every results table in every research doc already uses.

What legitimately stays one-off, and isn't proposed for extraction here: the actual config grid
swept (momentum coefficients, learning rates, batch sizes, drop probabilities, warmup steps),
the architecture/hyperparameters chosen per measurement, and the interpretation/decision write-up
- all inherently bespoke to the question being asked.

One concrete, already-diagnosed gap rides along: a long sweep launched via `run_in_background`,
redirected to a file, shows no interim progress at all until it exits or its stdout buffer fills
- Python fully buffers stdout when it isn't a tty, so a per-config `print()` meant to show
progress during a 20+ minute run silently queues instead of appearing. Found while running [an
array-based dropout sibling](dropout-array-layer.md)'s own stage-3 sweep; worth fixing in the
runner itself (`flush=True` on progress output) rather than rediscovering it on the next one.

**Constraint from [goals and strategy](goals-and-strategy.md#measurement-discipline-the-per-node-paths-two-jobs-and-the-one-it-doesnt-have)'s
"measurement discipline" section**: going forward, wall-clock and accuracy-at-scale sweeps compare
numpy/Rust array backends, not the per-node path - re-timing per-node is explicitly discouraged as
re-confirming an already-established result. The sweep runner's worker contract is therefore
backend-agnostic (any `(config, seed) -> result` callable) rather than built around the per-node
path specifically.

## scope

Two new modules, both reusing existing, already-tested building blocks rather than reimplementing
them:

**`indrajala_ml/benchmark_data.py`** - `build_mnist_digit_proxy(path, digits, examples_per_class,
split_fraction=0.8, seed=None)`, returning a small `BenchmarkProxy` (`train_x`/`train_y`/`test_x`/
`test_y` numpy arrays). `digits=[k]` (binary, one digit vs. every other) is built directly on
`ensemble_train.select_balanced_indices(labels, target_label, class_count, rng)` - already-tested,
already-documented stratified one-vs-rest index selection, exactly the logic every existing binary
sweep hand-rolled - then capped to `examples_per_class` per side. `len(digits) > 1` is a genuine
N-way multiclass proxy over exactly the listed digits (no "everything else" bucket), which needs a
small new stratified-by-digit-list selector since `select_balanced_indices` is structurally
one-vs-rest and doesn't generalize to this case. Both paths decode only the selected indices via
`mnist_data.load_mnist_records_at_indices` (never the full 60000/10000-record file), then split
into a stratified train/test pair that preserves per-class balance in both halves.

**`indrajala_ml/benchmark_sweep.py`** - `run_parameter_sweep(configs, seeds, worker_fn,
worker_count=None)`, a `multiprocessing.Pool` dispatching one job per (config, seed) pair (each
job handed its own fixed seed from `seeds`, with whichever RNG a `worker_fn` needs seeded left to
that function itself, exactly as every existing sweep script already does by hand), small
picklable results collected via `pool.imap`; `estimate_sweep_wallclock(worker_fn, sample_config,
sample_seed, planned_run_count, worker_count)`, running one real job serially and projecting total
wall-clock, replacing the informal calibration step every sweep so far has done by hand;
`summarize_sweep_results(results)`, rendering mean/stdev per config as the same markdown table
format already used everywhere.

Not in scope: migrating the wall-clock tables and methodology notes already living in
[goals and strategy](goals-and-strategy.md#measurement-discipline-the-per-node-paths-two-jobs-and-the-one-it-doesnt-have),
[research and analysis](research-and-analysis.md), [research: Rust
performance](research-rust-performance.md), [research: backprop
siblings](research-backprop-siblings.md), and each array-layer sibling's own "measurement plan"
section into this doc. That consolidation is a separate, not-yet-committed judgment call this
audit's own item explicitly left open ("worth determining... rather than assuming either answer up
front") - this workplan builds and documents the reusable infrastructure itself, it doesn't move
existing content. Also not in scope: running a real sweep with the new infrastructure. The first
real usage is deferred to whatever the next genuine measurement need turns out to be (e.g.
[convolutional layers](conv-array-layer.md)'s real-MNIST validation, or [an array-based ensemble
sibling](ensemble-array-layer.md)'s own measurement) - validated here only against a toy worker
function, not by re-running an already-closed sweep purely to reconfirm a result this codebase's
own measurement discipline already says not to re-time.

## design

`build_mnist_digit_proxy` is deliberately additive to `mnist_data.py`'s existing loaders rather
than a replacement for any of them - `load_mnist_labels` and `load_mnist_records_at_indices`
already do exactly what's needed (cheap label-only scan, then decode only a specific index list),
and `select_balanced_indices` already does exactly the binary stratified-selection work. The new
code's actual job is narrow: cap an already-balanced index selection down to a fixed
`examples_per_class`, generalize stratified selection to an explicit multi-digit list where no
existing helper reaches, and produce a stratified train/test split - not reinventing dataset
loading or class balancing.

`run_parameter_sweep`'s `worker_fn` must be a module-level function, not a local closure or
lambda - the same picklability constraint `ensemble_train.py`'s own worker functions already
have, for the same reason: `Pool` workers receive tasks through a queue, which pickles whatever
callable and arguments each task carries. Rather than re-pickling `worker_fn` on every one of
potentially hundreds of (config, seed) tasks, it's pickled once per worker process via `Pool`'s
own `initializer`/`initargs` - cheap even when `worker_fn` closes over a real, non-trivial shared
context (typically one `BenchmarkProxy`, built once by the caller before this call). Only
`configs` and `seeds` travel through `imap`'s own per-job queue.

Unlike `ensemble_train.py`'s own workers, `run_parameter_sweep` does not reseed any RNG itself -
different `worker_fn`s may need to seed Python's `random`, numpy's RNG, or (for eval-mode-only
paths) nothing at all, so which RNG to seed is left to `worker_fn`, exactly as every existing
sweep script already does by hand; the runner's only job is handing each job its own fixed seed
from the caller's `seeds` list.

## tests

`tests/test_benchmark_data.py`, following `test_mnist_data.py`'s own conventions: the real bundled
`data/mnist/mnist-train.bin`/`mnist-test.bin` files (no synthetic fixture data), small
`examples_per_class` values for speed, assertions against known real-label distributions. Covers:
binary proxy shape/balance, multiclass proxy shape/balance across an arbitrary digit subset,
train/test split balance, determinism under a fixed `seed`.

`tests/test_benchmark_sweep.py`, against a cheap deterministic toy `worker_fn` (a pure function of
`(config, seed)`, not real model training) - matches this workplan's own validation approach
above. Covers: `run_parameter_sweep` dispatches every (config, seed) pair exactly once and
aggregates correctly; `estimate_sweep_wallclock` returns a sane projection for a known-duration toy
worker; `summarize_sweep_results`'s markdown output format.

## delivery stages (each its own PR, per this repo's practice)

1. This design document.
2. `benchmark_data.py`'s binary one-vs-rest proxy builder (reusing `select_balanced_indices`) +
   `tests/test_benchmark_data.py`'s binary-path tests.
3. `benchmark_data.py`'s multiclass digit-subset extension + its own tests.
4. `benchmark_sweep.py` (runner + calibration probe + aggregation, including the stdout-flush fix)
   + `tests/test_benchmark_sweep.py` against the toy worker function.
5. Docs closeout: [structure](structure.md#possible-next-steps)'s "audit of hand-rolled
   measurement scripts" entry updated to reflect the audit's finding and link here.
