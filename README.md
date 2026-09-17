# indrajala-ml

A small, dependency-light implementation of two classifier families, built from first
principles. `LinearClassifierNetwork` composes state (input) nodes and association
(weighted, thresholded) nodes/layers per Rosenblatt's perceptron (1958), trained with the
classic perceptron learning rule (and a MADALINE-style minimum-disturbance rule once more
than one hidden node is used). `BackpropClassifierNetwork` is a sigmoid, gradient-descent
network of arbitrary depth, added alongside it - a genuinely different learning rule, not a
retrofit - so it can represent targets (like XOR) the discrete model structurally can't.

## docs

Grouped by kind, mirrored 1:1 by `docs/`'s own folder layout - each section below is a directory.

### project (`docs/project/`)

Orientation: what this project is, how it's laid out, and how to run it.

- [goals and strategy](docs/project/goals-and-strategy.md) — what this project is actually for,
  and how to evaluate a proposed addition
- [structure](docs/project/structure.md) — module layout, and how each network composes, including
  every `BackpropClassifierNetwork` sibling (Adam included) and which are adopted vs. kept as
  tested nulls
- [setup](docs/project/setup.md) — requirements, install, running the tests, fetching real
  MNIST/UCI digits from dedicated, checksum-verified `indrajala-datasets-*` repos, and
  [CI](.github/workflows/ci.yml), which runs the full test suite on every push/PR
- [demos](docs/project/demos.md) — the demo scripts and what each one shows
- [theory](docs/project/theory.md) — Rosenblatt's perceptron theory, and reference material

### architecture (`docs/architecture/`)

What the array/Rust performance backend is and how it's built, independent of any one model
capability.

- [vectorization](docs/architecture/vectorization.md) — numpy-backed array classes (kept as a
  permanent benchmarking mirror) and a hand-built Rust array core (the production backend)
- [vectorized array-based classes](docs/architecture/vectorized-array-classes.md) — the numpy
  `ArrayLayer`/`VectorizedMultiClassBackpropClassifierNetwork` design and its numerical-parity
  validation against the per-node reference
- [the Rust array core](docs/architecture/rust-array-core.md) — `indrajala_ml_array`'s operation
  subset, parity testing, and the one documented RNG exception
- [the numpy interface subset](docs/architecture/numpy-interface-subset.md) — exactly which numpy
  operations this codebase re-implements, and which are explicitly out of scope
- [Rust production cutover](docs/architecture/rust-production-cutover.md) — how the Rust core
  became primary in production: a debug-build fix, a matmul reorder, fused per-layer calls, and
  matmul SIMD work, landing on a Rust core that's **faster than numpy** on real training runs -
  currently ~3.6x at UCI digits scale, ~2.6x at real MNIST scale (3.40x/1.31x at first cutover,
  improved by later matmul SIMD work - see
  [research and analysis](docs/research/research-and-analysis.md))
- [benchmarking](docs/architecture/benchmarking.md) — reusable measurement infrastructure (a fixed
  MNIST-proxy builder, a fork-based sweep runner, wall-clock calibration, and mean/stdev
  aggregation) for the hand-rolled measurement scripts every real-scale sweep used to re-roll from
  scratch

### features (`docs/features/`)

Built, adopted, model-level capabilities that aren't `BackpropClassifierNetwork` siblings on their
own (see "structure" above for those).

- [mini-batch gradient descent](docs/features/mini-batch-gradient-descent.md) — batched gradient
  updates, and the momentum re-test they unblocked, plus the learning-rate warmup schedule that
  later fixed the large-batch divergence they surfaced
- [convolutional layers](docs/features/convolutional-layers.md) — a from-scratch conv layer, local
  receptive fields and weight sharing, measured against the dense baseline on UCI digits
- [dropout](docs/features/dropout.md) — stochastic hidden-unit regularization, built and measured
  (via its own array-based sibling): a reconfirmed null, the same as L2's own finding

### research (`docs/research/`)

Investigations behind a design decision, with the measurements that drove it - split by theme,
starting from the index.

- [research and analysis](docs/research/research-and-analysis.md) — index: what this collection of
  docs is for, and links into each themed sub-doc below
- [multi-class architecture and loss functions](docs/research/research-multiclass-and-loss.md)
- [backprop sibling measurements](docs/research/research-backprop-siblings.md) — including every
  array/Rust-matmul-backed sibling port (momentum, L2, ReLU, dropout) and its own wall-clock/
  accuracy result
- [Rust core performance](docs/research/research-rust-performance.md)
- [Adam optimizer](docs/research/research-adam-optimizer.md) — including the array-based Adam
  sibling's own wall-clock/accuracy result

### design docs (`docs/design-docs/`)

Finished workplans, trimmed down to the design/API reference each one's own implementation still
cites in its docstrings, once their actual measured results were folded into
[research and analysis](docs/research/research-and-analysis.md) (or, for dataset sourcing,
[setup](docs/project/setup.md#mnist-data)) as the permanent record of what was found.

#### Adam (`docs/design-docs/adam/`)

- [Adam optimizer](docs/design-docs/adam/adam-optimizer.md) —
  `AdamBackpropClassifierNetwork`'s per-parameter adaptive-learning-rate design
- [an array-based Adam sibling](docs/design-docs/adam/adam-array-layer.md) — the
  numpy/Rust-matmul-backed port

#### array-ported siblings (`docs/design-docs/array-siblings/`)

- [an array-based momentum sibling](docs/design-docs/array-siblings/momentum-array-layer.md)
- [an array-based L2 sibling](docs/design-docs/array-siblings/l2-array-layer.md)
- [an array-based ReLU sibling](docs/design-docs/array-siblings/relu-array-layer.md)
- [an array-based softmax sibling](docs/design-docs/array-siblings/softmax-array-layer.md)
- [an array-based dropout sibling](docs/design-docs/array-siblings/dropout-array-layer.md) —
  including the Rust core's first RNG primitive
- [an array-based binary cross-entropy
  sibling](docs/design-docs/array-siblings/binary-cross-entropy-array-layer.md) — the deferred
  real-MNIST retune came back a genuine, if modest, win (+0.37 points over the ensemble's
  documented baseline at `learning_rate=0.1`); the Rust stage needed no new Rust primitive at all

#### ensemble (`docs/design-docs/ensemble/`)

- [an array-based ensemble sibling](docs/design-docs/ensemble/ensemble-array-layer.md) — both
  backends, unconditionally: vectorizing this codebase's own best-performing, production-facing
  real-MNIST capability (96.01% held-out accuracy) down to 21.4s (83x faster), same accuracy

#### infrastructure (`docs/design-docs/infra/`)

- [a learning-rate schedule](docs/design-docs/infra/learning-rate-schedule.md) — the
  `lr_schedule.linear_warmup` design and the widened, schedule-accepting `learning_rate` parameter
- [dataset sourcing proposal](docs/design-docs/infra/dataset-sourcing-proposal.md) — the
  checksum-verified fetch mechanism and CI-side caching `scripts/fetch_datasets.py` implements

### proposals (`docs/proposals/`)

Written up front, per this repo's own practice.

- [an array-based convolutional layer](docs/proposals/conv-array-layer.md) — not started;
  vectorizing conv layers to finally afford the real-MNIST-scale validation the per-node path's
  ~30-minute-per-run cost has left unrun
