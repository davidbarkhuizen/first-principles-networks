# indrajala-ml

A small, dependency-light implementation of two classifier families, built from first
principles. `LinearClassifierNetwork` composes state (input) nodes and association
(weighted, thresholded) nodes/layers per Rosenblatt's perceptron (1958), trained with the
classic perceptron learning rule (and a MADALINE-style minimum-disturbance rule once more
than one hidden node is used). `BackpropClassifierNetwork` is a sigmoid, gradient-descent
network of arbitrary depth, added alongside it - a genuinely different learning rule, not a
retrofit - so it can represent targets (like XOR) the discrete model structurally can't.

## docs

- [goals and strategy](docs/goals-and-strategy.md) — what this project is actually for, and how to
  evaluate a proposed addition
- [structure](docs/structure.md) — module layout, and how each network composes, including every
  `BackpropClassifierNetwork` sibling (Adam included) and which are adopted vs. kept as tested
  nulls
- [setup](docs/setup.md) — requirements, install, running the tests, fetching real MNIST/UCI
  digits from dedicated, checksum-verified `indrajala-datasets-*` repos, and
  [CI](.github/workflows/ci.yml), which runs the full test suite on every push/PR
- [demos](docs/demos.md) — the demo scripts and what each one shows
- [theory](docs/theory.md) — Rosenblatt's perceptron theory, and reference material
- [research and analysis](docs/research-and-analysis.md) — investigations behind a design
  decision, with the measurements that drove it, including every array/Rust-matmul-backed sibling
  port (Adam, momentum, L2, ReLU, softmax, dropout) and its own wall-clock/accuracy result
- [benchmarking](docs/benchmarking.md) — reusable measurement infrastructure (a fixed MNIST-proxy
  builder, a fork-based sweep runner, wall-clock calibration, and mean/stdev aggregation) for the
  hand-rolled measurement scripts every real-scale sweep in the docs above used to re-roll from
  scratch
- [vectorization](docs/vectorization.md) — numpy-backed array classes (kept as a permanent
  benchmarking mirror) and a hand-built Rust array core (the production backend)
- [Rust production cutover](docs/rust-production-cutover.md) — how the Rust core became primary
  in production: a debug-build fix, a matmul reorder, fused per-layer calls, and matmul SIMD
  work, landing on a Rust core that's **faster than numpy** on real training runs - currently
  ~3.6x at UCI digits scale, ~2.6x at real MNIST scale (3.40x/1.31x at first cutover, improved by
  later matmul SIMD work - see [research and analysis](docs/research-and-analysis.md))
- [mini-batch gradient descent](docs/mini-batch-gradient-descent.md) — batched gradient updates,
  and the momentum re-test they unblocked, plus the learning-rate warmup schedule that later fixed
  the large-batch divergence they surfaced
- [convolutional layers](docs/convolutional-layers.md) — a from-scratch conv layer, local
  receptive fields and weight sharing, measured against the dense baseline on UCI digits
- [dropout](docs/dropout.md) — stochastic hidden-unit regularization, built and measured (via its
  own array-based sibling): a reconfirmed null, the same as L2's own finding
- [an array-based binary cross-entropy sibling](docs/binary-cross-entropy-array-layer.md) —
  proposed, not started; depends on the array-based ensemble sibling's own single-output array
  network
- [an array-based ensemble sibling](docs/ensemble-array-layer.md) — proposed, not started:
  vectorizing this codebase's own best-performing, production-facing real-MNIST capability
  (96.01% held-out accuracy), which has never been vectorized at all
- [an array-based convolutional layer](docs/conv-array-layer.md) — proposed, not started:
  vectorizing conv layers to finally afford the real-MNIST-scale validation the per-node path's
  ~30-minute-per-run cost has left unrun

## design docs

Finished workplans, trimmed down to the design/API reference each one's own implementation still
cites in its docstrings, once their actual measured results were folded into
[research and analysis](docs/research-and-analysis.md) (or, for dataset sourcing,
[setup](docs/setup.md#mnist-data)) as the permanent record of what was found:

- [Adam optimizer](docs/adam-optimizer.md) — `AdamBackpropClassifierNetwork`'s per-parameter
  adaptive-learning-rate design
- [an array-based Adam sibling](docs/adam-array-layer.md) — the numpy/Rust-matmul-backed port
- [an array-based momentum sibling](docs/momentum-array-layer.md) — the numpy/Rust-matmul-backed
  port
- [an array-based L2 sibling](docs/l2-array-layer.md) — the numpy/Rust-matmul-backed port
- [an array-based ReLU sibling](docs/relu-array-layer.md) — the numpy/Rust-matmul-backed port
- [an array-based softmax sibling](docs/softmax-array-layer.md) — the numpy/Rust-matmul-backed
  port
- [an array-based dropout sibling](docs/dropout-array-layer.md) — the numpy/Rust-matmul-backed
  port, including the Rust core's first RNG primitive
- [a learning-rate schedule](docs/learning-rate-schedule.md) — the `lr_schedule.linear_warmup`
  design and the widened, schedule-accepting `learning_rate` parameter
- [dataset sourcing proposal](docs/dataset-sourcing-proposal.md) — the checksum-verified fetch
  mechanism and CI-side caching `scripts/fetch_datasets.py` implements
