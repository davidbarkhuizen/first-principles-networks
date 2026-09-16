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
- [structure](docs/structure.md) — module layout, and how each network composes
- [setup](docs/setup.md) — requirements, install, and running the tests
- [dataset sourcing proposal](docs/dataset-sourcing-proposal.md) — fetching real MNIST/UCI digits
  from dedicated, checksum-verified `indrajala-datasets-*` repos, now built and shipped; unblocked
  [CI](.github/workflows/ci.yml), which now runs the full test suite on every push/PR
- [demos](docs/demos.md) — the demo scripts and what each one shows
- [theory](docs/theory.md) — Rosenblatt's perceptron theory, and reference material
- [research and analysis](docs/research-and-analysis.md) — investigations behind a design
  decision, with the measurements that drove it
- [vectorization](docs/vectorization.md) — numpy-backed array classes (kept as a permanent
  benchmarking mirror) and a hand-built Rust array core (the production backend)
- [Rust production cutover](docs/rust-production-cutover.md) — how the Rust core became primary
  in production: a debug-build fix, a matmul reorder, fused per-layer calls, and matmul SIMD
  work, landing on a Rust core that's **faster than numpy** on real training runs - currently
  ~3.6x at UCI digits scale, ~2.6x at real MNIST scale (3.40x/1.31x at first cutover, improved by
  later matmul SIMD work - see [research and analysis](docs/research-and-analysis.md))
- [mini-batch gradient descent](docs/mini-batch-gradient-descent.md) — batched gradient updates,
  and the momentum re-test they unblocked
- [convolutional layers](docs/convolutional-layers.md) — a from-scratch conv layer, local
  receptive fields and weight sharing, measured against the dense baseline on UCI digits
- [Adam optimizer](docs/adam-optimizer.md) — a per-parameter adaptive-learning-rate sibling, built
  and measured: needs its own retuned learning rate like every other sibling here, but once
  retuned it's a real, substantial win under large-batch training, at proxy and real-MNIST scale
- [a learning-rate schedule](docs/learning-rate-schedule.md) — warmup, built and measured: fixes
  the documented `batch_size=128` divergence, most cleanly paired with momentum
- [an array-based Adam sibling](docs/adam-array-layer.md) — a Rust-matmul-backed Adam, built and
  measured: both array-based backends 70x-794x faster per example than the per-node path,
  batch-size robustness confirmed to survive the port
- [dropout](docs/dropout.md) — stochastic hidden-unit regularization, proposed and not yet started
