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
- [benchmarking](docs/benchmarking.md) — proposed reusable infrastructure (a fixed MNIST-proxy
  builder, a fork-based sweep runner, wall-clock calibration, and mean/stdev aggregation) for the
  hand-rolled measurement scripts every real-scale sweep in the docs above has so far re-rolled
  from scratch
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
- [dropout](docs/dropout.md) — stochastic hidden-unit regularization, built and measured: a
  reconfirmed null, the same as L2's own finding, via its own array-based sibling below
- [an array-based momentum sibling](docs/momentum-array-layer.md) — proposed, not started:
  vectorizing momentum's own weight-update rule, to re-test its measured null at a scale/seed
  count the per-node path couldn't afford
- [an array-based L2 sibling](docs/l2-array-layer.md) — proposed, not started: the same idea for
  L2's own overfitting-gap measurement, and the escalation dropout's own workplan proposed but
  never got to run
- [an array-based ReLU sibling](docs/relu-array-layer.md) — proposed, not started: vectorizing
  ReLU's own genuine, adopted win, to check whether it holds at UCI-digits/real-MNIST scale, not
  just the toy XOR scenario it was measured on
- [an array-based softmax sibling](docs/softmax-array-layer.md) — proposed, not started:
  vectorizing softmax, to finally afford the real-MNIST learning-rate retune its own per-node
  investigation explicitly declined to run due to cost
- [an array-based dropout sibling](docs/dropout-array-layer.md) — built and measured: the direct
  fix for dropout's own abandoned, 40+-minute overfitting-gap sweep on the per-node path, now
  reran at 30-seed power (over 53x faster) - a reconfirmed null at both the original and an
  escalated scale
- [an array-based binary cross-entropy sibling](docs/binary-cross-entropy-array-layer.md) —
  proposed, not started; depends on the array-based ensemble sibling's own single-output array
  network
- [an array-based ensemble sibling](docs/ensemble-array-layer.md) — proposed, not started:
  vectorizing this codebase's own best-performing, production-facing real-MNIST capability
  (96.01% held-out accuracy), which has never been vectorized at all
- [an array-based convolutional layer](docs/conv-array-layer.md) — proposed, not started:
  vectorizing conv layers to finally afford the real-MNIST-scale validation the per-node path's
  ~30-minute-per-run cost has left unrun
