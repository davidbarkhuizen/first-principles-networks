# first-principles-networks

A small, dependency-light implementation of two classifier families, built from first
principles. `LinearClassifierNetwork` composes state (input) nodes and association
(weighted, thresholded) nodes/layers per Rosenblatt's perceptron (1958), trained with the
classic perceptron learning rule (and a MADALINE-style minimum-disturbance rule once more
than one hidden node is used). `BackpropClassifierNetwork` is a sigmoid, gradient-descent
network of arbitrary depth, added alongside it - a genuinely different learning rule, not a
retrofit - so it can represent targets (like XOR) the discrete model structurally can't.

## docs

- [structure](docs/structure.md) — module layout, and how each network composes
- [setup](docs/setup.md) — requirements, install, and running the tests
- [demos](docs/demos.md) — the demo scripts and what each one shows
- [theory](docs/theory.md) — Rosenblatt's perceptron theory, and reference material
- [research and analysis](docs/research-and-analysis.md) — investigations behind a design
  decision, with the measurements that drove it
- [vectorization](docs/vectorization.md) — numpy-backed array classes (kept as a permanent
  benchmarking mirror) and a hand-built Rust array core (the intended production backend, not
  yet wired in)
- [Rust production cutover](docs/rust-production-cutover.md) — workplan for making the Rust core
  primary in production, gated on a measured performance fix it currently needs first
- [mini-batch gradient descent](docs/mini-batch-gradient-descent.md) — batched gradient updates,
  and the momentum re-test they unblocked
- [convolutional layers](docs/convolutional-layers.md) — a from-scratch conv layer, local
  receptive fields and weight sharing, measured against the dense baseline on UCI digits
