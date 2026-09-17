from __future__ import annotations

from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


class CrossEntropyRustArrayBackpropClassifierNetwork(RustArrayBackpropClassifierNetwork):
    """
    The Rust-array-core-backed sibling of CrossEntropyArrayBackpropClassifierNetwork - see
    docs/architecture/rust-production-cutover.md's phase 1 and
    docs/design-docs/array-siblings/binary-cross-entropy-array-layer.md. Built unconditionally alongside every
    other Rust-backed sibling in this codebase, not gated behind a numpy-stage wall-clock verdict -
    per docs/architecture/rust-production-cutover.md's 2026-09-16 clarification.

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load/restore (including its
    plain-list tolerance for ensemble_train.py's multiprocessing path) are all inherited
    unchanged from RustArrayBackpropClassifierNetwork.
    """

    output_layer_cls = CrossEntropyRustArrayLayer
