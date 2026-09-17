# research and analysis

[← back to README](../README.md)

Write-ups of the investigations behind this codebase's design decisions - the measurements that
drove a choice, not just the choice itself. Unlike [structure](structure.md) (what the code does
now) or [demos](demos.md) (what each demo shows), this is where the *why* lives, condensed to the
question asked, what was measured, and the decision it produced.

Split by theme into its own sub-doc, each independently linkable by section:

## [multi-class architecture and loss functions](research-multiclass-and-loss.md)

How multi-class classification is structured (one-vs-rest vs. softmax, an ensemble of independent
classifiers vs. a shared hidden layer) and which loss function each uses.

- [parallelizing MNIST training](research-multiclass-and-loss.md#parallelizing-mnist-training)
- [softmax/cross-entropy re-alignment](research-multiclass-and-loss.md#softmaxcross-entropy-re-alignment)
- [binary cross-entropy for BackpropClassifierNetwork](research-multiclass-and-loss.md#binary-cross-entropy-for-backpropclassifiernetwork)
- [the ensemble/real-MNIST investigation](research-multiclass-and-loss.md#the-ensemblereal-mnist-investigation)
- [softmax on real full-scale MNIST](research-multiclass-and-loss.md#softmax-on-real-full-scale-mnist)
- [an array-based softmax sibling](research-multiclass-and-loss.md#an-array-based-softmax-sibling-the-retune-the-per-node-investigation-declined-to-spend-on)

## [backprop sibling measurements](research-backprop-siblings.md)

`BackpropClassifierNetwork`'s alternate-init, weight-update-rule, and activation siblings -
Xavier/Glorot init, momentum (including its mini-batch retests), ReLU, L2, and convolutional
layers.

- [Xavier/Glorot init: measured, not worth adopting](research-backprop-siblings.md#xavierglorot-init-measured-not-worth-adopting)
- [momentum: measured, not worth adopting](research-backprop-siblings.md#momentum-measured-not-worth-adopting)
- [ReLU hidden-layer activation: a clean win, once retuned](research-backprop-siblings.md#relu-hidden-layer-activation-a-clean-win-once-retuned)
- [L2 weight regularization: closes the overfitting gap, doesn't improve it](research-backprop-siblings.md#l2-weight-regularization-closes-the-overfitting-gap-doesnt-improve-it)
- [momentum under mini-batch gradients](research-backprop-siblings.md#momentum-under-mini-batch-gradients)
- [the learning-rate-vs-batch-size follow-up](research-backprop-siblings.md#the-learning-rate-vs-batch-size-follow-up-the-confound-was-real-and-momentum-still-doesnt-help)
- [the batch_size=128 divergence, retested with warmup](research-backprop-siblings.md#the-batch_size128-divergence-retested-with-warmup)
- [convolutional layers on UCI digits](research-backprop-siblings.md#convolutional-layers-on-uci-digits)
- [an array-based L2 sibling](research-backprop-siblings.md#an-array-based-l2-sibling-the-reconfirmed-null-holds-at-far-higher-power)
- [an array-based momentum sibling](research-backprop-siblings.md#an-array-based-momentum-sibling-unchanged-on-stronger-evidence)
- [an array-based ReLU sibling](research-backprop-siblings.md#an-array-based-relu-sibling-a-genuine-scale-dependent-finding)
- [an array-based dropout sibling](research-backprop-siblings.md#an-array-based-dropout-sibling-another-reconfirmed-null-now-at-real-power)

## [Rust core performance](research-rust-performance.md)

The investigation thread behind [the Rust production cutover](rust-production-cutover.md)'s
performance work - the debug-build discovery, fusing, and the `batch_size >= 32` matmul gap's
closing sequence (blocking, threading, SIMD).

- [the Rust array core's 65-335x-slower-than-numpy finding was a debug-build artifact](research-rust-performance.md#the-rust-array-cores-65-335x-slower-than-numpy-finding-was-a-debug-build-artifact)
- [phase 0b: fusing closes some further ground, but matmul is now the binding constraint](research-rust-performance.md#phase-0b-fusing-closes-some-further-ground-but-matmul-is-now-the-binding-constraint)
- [phase 2 tier 2: real per-example training is a genuine win at both scales measured](research-rust-performance.md#phase-2-tier-2-real-per-example-training-is-a-genuine-win-at-both-scales-measured)
- [build-flag tuning measured as a null](research-rust-performance.md#build-flag-tuning-measured-as-a-null)
- [cache-blocked matmul: a real, shape-dependent win, gated by size](research-rust-performance.md#cache-blocked-matmul-a-real-shape-dependent-win-gated-by-size)
- [threaded matmul: a real further win, one real bug caught, one refinement rejected](research-rust-performance.md#threaded-matmul-a-real-further-win-one-real-bug-caught-one-refinement-rejected)
- [explicit SIMD intrinsics: a real further win, with fused multiply-add kept consistent across every path](research-rust-performance.md#explicit-simd-intrinsics-a-real-further-win-with-fused-multiply-add-kept-consistent-across-every-path)
- [SIMD for the matvec production path: a bigger win than the batch>=32 work it followed](research-rust-performance.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)

## [Adam optimizer](research-adam-optimizer.md)

The measurement results for `AdamBackpropClassifierNetwork` (see
[structure](structure.md#backprop-siblings) for its design).

- [Adam: tuned-XOR measurement (stage 3)](research-adam-optimizer.md#adam-tuned-xor-measurement-stage-3-of-the-adam-optimizer-workplan)
- [Adam under batch size: a much bigger, cleaner win (stage 4)](research-adam-optimizer.md#adam-under-batch-size-a-much-bigger-cleaner-win-stage-4-of-the-adam-optimizer-workplan)
- [Adam at real-MNIST-ensemble scale: the proxy result holds (stage 5)](research-adam-optimizer.md#adam-at-real-mnist-ensemble-scale-the-proxy-result-holds-stage-5-of-the-adam-optimizer-workplan)
- [RMSprop: the second-moment term alone accounts for Adam's batch-size win (stage 7)](research-adam-optimizer.md#rmsprop-the-second-moment-term-alone-accounts-for-adams-batch-size-win-stage-7-of-the-adam-optimizer-workplan)
- [an array-based Adam sibling](research-adam-optimizer.md#an-array-based-rust-matmul-backed-adam-sibling-the-wall-clock-win-the-accuracy-result-was-missing)
