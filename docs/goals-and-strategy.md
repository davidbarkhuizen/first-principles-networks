# goals and strategy

[← back to README](../README.md)

This document exists because it's easy to let a project like this drift toward "add more
capability" as a default, without ever asking what the capability is *for*. It states plainly
what this project is trying to be, and how to tell the difference when a new piece of work is
proposed.

## what this project actually is

A from-scratch, hand-derived implementation of two classifier families (Rosenblatt's perceptron/
MADALINE, and backprop), extended over time with siblings (softmax, ReLU, momentum, L2, conv,
ensemble) and a from-scratch Rust array core - each addition built and *measured*, not assumed,
with the investigation behind it kept alongside the code
([research and analysis](research-and-analysis.md)).

Its real value is in that rigor, not in the model capability itself: every non-obvious decision
here has a documented measurement behind it, including the ones that didn't pan out - momentum
and L2 regularization are still in the codebase as real, tested capabilities despite measuring as
nulls on every scenario tried, because reporting a clean null honestly is itself the point, not a
failure to hide. The Rust array core's own history is the same pattern at a different scale: a
debug-build artifact that made the crate look 65-335x slower than numpy before anyone thought to
check the build flag, a matmul gap that was assumed not to matter at `batch_size=1` and turned out
to be 97% of a fused call's cost once actually measured - caught because the practice here is to
measure before concluding, not after.

That combination - genuine first-principles construction plus an honest, checked record of what
worked and what didn't - is the asset. It's pedagogical and demonstrative value: a place to
actually understand how these algorithms work, including the numerical subtleties (float64
summation order, RNG non-reproducibility across backends, IEEE-754 FMA semantics) that using a
real framework would hide.

## what success looks like here

- **Every addition has a real measurement behind it**, positive or negative, recorded honestly in
  [research and analysis](research-and-analysis.md) - not "this should help" left unchecked.
- **The docs stay in sync with the code.** A reader should be able to trust `structure.md`'s
  description of current state without cross-checking the source - a standing practice, not a
  one-time cleanup, the same way the matmul SIMD work's own docs got revisited and corrected once
  the code moved past what they said (see [research and
  analysis](research-and-analysis.md#simd-for-the-matvec-production-path-a-bigger-win-than-the-batch32-work-it-followed)).
- **New capability is added the way existing capability was**: hand-derived, sibling-not-retrofit,
  parity-tested against a reference, with the investigation kept legible. Speed of arrival matters
  less than whether the result can be trusted.
- **The project stays a place someone could actually learn from** - by reading the code, or by
  reading the research-and-analysis trail of how a decision was reached.

## strategy: what gets prioritized

Work that deepens the first-principles rigor this project already has, in roughly this order of
fit:

1. **New model primitives built and measured the existing way** - each as a hand-derived sibling,
   checked against a reference the way every existing sibling was, with the measurement (does it
   actually help, on what, by how much) written up regardless of outcome.
2. **Deepening what's already here** - extending an existing mechanism (the conv layer, an
   open measurement question already on record) rather than adding a structurally new one.
3. **Infrastructure that removes friction from doing (1) and (2) honestly** - protects the rigor
   (a change can't silently break the numbers a decision was based on, a measurement stays
   reproducible from a fresh checkout).

See [structure](structure.md#possible-next-steps) for the concrete, current candidates in each
tier - kept there, alongside the architecture they'd extend, rather than duplicated here where the
two lists could drift out of sync with each other.

## how to evaluate a proposed addition

Before starting work on something new, it should be able to answer:

- **Does this deepen understanding, or just add capability?** Implementing Adam by hand and
  measuring it against SGD deepens understanding. Wrapping `torch.optim.Adam` would add capability
  without any of the value this project actually provides.
- **Can it be measured, and will the measurement be recorded honestly either way?** If a proposed
  addition can't be checked against a reference or a baseline, it doesn't fit this project's own
  practice, regardless of how useful it sounds.
- **Would explaining it in [research and analysis](research-and-analysis.md) actually be
  interesting to read?** If the honest write-up would just be "we added X because other libraries
  have X," that's a signal it doesn't belong here.
