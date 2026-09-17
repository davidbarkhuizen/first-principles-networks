# benchmarking

[← back to README](../README.md)

**Note:** this is the intended future home for benchmarking-related information in this
codebase - wall-clock measurements, speedup tables, sweep methodology, and the infrastructure
gaps found while running them. Nothing has been moved here yet: that content currently lives
scattered across [goals and strategy](goals-and-strategy.md#measurement-discipline-the-per-node-paths-two-jobs-and-the-one-it-doesnt-have)'s
own "measurement discipline" section, [research and analysis](research-and-analysis.md),
[research: Rust performance](research-rust-performance.md), [research: backprop
siblings](research-backprop-siblings.md), [structure](structure.md#possible-next-steps)'s own
"audit of hand-rolled measurement scripts" item, and each array-layer sibling's own "measurement
plan"/"wall-clock" section (e.g. [an array-based dropout
sibling](dropout-array-layer.md#measurement-plan)). Nothing there is being duplicated or
relocated by this note alone - existing content stays where it is, cross-linked as it always has
been, until it's deliberately migrated.

Going forward, new benchmarking-related information (methodology notes, reusable measurement
infrastructure, and cross-cutting findings that don't belong to one specific sibling's own doc)
should accumulate here instead of being scattered further - see [structure](structure.md#possible-next-steps)'s
own "audit of hand-rolled measurement scripts for reusable infrastructure" item for the concrete
occasion this page exists to eventually consolidate.
