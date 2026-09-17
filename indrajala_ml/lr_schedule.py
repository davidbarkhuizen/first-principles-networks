from __future__ import annotations

from typing import Callable


def linear_warmup(target_rate: float, warmup_steps: int) -> Callable[[int], float]:
    """
    A schedule function (step index -> learning_rate) that ramps linearly from target_rate/
    warmup_steps up to target_rate over warmup_steps steps, then holds at target_rate
    thereafter - the standard fix Goyal et al. 2017 pair with the linear-scaling rule, since
    applying a batch-size-scaled rate from step one is unstable (see
    docs/design-docs/infra/learning-rate-schedule.md for the concrete failure case this targets:
    batch_size=128's fixed 54-58%-coin-flip divergence).

    step + 1 (rather than step) is deliberate: the very first call, at step=0, gets
    target_rate/warmup_steps rather than 0.0 - a literal 0.0 first step would waste one
    iteration entirely, learning nothing. See docs/design-docs/infra/learning-rate-schedule.md's own "design"
    section for why this is a pinned choice, not a derivation.
    """

    def schedule(step: int) -> float:
        return target_rate * min((step + 1) / warmup_steps, 1.0)

    return schedule
