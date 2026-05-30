"""Robust gating of group-normalized advantages.

Adapted from "Skill-Conditioned Gated Self-Distillation for LLM Reasoning"
(SGSD, https://arxiv.org/abs/2605.28791v1). The paper's full method turns a
sparse verifier outcome into dense token-level supervision by validating
skill-conditioned teacher hypotheses and then applying a *robust gated
objective* that "distills informative teacher-student disagreements while
suppressing uncertain or extreme signals."

We deliver that gating insight at the place this inference/rollout repo can
actually host it: the GRPO group-advantage stage. After per-group standard
normalization (``a = (r - mean) / std``), the standardized advantage is the
disagreement signal we keep or discard:

  * **uncertain** signals sit near the group mean (``|a|`` small) — low
    information, so a deadband soft-threshold pushes them toward zero;
  * **extreme** signals are outliers (``|a|`` large) that destabilize the
    update — their excess magnitude is decayed back toward the informative
    band;
  * the **informative** middle band passes through, shifted by the deadband.

This keeps the multi-teacher / token-level distillation machinery (which
needs a trainer this repo does not host) out of scope while delivering the
training-stability result the team cares about as pure reward shaping.
"""

from __future__ import annotations

from collections.abc import Sequence

# Defaults expressed in units of group standard deviation, since the input is
# already standardized. ~0.5 sigma of deadband kills near-mean noise; signals
# beyond ~2.5 sigma are treated as extreme and their excess is decayed.
DEFAULT_DEADBAND = 0.5
DEFAULT_CLIP = 2.5
DEFAULT_EXTREME_DECAY = 0.5


def gate_group_advantages(
    advantages: Sequence[float],
    valid_mask: Sequence[bool] | None = None,
    *,
    deadband: float = DEFAULT_DEADBAND,
    clip: float = DEFAULT_CLIP,
    extreme_decay: float = DEFAULT_EXTREME_DECAY,
) -> list[float]:
    """Apply the SGSD-style robust gate to standardized group advantages.

    Args:
        advantages: standardized advantages for one group (``(r - mean) / std``).
        valid_mask: optional per-item flags; ``False`` entries are passed through
            untouched (e.g. out-of-range penalty rewards that were never
            standardized). Defaults to gating every entry.
        deadband: magnitude (in sigma) below which a signal is deemed uncertain
            and soft-thresholded toward zero.
        clip: magnitude (in sigma) above which a signal is deemed extreme; the
            excess beyond ``clip`` (measured after the deadband shift) is decayed.
        extreme_decay: multiplier in ``[0, 1]`` applied to the excess magnitude of
            extreme signals. ``0`` hard-caps at the band edge; ``1`` disables
            extreme suppression.

    Returns:
        A new list of gated advantages, same length as ``advantages``.
    """
    if deadband < 0:
        raise ValueError(f"deadband must be non-negative, got {deadband}")
    if clip <= deadband:
        raise ValueError(f"clip ({clip}) must be greater than deadband ({deadband})")
    if not 0.0 <= extreme_decay <= 1.0:
        raise ValueError(f"extreme_decay must be in [0, 1], got {extreme_decay}")

    if valid_mask is None:
        valid_mask = [True] * len(advantages)

    band = clip - deadband  # width of the informative band after the deadband shift
    gated: list[float] = []
    for advantage, valid in zip(advantages, valid_mask):
        if not valid:
            gated.append(advantage)
            continue

        sign = 1.0 if advantage >= 0 else -1.0
        mag = abs(advantage)

        # Suppress uncertain (near-mean) signals.
        mag = max(0.0, mag - deadband)

        # Suppress extreme (outlier) signals by decaying their excess magnitude.
        if mag > band:
            mag = band + (mag - band) * extreme_decay

        gated.append(sign * mag)

    return gated


def gating_stats(before: Sequence[float], after: Sequence[float]) -> dict[str, float]:
    """Lightweight diagnostics for a gated group, for logging / tests.

    Reports how many signals were fully suppressed (driven to zero) and the
    fraction of total advantage magnitude retained after gating.
    """
    suppressed = sum(1 for a, b in zip(before, after) if a != 0.0 and b == 0.0)
    total_before = sum(abs(a) for a in before)
    total_after = sum(abs(b) for b in after)
    retained = (total_after / total_before) if total_before > 0 else 1.0
    return {
        "num_suppressed": float(suppressed),
        "magnitude_retained": retained,
    }
