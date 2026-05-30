"""Stochastic recovery of near-boundary policy-gradient signals.

In clipping-based GRPO/PPO objectives the policy-gradient loss is the
pessimistic maximum of an unclipped and a clipped term::

    pg_unclipped = -ratio * advantage
    pg_clipped   = -clamp(ratio, 1 - eps_low, 1 + eps_high) * advantage
    pg           = max(pg_unclipped, pg_clipped)

For every token whose importance ratio falls *outside* the clip window the
clipped term wins and ``clamp`` has zero gradient there, so the token's
contribution to the policy gradient is hard-discarded. Tokens that sit only
slightly beyond the boundary still carry informative signal, and throwing all
of it away is the "clipping bottleneck".

This module implements **Near-boundary Stochastic Rescue (NSR)**: for the
clipped tokens that lie within a thin band just beyond the clip edge, the
unclipped (gradient-bearing) loss is stochastically retained with probability
``rescue_prob``. Rescue is *boundary-local* — tokens far outside the window are
never rescued — and *stochastic* rather than a deterministic gradient rescale,
which the source paper finds consistently more effective.

Adapted from "Clipping Bottleneck: Stabilizing RLVR via Stochastic Recovery of
Near-Boundary Signals" (arXiv:2605.22703). Contributed via Remyx Recommendation
(https://engine.remyx.ai).
"""

from __future__ import annotations

import torch

__all__ = ["near_boundary_mask", "near_boundary_stochastic_rescue"]


def near_boundary_mask(
    ratio: torch.Tensor,
    pg_losses_unclipped: torch.Tensor,
    pg_losses_clipped: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    boundary_margin: float,
) -> torch.Tensor:
    """Boolean mask of tokens that are clipped *and* near the clip boundary.

    A token qualifies when the clipped term is the one being applied (i.e.
    clipping actually took effect) and its importance ratio sits within a
    relative ``boundary_margin`` band immediately beyond the clip edge:

        upper edge: (1 + eps_clip_high, (1 + eps_clip_high) * (1 + margin)]
        lower edge: [(1 - eps_clip) * (1 - margin), 1 - eps_clip)

    Args:
        ratio: importance ratio ``pi_new / pi_old`` per token.
        pg_losses_unclipped: ``-ratio * advantage``.
        pg_losses_clipped: pessimistic ``max(unclipped, clipped)`` loss.
        eps_clip: lower clip range (the ``1 - eps_clip`` edge).
        eps_clip_high: upper clip range (the ``1 + eps_clip_high`` edge).
        boundary_margin: relative width of the near-boundary band (> 0).

    Returns:
        Boolean tensor, ``True`` where a token is rescue-eligible.
    """
    # Clipping is "active" exactly where the clipped term dominates the loss.
    clip_active = pg_losses_clipped > pg_losses_unclipped

    upper = 1.0 + eps_clip_high
    lower = 1.0 - eps_clip
    near_upper = (ratio > upper) & (ratio <= upper * (1.0 + boundary_margin))
    near_lower = (ratio < lower) & (ratio >= lower * (1.0 - boundary_margin))
    return clip_active & (near_upper | near_lower)


def near_boundary_stochastic_rescue(
    ratio: torch.Tensor,
    pg_losses_unclipped: torch.Tensor,
    pg_losses_clipped: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    rescue_prob: float,
    boundary_margin: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Stochastically retain unclipped signal for near-boundary clipped tokens.

    For each rescue-eligible token (see :func:`near_boundary_mask`) the
    gradient-bearing ``pg_losses_unclipped`` value is substituted for the
    hard-clipped value with probability ``rescue_prob``; all other tokens keep
    the standard pessimistic clipped loss, so the result is identical to plain
    hard clipping when nothing is rescued.

    Args:
        ratio: importance ratio per token.
        pg_losses_unclipped: ``-ratio * advantage`` (gradient flows).
        pg_losses_clipped: pessimistic clipped loss (zero gradient past edge).
        eps_clip / eps_clip_high: clip window edges.
        rescue_prob: per-token probability of rescuing an eligible token.
        boundary_margin: relative width of the near-boundary band.
        generator: optional RNG for reproducible sampling.

    Returns:
        Loss tensor with near-boundary signals stochastically rescued.
    """
    eligible = near_boundary_mask(
        ratio,
        pg_losses_unclipped,
        pg_losses_clipped,
        eps_clip,
        eps_clip_high,
        boundary_margin,
    )
    draws = torch.rand(ratio.shape, device=ratio.device, dtype=ratio.dtype, generator=generator)
    rescue = eligible & (draws < rescue_prob)
    return torch.where(rescue, pg_losses_unclipped, pg_losses_clipped)
