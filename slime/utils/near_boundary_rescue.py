"""Near-boundary Stochastic Rescue (NSR) for GRPO/PPO-style clipping.

Adapted from "Clipping Bottleneck: Stabilizing RLVR via Stochastic Recovery
of Near-Boundary Signals" (https://arxiv.org/abs/2605.22703v1).

Hard clipping in GRPO/PPO objectives sets the policy gradient to zero for every
token whose importance ratio falls outside ``[1 - eps_clip, 1 + eps_clip_high]``.
The paper observes that informative signals frequently live just *beyond* that
boundary, and that discarding all of them is a practical bottleneck for training
stability. NSR is a minimal, plug-and-play fix: tokens that sit in a thin band
immediately past the clip threshold are *stochastically* kept, restoring their
un-clamped gradient instead of throwing it away.

This module exposes the rescue decision as pure tensor ops so it can be dropped
into an existing clipped policy-loss call site (see
``slime.utils.ppo_utils.compute_policy_loss``).
"""

import torch

__all__ = ["near_boundary_rescue"]


def near_boundary_rescue(
    ratio: torch.Tensor,
    pg_losses_unclipped: torch.Tensor,
    pg_losses_clipped: torch.Tensor,
    clipfrac: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    nsr_band: float,
    nsr_prob: float,
):
    """Stochastically rescue tokens that sit just beyond the clip boundary.

    Args:
        ratio: Per-token importance ratio ``exp(logp - old_logp)``.
        pg_losses_unclipped: ``-ratio * advantages`` (the signal hard clipping
            discards once a token leaves the trust region).
        pg_losses_clipped: The hard-clipped policy loss actually used by the
            baseline objective (e.g. ``max(unclipped, clamped)``).
        clipfrac: Per-token indicator (float ``0/1``) of where hard clipping is
            currently active, i.e. where the gradient is being killed.
        eps_clip: Lower clip range (boundary at ``1 - eps_clip``).
        eps_clip_high: Upper clip range (boundary at ``1 + eps_clip_high``).
        nsr_band: Width of the near-boundary band (in ratio units) that is
            eligible for rescue. Only tokens within this distance past the
            violated bound can be recovered.
        nsr_prob: Maximum rescue probability, applied right at the boundary and
            decaying linearly to zero at the far edge of the band.

    Returns:
        Tuple ``(pg_losses, rescue_frac)`` where ``pg_losses`` is the policy
        loss with rescued tokens restored to their un-clamped value, and
        ``rescue_frac`` is the per-token float mask of rescued tokens (averaging
        it yields the fraction of tokens rescued, useful as a training metric).
    """
    # Signed distance past the nearest violated boundary. Above the upper bound
    # the overshoot is ``ratio - (1 + eps_clip_high)``; below the lower bound it
    # is ``(1 - eps_clip) - ratio``. At most one of these is positive.
    upper_overshoot = ratio - (1.0 + eps_clip_high)
    lower_overshoot = (1.0 - eps_clip) - ratio
    overshoot = torch.maximum(upper_overshoot, lower_overshoot)

    # Eligible == hard clipping is active *and* the token is inside the band.
    clipped = clipfrac > 0
    in_band = clipped & (overshoot > 0) & (overshoot <= nsr_band)

    # Rescue probability decays linearly with distance from the boundary, so the
    # rescue stays boundary-local as the paper prescribes.
    decay = (1.0 - overshoot / nsr_band).clamp(min=0.0, max=1.0)
    rescue_prob = nsr_prob * decay

    draw = torch.rand_like(ratio)
    rescue = in_band & (draw < rescue_prob)

    rescue_frac = rescue.to(pg_losses_clipped.dtype)
    pg_losses = torch.where(rescue, pg_losses_unclipped, pg_losses_clipped)
    return pg_losses, rescue_frac
