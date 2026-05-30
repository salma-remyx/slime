"""Tests for Near-boundary Stochastic Rescue (NSR) wired into the policy loss.

Exercises the integration into the existing call site
``slime.utils.ppo_utils.compute_policy_loss`` plus the standalone rescue
helper in ``slime.utils.clip_signal_rescue``.
"""

import torch

# torch.compile decorates compute_policy_loss; fall back to eager if the CI
# image has no C compiler so the integration assertions still run.
import torch._dynamo

torch._dynamo.config.suppress_errors = True

# Import from the EXISTING (non-new) call-site module to prove integration.
from slime.utils.ppo_utils import compute_policy_loss
from slime.utils.clip_signal_rescue import near_boundary_mask, near_boundary_stochastic_rescue

EPS = 0.2  # clip window -> [0.8, 1.2]


def _ppo_kl_for_ratio(ratio: torch.Tensor) -> torch.Tensor:
    # ratio = exp(-ppo_kl)  =>  ppo_kl = -log(ratio)
    return -torch.log(ratio)


def test_nsr_off_matches_hard_clipping():
    """rescue_prob=0 / margin=0 must be byte-identical to plain hard clipping."""
    ratio = torch.tensor([1.25, 1.5, 0.7, 1.0])
    adv = torch.tensor([1.0, 1.0, -1.0, 1.0])
    ppo_kl = _ppo_kl_for_ratio(ratio)

    base_loss, base_clipfrac = compute_policy_loss(ppo_kl, adv, EPS, EPS)
    off_loss, off_clipfrac = compute_policy_loss(
        ppo_kl, adv, EPS, EPS, nsr_rescue_prob=0.0, nsr_boundary_margin=0.1
    )
    # margin>0 but prob=0 -> still a no-op
    off2_loss, _ = compute_policy_loss(
        ppo_kl, adv, EPS, EPS, nsr_rescue_prob=0.5, nsr_boundary_margin=0.0
    )

    torch.testing.assert_close(off_loss, base_loss)
    torch.testing.assert_close(off2_loss, base_loss)
    torch.testing.assert_close(off_clipfrac, base_clipfrac)


def test_nsr_rescues_near_boundary_token_through_call_site():
    """A clipped token just past the edge gets its unclipped loss back."""
    # ratio 1.25 is just past the 1.2 upper edge; within a 10% band (<=1.32).
    ratio = torch.tensor([1.25])
    adv = torch.tensor([1.0])
    ppo_kl = _ppo_kl_for_ratio(ratio)

    base_loss, _ = compute_policy_loss(ppo_kl, adv, EPS, EPS)
    # prob=1.0 -> deterministic rescue, no RNG dependence.
    nsr_loss, _ = compute_policy_loss(
        ppo_kl, adv, EPS, EPS, nsr_rescue_prob=1.0, nsr_boundary_margin=0.1
    )

    # Hard clipping caps the loss at -1.2; NSR restores the unclipped -1.25.
    torch.testing.assert_close(base_loss, torch.tensor([-1.2]))
    torch.testing.assert_close(nsr_loss, torch.tensor([-1.25]))


def test_nsr_does_not_rescue_far_out_of_bound():
    """Tokens beyond the boundary band stay hard-clipped (boundary-local)."""
    ratio = torch.tensor([1.5])  # 1.5 > 1.2 * 1.1 = 1.32 -> outside the band
    adv = torch.tensor([1.0])
    ppo_kl = _ppo_kl_for_ratio(ratio)

    base_loss, _ = compute_policy_loss(ppo_kl, adv, EPS, EPS)
    nsr_loss, _ = compute_policy_loss(
        ppo_kl, adv, EPS, EPS, nsr_rescue_prob=1.0, nsr_boundary_margin=0.1
    )
    torch.testing.assert_close(nsr_loss, base_loss)


def test_near_boundary_mask_selects_only_active_near_clipped():
    ratio = torch.tensor([1.25, 1.5, 1.0, 0.75])
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0])
    pg1 = -ratio * adv
    pg2 = -ratio.clamp(1 - EPS, 1 + EPS) * adv
    pg_clipped = torch.maximum(pg1, pg2)

    mask = near_boundary_mask(ratio, pg1, pg_clipped, EPS, EPS, boundary_margin=0.1)
    # 1.25: near upper & clipped -> True
    # 1.5 : clipped but outside band -> False
    # 1.0 : not clipped -> False
    # 0.75: ratio<0.8 (within band of 0.72) and adv<0 -> clip active -> True
    assert mask.tolist() == [True, False, False, True]


def test_stochastic_rescue_fraction_is_calibrated():
    """With many eligible tokens, rescued fraction tracks rescue_prob."""
    n = 20000
    ratio = torch.full((n,), 1.25)  # all near-boundary
    adv = torch.ones(n)
    pg1 = -ratio * adv
    pg2 = -ratio.clamp(1 - EPS, 1 + EPS) * adv
    pg_clipped = torch.maximum(pg1, pg2)

    gen = torch.Generator().manual_seed(0)
    rescued = near_boundary_stochastic_rescue(
        ratio, pg1, pg_clipped, EPS, EPS, rescue_prob=0.5, boundary_margin=0.1, generator=gen
    )
    frac = (rescued == pg1).float().mean().item()
    assert 0.47 < frac < 0.53
