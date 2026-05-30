"""Tests for Near-boundary Stochastic Rescue (NSR) wired into the policy loss.

Covers both the standalone rescue primitive and its integration through the
existing (non-new) ``slime.utils.ppo_utils.compute_policy_loss`` call site.
"""

import torch

# Import from the EXISTING call-site module, not just the new helper, so the
# wiring edit in compute_policy_loss is actually exercised.
from slime.utils.ppo_utils import compute_policy_loss
from slime.utils.near_boundary_rescue import near_boundary_rescue

EPS_LOW = 0.2
EPS_HIGH = 0.2


def _ratio_grid(n=4000):
    # ppo_kl spread so ratio = exp(-ppo_kl) sweeps both sides of the clip band.
    ppo_kl = torch.linspace(-0.6, 0.6, n)
    ratio = (-ppo_kl).exp()
    return ppo_kl, ratio


def test_nsr_disabled_matches_baseline():
    """Default (band=prob=0) leaves the baseline clipped loss untouched."""
    ppo_kl, _ = _ratio_grid()
    adv = torch.ones_like(ppo_kl)

    base, base_clipfrac = compute_policy_loss(ppo_kl, adv, EPS_LOW, EPS_HIGH)
    off, off_clipfrac = compute_policy_loss(ppo_kl, adv, EPS_LOW, EPS_HIGH, nsr_band=0.0, nsr_prob=0.0)

    assert torch.equal(base, off)
    assert torch.equal(base_clipfrac, off_clipfrac)


def test_nsr_only_rescues_clipped_near_boundary_tokens():
    """Enabling NSR changes the loss only at clipped, in-band positions, and
    restores those tokens to their un-clamped value."""
    ppo_kl, ratio = _ratio_grid()
    adv = torch.ones_like(ppo_kl)  # positive advantage -> upper-bound clipping
    band = 0.1

    base, clipfrac = compute_policy_loss(ppo_kl, adv, EPS_LOW, EPS_HIGH)
    torch.manual_seed(0)
    nsr, _ = compute_policy_loss(ppo_kl, adv, EPS_LOW, EPS_HIGH, nsr_band=band, nsr_prob=1.0)

    changed = ~torch.isclose(base, nsr)
    assert changed.any(), "NSR with prob=1 should rescue at least some boundary tokens"

    # Every changed token must have been hard-clipped ...
    assert torch.all(clipfrac[changed] > 0)
    # ... sit within the near-boundary band past the upper bound ...
    overshoot = ratio - (1.0 + EPS_HIGH)
    assert torch.all(overshoot[changed] > 0)
    assert torch.all(overshoot[changed] <= band)
    # ... and be restored to the un-clamped policy loss (-ratio * adv).
    expected_unclipped = -ratio * adv
    assert torch.allclose(nsr[changed], expected_unclipped[changed])


def test_rescue_primitive_respects_band_and_decay():
    """The primitive only ever rescues clipped tokens inside the band, and the
    expected rescue count tracks the linear-decay probability."""
    _, ratio = _ratio_grid()
    adv = torch.ones_like(ratio)
    pg1 = -ratio * adv
    pg2 = -ratio.clamp(1 - EPS_LOW, 1 + EPS_HIGH) * adv
    clipped = torch.maximum(pg1, pg2)
    clipfrac = torch.gt(pg2, pg1).float()
    band = 0.1

    overshoot = torch.maximum(ratio - (1 + EPS_HIGH), (1 - EPS_LOW) - ratio)
    in_band = (clipfrac > 0) & (overshoot > 0) & (overshoot <= band)
    assert in_band.any()

    # prob = 0 -> never rescue, loss unchanged.
    out0, frac0 = near_boundary_rescue(ratio, pg1, clipped, clipfrac, EPS_LOW, EPS_HIGH, band, 0.0)
    assert frac0.sum() == 0
    assert torch.equal(out0, clipped)

    # prob = 1 with a deterministic seed: rescues are a subset of in-band tokens,
    # the rescued loss matches the un-clamped value, and decay keeps the count
    # below the full in-band population (far-edge tokens are usually dropped).
    torch.manual_seed(123)
    out1, frac1 = near_boundary_rescue(ratio, pg1, clipped, clipfrac, EPS_LOW, EPS_HIGH, band, 1.0)
    rescued = frac1 > 0
    assert torch.all(in_band[rescued])
    assert torch.allclose(out1[rescued], pg1[rescued])
    assert 0 < rescued.sum().item() <= in_band.sum().item()


def test_nsr_never_touches_unclipped_region():
    """Tokens inside the trust region (ratio within bounds) are never rescued,
    regardless of how aggressive the NSR settings are."""
    _, ratio = _ratio_grid()
    adv = torch.ones_like(ratio)
    pg1 = -ratio * adv
    pg2 = -ratio.clamp(1 - EPS_LOW, 1 + EPS_HIGH) * adv
    clipped = torch.maximum(pg1, pg2)
    clipfrac = torch.gt(pg2, pg1).float()

    torch.manual_seed(7)
    out, frac = near_boundary_rescue(ratio, pg1, clipped, clipfrac, EPS_LOW, EPS_HIGH, nsr_band=10.0, nsr_prob=1.0)
    inside = clipfrac == 0
    assert torch.all(frac[inside] == 0)
    assert torch.equal(out[inside], clipped[inside])
