"""Integration tests for SGSD-style robust advantage gating.

Exercises the wiring in ``slime_plugins/rollout_buffer/generator/base_generator.py``
(an existing, non-new module) that calls into the new ``gated_advantage`` module.
The generator file is loaded the same way the rollout buffer loads it at runtime
(``spec_from_file_location`` with no parent package), which also covers the
standalone-import fallback added to ``base_generator``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_DIR = REPO_ROOT / "slime_plugins" / "rollout_buffer" / "generator"


def _install_stubs() -> None:
    """Stub the heavy runtime deps base_generator imports at module top level."""
    for name in ("openai", "requests", "tqdm", "aiohttp"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules["openai"], "OpenAI"):
        sys.modules["openai"].OpenAI = object
    if not hasattr(sys.modules["tqdm"], "tqdm"):
        sys.modules["tqdm"].tqdm = lambda *a, **k: None

    # Replace the reward hub (pulls in aiohttp etc.) with a tiny stub.
    rm_hub = types.ModuleType("slime.rollout.rm_hub")
    rm_hub.get_deepscaler_rule_based_reward = lambda response, label: 0.0
    sys.modules["slime.rollout.rm_hub"] = rm_hub


def _load_base_generator():
    """Load base_generator.py standalone, exactly like buffer.py does."""
    _install_stubs()
    path = GENERATOR_DIR / "base_generator.py"
    spec = importlib.util.spec_from_file_location("generator_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base_generator():
    return _load_base_generator()


def _make_group(instance_id, rewards):
    data = [{"reward": r, "messages": [{"role": "assistant", "content": "x"}]} for r in rewards]
    return (instance_id, data)


def test_gate_unit_behavior():
    # Import the new module directly to assert the core gating contract.
    sys.path.insert(0, str(GENERATOR_DIR))
    import gated_advantage as ga

    # uncertain (|a| <= deadband) -> fully suppressed to 0
    assert ga.gate_group_advantages([0.0, 0.3, -0.5])[0] == 0.0
    assert ga.gate_group_advantages([0.3])[0] == 0.0
    assert ga.gate_group_advantages([-0.5])[0] == 0.0

    # informative band -> shifted down by exactly the deadband, sign preserved
    gated = ga.gate_group_advantages([1.0, -1.0], deadband=0.5, clip=2.5, extreme_decay=0.5)
    assert gated[0] == pytest.approx(0.5)
    assert gated[1] == pytest.approx(-0.5)

    # extreme (|a| > clip) -> excess beyond the band is decayed
    # mag 4.0 -> -deadband 3.5 -> band=2.0, excess 1.5 * 0.5 = 0.75 -> 2.75
    assert ga.gate_group_advantages([4.0], deadband=0.5, clip=2.5, extreme_decay=0.5)[0] == pytest.approx(2.75)
    # full standardization (no gating) would have kept 4.0, so gating shrinks it
    assert ga.gate_group_advantages([4.0])[0] < 4.0


def test_gate_validates_params():
    sys.path.insert(0, str(GENERATOR_DIR))
    import gated_advantage as ga

    with pytest.raises(ValueError):
        ga.gate_group_advantages([1.0], deadband=-1.0)
    with pytest.raises(ValueError):
        ga.gate_group_advantages([1.0], clip=0.3, deadband=0.5)
    with pytest.raises(ValueError):
        ga.gate_group_advantages([1.0], extreme_decay=2.0)


def test_normalize_group_data_applies_gating(base_generator):
    # rewards [0,0,0,1]: mean=0.25, std=0.4330127
    #   r=0 -> standardized -0.5773502 -> gated -0.0773502
    #   r=1 -> standardized  1.7320508 -> gated  1.2320508
    group = _make_group("inst-1", [0.0, 0.0, 0.0, 1.0])
    instance_id, data = base_generator.normalize_group_data(group)

    assert instance_id == "inst-1"
    rewards = [item["reward"] for item in data]
    raw = [item["raw_reward"] for item in data]

    assert raw == [0.0, 0.0, 0.0, 1.0]
    # gating shifted every standardized magnitude down by the 0.5 deadband
    assert rewards[0] == pytest.approx(-0.0773502, abs=1e-5)
    assert rewards[3] == pytest.approx(1.2320508, abs=1e-5)
    # plain GRPO would have produced -0.5773502 / 1.7320508; gating shrinks both
    assert abs(rewards[0]) < 0.5773502
    assert abs(rewards[3]) < 1.7320508


def test_normalize_group_data_passes_through_penalties(base_generator):
    # Out-of-range penalty rewards (e.g. -1) are not standardized and must
    # survive gating untouched.
    group = _make_group("inst-2", [0.0, 1.0, -1.0])
    _, data = base_generator.normalize_group_data(group)
    rewards = [item["reward"] for item in data]
    assert rewards[2] == -1.0


def test_transform_group_hook_is_wired(base_generator):
    # The buffer discovers and calls `transform_group`; it must return a
    # (instance_id, data) pair with gated rewards.
    assert hasattr(base_generator, "transform_group")
    result = base_generator.transform_group(_make_group("inst-3", [0.0, 0.0, 1.0, 1.0]), "math")
    instance_id, data = result
    assert instance_id == "inst-3"
    # symmetric group: standardized magnitudes are ~1.0, gated to ~0.5
    mags = sorted(abs(item["reward"]) for item in data)
    assert all(m == pytest.approx(0.5, abs=1e-6) for m in mags)
