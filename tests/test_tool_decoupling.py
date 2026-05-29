"""Tests for tool invocation/execution decoupling diagnostics.

Covers the standalone metric (``tool_decoupling_metric``) and, crucially,
the wiring edit in the existing rollout call site
(``slime_plugins.rollout_buffer.generator.base_generator.worker_process``):
every emitted rollout item must now carry a ``tool_decoupling`` block in its
``extra_info``.
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import from the NEW capability module ...
from slime_plugins.rollout_buffer.generator.tool_decoupling_metric import (
    analyze_tool_decoupling,
    decoupling_bonus,
    extract_tool_events,
)

# ... and from the EXISTING (non-new) call-site module to prove integration.
from slime_plugins.rollout_buffer.generator.base_generator import worker_process


NUM_GPUS = 0


# ---------------------------------------------------------------------------
# Standalone metric behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_tool_use_is_vacuously_decoupled():
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "It is 4."},
    ]
    m = analyze_tool_decoupling(messages)
    assert m["uses_tools"] is False
    assert m["num_invocations"] == 0
    assert m["decoupling_score"] == 1.0
    assert m["disrupted"] is False


@pytest.mark.unit
def test_immediate_execution_is_coupled():
    # OpenAI-style: assistant invokes, tool result immediately follows.
    messages = [
        {"role": "user", "content": "Compute 12 * 7."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "84"},
        {"role": "assistant", "content": "The answer is 84."},
    ]
    m = analyze_tool_decoupling(messages)
    assert m["uses_tools"] is True
    assert m["num_invocations"] == 1
    assert m["num_executions"] == 1
    assert m["num_coupled"] == 1
    assert m["num_delayed"] == 0
    assert m["decoupling_score"] == 0.0
    assert m["disrupted"] is True


@pytest.mark.unit
def test_delayed_execution_scores_high():
    # Inline format: the model invokes, keeps reasoning, then consumes the
    # result. Reasoning sits between invocation and execution => decoupled.
    messages = [
        {"role": "user", "content": "Solve the system."},
        {
            "role": "assistant",
            "content": (
                "<tool_call>solve(eq)</tool_call> "
                "Meanwhile I can reason about the structure of the solution. "
                "<tool_response>x=3</tool_response> So x = 3."
            ),
        },
    ]
    m = analyze_tool_decoupling(messages)
    assert m["num_delayed"] == 1
    assert m["num_coupled"] == 0
    assert m["decoupling_score"] == 1.0
    assert m["mean_reasoning_gap"] > 0


@pytest.mark.unit
def test_batched_invocations_are_delayed():
    # Two invocations before any execution => both batched => fully decoupled.
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
    ]
    m = analyze_tool_decoupling(messages)
    assert m["num_invocations"] == 2
    assert m["decoupling_score"] == 1.0


@pytest.mark.unit
def test_inline_tags_are_recognized():
    messages = [
        {
            "role": "assistant",
            "content": (
                "I will call a tool. <tool_call>search(x)</tool_call> "
                "<tool_response>found</tool_response> Done."
            ),
        }
    ]
    events = extract_tool_events(messages)
    kinds = [k for k, _ in events]
    assert "invoke" in kinds
    assert "exec" in kinds
    m = analyze_tool_decoupling(messages)
    assert m["uses_tools"] is True
    assert m["num_invocations"] == 1


@pytest.mark.unit
def test_decoupling_bonus_is_opt_in():
    m = analyze_tool_decoupling([{"role": "assistant", "content": "no tools"}])
    assert decoupling_bonus(m) == 0.0
    assert decoupling_bonus(m, weight=0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration: the call-site wiring in base_generator.worker_process
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_worker_process_attaches_decoupling_metric():
    """worker_process must stamp ``tool_decoupling`` onto each item's extra_info."""

    canned_messages = [
        {"role": "user", "content": "Compute something."},
        # Batched invocations (both emitted before any execution) => decoupled.
        {"role": "assistant", "content": "thinking", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
        {"role": "assistant", "content": "final answer"},
    ]

    def fake_rollout(client, prompt, sampling_params):
        return canned_messages, "stop"

    def fake_reward(item):
        return 1.0

    task_queue: queue.Queue = queue.Queue()
    done_queue: queue.Queue = queue.Queue()
    task_queue.put({"prompt": "p", "instance_id": "id-0"})
    task_queue.put("STOP")

    worker_process(
        task_queue,
        done_queue,
        fake_rollout,
        fake_reward,
        client=None,
        sampling_params={"max_tokens": 16},
    )

    output_item = done_queue.get_nowait()
    assert output_item["instance_id"] == "id-0"
    assert output_item["reward"] == 1.0

    decoupling = output_item["extra_info"]["tool_decoupling"]
    assert decoupling["uses_tools"] is True
    assert decoupling["num_invocations"] == 2
    # Both calls are batched before execution => fully decoupled.
    assert decoupling["decoupling_score"] == 1.0

    assert done_queue.get_nowait() == "COMPLETE"
