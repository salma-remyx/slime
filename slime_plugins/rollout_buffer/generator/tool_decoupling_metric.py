"""Tool invocation/execution decoupling metrics for rollout trajectories.

Adapted from *Implicit Hierarchical GRPO: Decoupling Tool Invocation from
Execution for Tool-Integrated Mathematical Reasoning*
(https://arxiv.org/abs/2605.18500v1).

The paper's central observation is that tightly *coupling* a tool invocation
with its *immediate* execution interrupts the model's reasoning stream and
hurts tool-integrated reasoning (TIR). Its preferred regime is *delayed
execution*: the policy may emit one or more tool invocations and keep
reasoning, deferring execution to explicit control points.

We cannot ship the paper's IH-GRPO surrogate loss here (that lives in the
trainer, which this repo does not host on the rollout side). What we *can*
do, purely from rollout output, is measure how decoupled each trajectory
already is — a diagnostic signal that travels with the sample into the
buffer and can later drive reward shaping, filtering, or curriculum.

A trajectory's ``messages`` are flattened into an ordered event stream of
``invoke`` (tool call emitted), ``exec`` (execution result observed), and
``reason`` (free reasoning text) events. Both OpenAI-style ``tool_calls`` /
``role: "tool"`` messages and inline ``<tool_call>`` / ``<tool_response>``
tags (the formats the repo already parses in ``slime/agent/parsing.py``)
are recognized.
"""

from __future__ import annotations

import re
from typing import Any

# Inline-tag fallbacks for harnesses that keep tool traffic inside assistant
# text rather than structured message fields. Kept deliberately permissive so
# the metric degrades gracefully across model output conventions.
_INVOKE_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_EXEC_TAG_RE = re.compile(
    r"<(?:tool_response|tool_result|observation|interpreter|output)>(.*?)"
    r"</(?:tool_response|tool_result|observation|interpreter|output)>",
    re.DOTALL,
)

# Event kinds in the flattened stream.
INVOKE = "invoke"
EXEC = "exec"
REASON = "reason"


def _inline_events(content: str) -> list[tuple[str, int]]:
    """Split one text blob into ordered (kind, length) events by tag position.

    Text that sits outside any recognized tag is emitted as ``reason`` events
    so that reasoning interleaved with inline tool traffic is accounted for.
    """
    spans: list[tuple[int, int, str]] = []
    for m in _INVOKE_TAG_RE.finditer(content):
        spans.append((m.start(), m.end(), INVOKE))
    for m in _EXEC_TAG_RE.finditer(content):
        spans.append((m.start(), m.end(), EXEC))
    spans.sort()

    events: list[tuple[str, int]] = []
    cursor = 0
    for start, end, kind in spans:
        gap = content[cursor:start].strip()
        if gap:
            events.append((REASON, len(gap)))
        events.append((kind, end - start))
        cursor = end
    tail = content[cursor:].strip()
    if tail:
        events.append((REASON, len(tail)))
    return events


def extract_tool_events(messages: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Flatten chat ``messages`` into an ordered (kind, length) event stream.

    Structured fields take priority: an assistant message with a non-empty
    ``tool_calls`` list contributes one ``invoke`` per call, and a
    ``role: "tool"`` message contributes one ``exec``. Free-form content is
    scanned for inline tags as a fallback so mixed-format trajectories still
    register their tool traffic.
    """
    events: list[tuple[str, int]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            # Some adapters carry a list of content blocks; join their text.
            content = " ".join(
                str(block.get("text", "")) for block in content if isinstance(block, dict)
            )
        content = content or ""

        if role == "tool":
            events.append((EXEC, len(content)))
            continue

        tool_calls = msg.get("tool_calls") or []
        if role == "assistant" and tool_calls:
            text = content.strip()
            if text:
                events.append((REASON, len(text)))
            for _ in tool_calls:
                events.append((INVOKE, 0))
            continue

        # No structured tool fields: fall back to inline-tag scanning.
        if content.strip():
            inline = _inline_events(content)
            if inline:
                events.extend(inline)
            else:
                events.append((REASON, len(content.strip())))
    return events


def analyze_tool_decoupling(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Quantify how decoupled tool invocation is from execution in a trajectory.

    Returns a JSON-serializable dict of diagnostics. The headline figure is
    ``decoupling_score`` in ``[0, 1]``: the fraction of invocations that are
    *delayed* — i.e. followed by further reasoning or another invocation
    before their execution result appears, rather than being immediately
    coupled to an execution. ``1.0`` matches the paper's preferred decoupled
    regime; ``0.0`` is fully coupled, immediate execution.

    Trajectories with no tool invocations get ``decoupling_score == 1.0`` and
    ``uses_tools == False`` (vacuously decoupled, so they are never penalized
    by a decoupling-based filter).
    """
    events = extract_tool_events(messages)

    num_invocations = sum(1 for kind, _ in events if kind == INVOKE)
    num_executions = sum(1 for kind, _ in events if kind == EXEC)

    if num_invocations == 0:
        return {
            "uses_tools": False,
            "num_invocations": 0,
            "num_executions": num_executions,
            "num_coupled": 0,
            "num_delayed": 0,
            "decoupling_score": 1.0,
            "mean_reasoning_gap": 0.0,
            "disrupted": False,
        }

    num_coupled = 0
    num_delayed = 0
    reasoning_gaps: list[int] = []

    for idx, (kind, _length) in enumerate(events):
        if kind != INVOKE:
            continue
        prev_kind = events[idx - 1][0] if idx > 0 else None
        next_kind = events[idx + 1][0] if idx + 1 < len(events) else None

        # An invocation is *coupled* only when it stands alone and its result
        # follows immediately: it is not part of a batch (previous event is
        # another invocation) and execution arrives with no intervening
        # reasoning. Everything else — batched invocations, reasoning before
        # the result, or no result at all — is *delayed* / decoupled.
        coupled = prev_kind != INVOKE and next_kind == EXEC
        if coupled:
            num_coupled += 1
        else:
            num_delayed += 1

        # Reasoning emitted between this invocation and its execution result.
        gap = 0
        for follow_kind, follow_len in events[idx + 1 :]:
            if follow_kind == EXEC:
                break
            if follow_kind == REASON:
                gap += follow_len
        reasoning_gaps.append(gap)

    decoupling_score = num_delayed / num_invocations
    mean_reasoning_gap = sum(reasoning_gaps) / len(reasoning_gaps) if reasoning_gaps else 0.0

    return {
        "uses_tools": True,
        "num_invocations": num_invocations,
        "num_executions": num_executions,
        "num_coupled": num_coupled,
        "num_delayed": num_delayed,
        "decoupling_score": decoupling_score,
        "mean_reasoning_gap": mean_reasoning_gap,
        "disrupted": num_coupled > 0,
    }


def decoupling_bonus(metrics: dict[str, Any], weight: float = 0.0) -> float:
    """Optional reward-shaping term derived from a decoupling-metrics dict.

    Returns ``weight * decoupling_score`` so callers can nudge the policy
    toward the paper's delayed-execution regime. Defaults to ``0.0`` (a
    no-op) so recording the metric never silently changes rewards; opt in by
    passing a positive ``weight`` at the call site.
    """
    if weight == 0.0:
        return 0.0
    return weight * float(metrics.get("decoupling_score", 0.0))
