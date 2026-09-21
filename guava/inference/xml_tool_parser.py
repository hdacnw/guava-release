"""Parse Qwen3.5-native XML tool calls out of an assistant response string.

Wire format the model emits:

    <think>
    reasoning text...
    </think>

    <tool_call>
    <function=move>
    <parameter=target_position>
    [0.516, -0.29, 0.018]
    </parameter>
    </function>
    </tool_call>

The values inside `<parameter>` are emitted by the chat template via
`json.dumps` for list/dict args and `str(...)` for scalars (see
`models/Qwen3.5-4B/chat_template.jinja:122`). We mirror that on the parse
side: try `json.loads` first, fall back to the raw string.

Returns a normalized record:

    {
      "thinking":  str | None,    # contents of <think>...</think>
      "tool_call": {              # None if the model declared the task done
        "name":      str,
        "arguments": dict[str, Any],
      } | None,
      "finished": bool,           # True if response says "Task complete." / "Task failed."
      "raw_text": str,            # any natural-language text outside think/tool_call
    }

This is a pure function — no side effects, no I/O, no model deps — so it's
easy to unit test.
"""
from __future__ import annotations

import json
import re
from typing import Any


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([\w\-]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([\w\-]+)>\s*\n?(.*?)\n?\s*</parameter>", re.DOTALL)
_FINISHED_RE = re.compile(r"\bTask\s+(complete|failed)\.?", re.IGNORECASE)


def _coerce(value: str) -> Any:
    """Mirror the chat template's `json.dumps`-on-list/dict, `str` on scalars.

    Try parsing as JSON first (covers list/dict/numbers/booleans). If that
    fails, return the raw stripped string.
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse(response: str) -> dict[str, Any]:
    """Parse an assistant response. See module docstring for return shape."""
    think_match = _THINK_RE.search(response)
    thinking = think_match.group(1).strip() if think_match else None

    tool_call = None
    tc_match = _TOOL_CALL_RE.search(response)
    if tc_match:
        tc_inner = tc_match.group(1).strip()
        # v5 training stores tool_call body as JSON: `{"name": "...", "arguments": {...}}`.
        # Older Hermes-style XML (`<function=name><parameter=k>v</parameter></function>`)
        # is also accepted for compatible model outputs. Try JSON first, then XML.
        if tc_inner.startswith("{"):
            try:
                obj = json.loads(tc_inner)
                tool_call = {
                    "name": obj.get("name") or "",
                    "arguments": obj.get("arguments") or {},
                }
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"<tool_call> body looked like JSON but didn't parse: {tc_inner!r} ({e})"
                )
        else:
            fn_match = _FUNCTION_RE.search(tc_inner)
            if not fn_match:
                raise ValueError(
                    f"Found <tool_call> but no <function=...> inside: {tc_inner!r}"
                )
            name = fn_match.group(1)
            params_block = fn_match.group(2)
            arguments = {
                m.group(1): _coerce(m.group(2).strip())
                for m in _PARAM_RE.finditer(params_block)
            }
            tool_call = {"name": name, "arguments": arguments}

    finished = bool(_FINISHED_RE.search(response))

    # Strip out structured blocks; what's left is free-form text the model
    # sometimes emits between </think> and <tool_call> or after the tool call.
    text_only = _THINK_RE.sub("", response)
    text_only = _TOOL_CALL_RE.sub("", text_only)
    raw_text = text_only.strip()

    return {
        "thinking": thinking,
        "tool_call": tool_call,
        "finished": finished,
        "raw_text": raw_text,
    }
