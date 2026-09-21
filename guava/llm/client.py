from __future__ import annotations
import json, re
import warnings
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class ToolCall:
    tool_name: str | None   # None = text-only response (no tool called)
    tool_args: dict
    reasoning: str          # extracted from <think>...</think>
    text: str               # full text content of the message
    raw_content: str | None = None
    model_message: dict | None = None  # response only, never request credentials/images
    reasoning_source: str = 'unknown'


def query_plan(provider, messages: list[dict], cfg) -> str:
    resp = provider.complete(messages, temperature=cfg.llm.temperature, max_tokens=512)
    return _text(resp)


def query_tool_call(provider, messages: list[dict], cfg) -> ToolCall:
    # Request text-embedded calls for the training format. This does not
    # guarantee a rationale: extraction is checked and missing text is flagged.
    resp = provider.complete(
        messages,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )
    choice = resp['choices'][0]
    if choice.get('finish_reason') in ('length', 'content_filter'):
        raise ValueError('Incomplete or filtered model response; no action executed')
    msg = choice['message']
    raw = msg.get("content") or ""

    # Extract <think> reasoning
    reasoning = ""
    reasoning_source = 'missing'
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
        if reasoning:
            reasoning_source = 'content_think'
    if not reasoning:
        for field in ('reasoning_content', 'reasoning'):
            value = msg.get(field)
            if isinstance(value, str) and value.strip():
                reasoning = value.strip()
                reasoning_source = field
                break

    def result(name, args):
        if name and not reasoning:
            warnings.warn('Tool response has no extractable rationale; retain as context-only in SFT. '
                          'No action is retried for this formatting issue.', RuntimeWarning, stacklevel=2)
        return ToolCall(tool_name=name, tool_args=args, reasoning=reasoning,
                        text=raw, raw_content=raw, model_message=deepcopy(msg),
                        reasoning_source=reasoning_source)

    # Native tool call from the model
    if msg.get("tool_calls"):
        if len(msg['tool_calls']) != 1:
            raise ValueError('Expected exactly one action, not multiple tool calls')
        tc = msg["tool_calls"][0]["function"]
        args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
        return result(tc["name"], args)

    # Text-embedded JSON fallback (some models return tool calls in text)
    action_text = re.sub(r'<think>.*?</think>', '', raw, flags=re.S)
    if action_text.count('<tool_call>') > 1:
        raise ValueError('Expected exactly one action, not multiple tool calls')
    name, args = _parse_tool_from_text(action_text)
    if name:
        return result(name, args)

    # No tool called — text-only response (agent is thinking, planning, or signalling completion)
    return result(None, {})


def query_verify(provider, image_b64: str, action_desc: str, expected: str, cfg) -> dict:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": (
            f"Action taken: {action_desc}\n"
            f"Expected outcome: {expected}\n\n"
            "Look at the image and respond in JSON only: "
            '{"ok": true/false, "observation": "one sentence describing what you see", '
            '"issue": "if not ok, what specifically went wrong"}'
        )},
    ]
    messages = [{"role": "user", "content": content}]
    resp = provider.complete(messages, temperature=0.1, max_tokens=200)
    text = _text(resp)
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {"ok": True, "observation": text, "issue": ""}
    except Exception:
        return {"ok": True, "observation": text, "issue": ""}


def _text(resp: dict) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def _parse_tool_from_text(text: str) -> tuple[str | None, dict]:
    """Try to extract a tool name + args from free-form text.

    Handles the formats models commonly produce:
      0. <tool_call>{...}</tool_call>  (primary format we train for)
      1. {"name": "tool", "arguments": {...}}
      2. <function=tool_name {...}>   (Llama / Mistral style)
      3. {"tool": "tool", "args": {...}}
    Uses raw_decode so trailing garbage (e.g. extra '}') does not break parsing.
    """

    def _try_json(s: str) -> dict | None:
        try:
            obj, _ = json.JSONDecoder().raw_decode(s.strip())
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    # Format 0: <tool_call>{...}</tool_call>
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if m:
        obj = _try_json(m.group(1))
        if obj and isinstance(obj.get("arguments"), dict):
            return obj["name"], obj["arguments"]

    # Format 1: {"name": "...", "arguments": {...}}
    for m in re.finditer(r'"name"\s*:\s*"(\w+)"', text):
        start = text.rfind("{", 0, m.start())
        if start >= 0:
            obj = _try_json(text[start:])
            if obj and isinstance(obj.get("arguments"), dict):
                return obj["name"], obj["arguments"]

    # Format 2: <function=tool_name {...}>
    m = re.search(r"<function=(\w+)\s*(\{)", text)
    if m:
        obj = _try_json(text[m.start(2):])
        if obj:
            return m.group(1), obj

    # Format 3: {"tool": "...", "args": {...}}
    for m in re.finditer(r'"tool"\s*:\s*"(\w+)"', text):
        start = text.rfind("{", 0, m.start())
        if start >= 0:
            obj = _try_json(text[start:])
            if obj and isinstance(obj.get("args"), dict):
                return obj["tool"], obj["args"]

    return None, {}
