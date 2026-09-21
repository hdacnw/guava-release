from __future__ import annotations
import json, requests
from dataclasses import dataclass


@dataclass
class AnthropicProvider:
    api_key: str
    model: str  # e.g. "claude-opus-4-5"

    _BASE = "https://api.anthropic.com/v1"

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        system, user_msgs = _split_system(messages)
        body: dict = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": user_msgs,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = _to_anthropic_tools(tools)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = requests.post(f"{self._BASE}/messages", json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        return _from_anthropic_response(resp.json())


def _split_system(messages):
    system = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"] if isinstance(m["content"], str) else ""
        else:
            rest.append(m)
    return system, rest


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        f = t["function"]
        out.append({"name": f["name"], "description": f.get("description", ""), "input_schema": f["parameters"]})
    return out


def _from_anthropic_response(raw: dict) -> dict:
    tool_calls, text = [], ""
    for block in raw.get("content", []):
        if block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
            })
        elif block["type"] == "text":
            text += block["text"]
    message: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}
