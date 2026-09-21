from __future__ import annotations
import json, requests
from dataclasses import dataclass


@dataclass
class GoogleProvider:
    api_key: str
    model: str  # e.g. "gemini-2.5-pro-preview-03-25"

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        """Call Google AI Studio and return an OpenAI-compatible response dict."""
        contents, system = _to_gemini_messages(messages)
        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": _to_gemini_tools(tools)}]

        url = f"{self._BASE}/{self.model}:generateContent?key={self.api_key}"
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        raw = resp.json()
        return _from_gemini_response(raw)


def _to_gemini_messages(messages: list[dict]) -> tuple[list[dict], str]:
    """Convert OpenAI messages → Gemini contents. Returns (contents, system_text)."""
    system = ""
    contents = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            system = content if isinstance(content, str) else " ".join(
                p["text"] for p in content if p.get("type") == "text"
            )
            continue
        gemini_role = "user" if role == "user" else "model"
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for p in content:
                if p.get("type") == "text":
                    parts.append({"text": p["text"]})
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    if url.startswith("data:"):
                        media, b64 = url.split(",", 1)
                        mime = media.split(";")[0].split(":")[1]
                        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        contents.append({"role": gemini_role, "parts": parts})
    return contents, system


def _to_gemini_tools(tools: list[dict]) -> list[dict]:
    return [t["function"] for t in tools]


def _from_gemini_response(raw: dict) -> dict:
    """Wrap Gemini response as an OpenAI-compatible dict."""
    candidate = raw["candidates"][0]
    parts = candidate["content"]["parts"]
    tool_calls = []
    text = ""
    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "id": "call_0",
                "type": "function",
                "function": {"name": fc["name"], "arguments": json.dumps(fc.get("args", {}))},
            })
        elif "text" in part:
            text += part["text"]

    message: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}
