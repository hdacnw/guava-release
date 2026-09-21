from __future__ import annotations
import time
import random
import requests
from dataclasses import dataclass, field
from urllib.parse import urlparse

# Status codes that warrant a retry
_RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class OpenAICompatProvider:
    base_url: str
    api_key: str = field(repr=False)
    model: str

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict = {"model": self.model, "messages": messages, **kwargs}
        # Hosted reasoning models may reject explicit sampling temperatures.
        if urlparse(self.base_url).hostname == 'openrouter.ai':
            body.pop('temperature', None)
        if "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        url = self.base_url.rstrip("/") + "/chat/completions"
        delay = 5.0
        for attempt in range(15):
            resp = requests.post(url, json=body, headers=headers, timeout=120)
            if resp.status_code not in _RETRY_STATUSES:
                break
            # Respect Retry-After header if present (common on 429).
            # Guard against non-numeric values (OpenRouter sometimes sends an HTTP date string).
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else None
            except (ValueError, TypeError):
                wait = None
            # 429 rate-limit: use Retry-After if given, else at least 30 s
            if wait is None:
                wait = max(30.0, delay) if resp.status_code == 429 else delay
            wait += random.uniform(0, 2)
            print(f"[openai_compat] HTTP {resp.status_code} — retrying in {wait:.1f}s (attempt {attempt + 1})")
            time.sleep(wait)
            delay = min(delay * 2, 120)  # exponential backoff, cap at 2 min

        if not resp.ok:
            raise requests.exceptions.HTTPError(
                f"HTTP {resp.status_code}; request/response body suppressed", response=resp
            )
        return resp.json()
