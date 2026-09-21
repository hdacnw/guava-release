"""Hosted model adapter for the benchmark's chat-style interface."""
import os
from types import SimpleNamespace

from guava.config import LLMConfig
from guava.llm.providers import get_provider
from guava.llm.provider_settings import DEFAULTS


class RemoteEvalClient:
    def __init__(self, backend, model):
        endpoint, key_env, _ = DEFAULTS[backend]
        if not os.environ.get(key_env):
            raise RuntimeError(f'{key_env} is missing')
        self.model = model
        self.provider = get_provider(LLMConfig(base_url=endpoint, api_key_env=key_env, model=model))
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model, messages, temperature, max_tokens):
        if model != self.model:
            raise ValueError('Requested model differs from configured model')
        messages = [{**m, 'role': 'user' if m['role'] == 'tool' else m['role']}
                    for m in messages]
        # Leave sampling to the hosted provider; reasoning models may reject temperature.
        data = self.provider.complete(messages, max_tokens=max_tokens)
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop' or choice['message'].get('tool_calls'):
            raise ValueError('Incomplete, filtered, or unsupported hosted response; no action executed')
        if not isinstance(choice['message'].get('content'), str) or not choice['message']['content'].strip():
            raise ValueError('Hosted response has no public text; no action executed')
        usage = data.get('usage')
        return SimpleNamespace(
            response_audit=data.get('response_audit'),
            choices=[SimpleNamespace(finish_reason=choice['finish_reason'],
                message=SimpleNamespace(content=choice['message'].get('content')))],
            usage=SimpleNamespace(model_dump=lambda: usage) if usage else None)
