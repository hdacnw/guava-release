"""Adapt the audited eval-ubuntu Responses provider to the tasks-native loop."""
import os
from pathlib import Path
from types import SimpleNamespace
from guava.llm.providers.responses import ResponsesProvider

MODEL = 'openai/openai/gpt-5.4'
ENDPOINT = 'https://inference-api.nvidia.com/v1/responses'


class NvidiaEvalClient:
    def __init__(self, root):
        key = os.environ.get('NVIDIA_API_KEY', '')
        key_file = Path(root) / '.env.branch'
        if not key and key_file.is_file():
            for line in key_file.read_text().splitlines():
                if line.startswith('NVIDIA_API_KEY='):
                    key = line.split('=', 1)[1].strip().strip('\"\'')
        if not key:
            raise RuntimeError('NVIDIA_API_KEY is missing')
        self.provider = ResponsesProvider(ENDPOINT, key, MODEL, 'medium')
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model, messages, temperature, max_tokens):
        if model != MODEL:
            raise ValueError('NVIDIA adapter requires the explicit GPT-5.4 model')
        # XML tools are text, not native function calls. Feedback images must
        # use a user role in the Responses schema; preserve every content part.
        converted = [{**m, 'role': 'user' if m['role'] == 'tool' else m['role']}
                     for m in messages]
        data = self.provider.complete(converted, max_tokens=max_tokens)
        choice = data['choices'][0]
        usage = data.get('usage')
        return SimpleNamespace(
            response_audit=data.get('response_audit'),
            choices=[SimpleNamespace(finish_reason=choice['finish_reason'],
                      message=SimpleNamespace(content=choice['message']['content']))],
            usage=SimpleNamespace(model_dump=lambda: usage) if usage else None)
