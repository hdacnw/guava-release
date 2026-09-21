"""Responses gateway adapter. Never expose request credentials in errors."""
from dataclasses import dataclass, field
import time
import requests
from .output_error import IncompleteOutput


def responses_input(messages):
    result = []
    for msg in messages:
        role = msg['role']
        if role not in ('system', 'developer', 'user', 'assistant'):
            raise ValueError(f'Unsupported Responses history role: {role}')
        content = msg['content']
        if isinstance(content, str):
            result.append({'role': role, 'content': content})
            continue
        parts = []
        for part in content:
            if part['type'] == 'text':
                parts.append({'type': 'output_text' if role == 'assistant' else 'input_text',
                              'text': part['text']})
            elif part['type'] == 'image_url' and role == 'user':
                parts.append({'type': 'input_image', 'image_url': part['image_url']['url']})
            else:
                raise ValueError('Unsupported Responses content part')
        result.append({'role': role, 'content': parts})
    return result


@dataclass
class ResponsesProvider:
    base_url: str
    api_key: str = field(repr=False)
    model: str = 'openai/openai/gpt-5.4'
    reasoning_effort: str = 'medium'
    timeout: int = 180

    def complete(self, messages, tools=None, **kwargs):
        if not self.api_key:
            raise ValueError('API key environment variable is unset')
        if tools:
            raise ValueError('Collection uses text-embedded calls, not native tool schemas')
        body = {'model': self.model, 'input': responses_input(messages),
                'max_output_tokens': kwargs.get('max_tokens', 8192),
                'reasoning': {'effort': self.reasoning_effort}, 'store': False}
        # GPT-5.4 with reasoning does not accept temperature. No silent model fallback.
        url = self.base_url.rstrip('/')
        if not url.endswith('/responses'):
            url += '/responses'
        for attempt in range(3):
            try:
                response = requests.post(url, headers={'Authorization': 'Bearer ' + self.api_key},
                                         json=body, timeout=self.timeout)
            except requests.RequestException:
                raise RuntimeError('Responses transport failed (details suppressed to protect credentials)') from None
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                break
            time.sleep(2 ** attempt)
        if not response.ok:
            raise RuntimeError(f'Responses HTTP {response.status_code}; request/response body suppressed')
        data = response.json()
        if data.get('status') != 'completed':
            # Deliberately omit reasoning blocks and request/credential data.
            raise IncompleteOutput({'status': data.get('status'),
                'incomplete_details': data.get('incomplete_details'),
                'response_id': data.get('id'), 'usage': data.get('usage'),
                'content': '\n'.join(p.get('text', '') for item in data.get('output', [])
                    if item.get('type') == 'message' for p in item.get('content', [])
                    if p.get('type') == 'output_text')})
        parts = []
        for item in data.get('output', []):
            if item.get('type') == 'reasoning':
                continue  # Never turn hidden reasoning or a summary into a public action rationale.
            if item.get('type') != 'message' or item.get('role') != 'assistant':
                raise ValueError('Unexpected Responses output item; no action executed')
            for part in item.get('content', []):
                if part.get('type') != 'output_text':
                    raise ValueError('Refusal or unsupported Responses content; no action executed')
                parts.append(part['text'])
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '\n'.join(parts)}}],
                'id': data.get('id'), 'usage': data.get('usage'),
                # Preserve public output boundaries without request credentials,
                # hidden reasoning, reasoning summaries or encrypted payloads.
                'response_audit': {
                    'id': data.get('id'), 'status': data.get('status'), 'model': data.get('model'),
                    'output': [
                        {'type': item.get('type'), 'id': item.get('id'),
                         'role': item.get('role'), 'status': item.get('status'),
                         'content': [{'type': p.get('type'), 'text': p.get('text', '')}
                                     for p in item.get('content', []) if p.get('type') == 'output_text']}
                        if item.get('type') == 'message' else
                        {'type': item.get('type'), 'id': item.get('id'), 'content_omitted': True}
                        for item in data.get('output', [])]}}
