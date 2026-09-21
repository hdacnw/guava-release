import os
from guava.config import LLMConfig


def get_provider(cfg: LLMConfig):
    if cfg.base_url.rstrip('/').endswith('/responses'):
        from guava.llm.providers.responses import ResponsesProvider
        return ResponsesProvider(cfg.base_url, os.environ.get(cfg.api_key_env, ''), cfg.model)
    from guava.llm.providers.openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env, ''), model=cfg.model)
