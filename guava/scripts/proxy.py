"""Guava LLM proxy server.

Presents a single OpenAI-compatible endpoint at /v1/chat/completions and
routes requests to the configured backend (OpenRouter, OpenAI, Google,
Anthropic, or a local vLLM server). This proxy is optional; collection now
connects directly to OpenRouter by default. To use the proxy, explicitly set
llm.base_url to http://127.0.0.1:8110/v1.

Usage:
    # OpenRouter
    uv run python -m guava.scripts.proxy \\
        --provider openrouter --api-key YOUR_KEY \\
        --model "google/gemini-2.5-pro-preview"

    # Google AI Studio
    uv run python -m guava.scripts.proxy \\
        --provider google --api-key YOUR_KEY \\
        --model "gemini-2.5-pro-preview"

    # Anthropic
    uv run python -m guava.scripts.proxy \\
        --provider anthropic --api-key YOUR_KEY \\
        --model "claude-sonnet-4-5"

    # Local vLLM  (no api-key needed)
    uv run python -m guava.scripts.proxy \\
        --provider vllm --base-url http://localhost:8000/v1 \\
        --model "your-local-model"
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    provider: str = "openrouter"
    """Backend provider: openrouter | openai | google | anthropic | vllm."""

    model: str = "google/gemma-4-31b-it:free"
    """Model name to use for all requests through this proxy."""

    api_key: str = "none"
    """API key for the provider (not required for vllm)."""

    base_url: str | None = None
    """Base URL override. Required for vllm; optional for openrouter/openai."""

    host: str = "0.0.0.0"
    port: int = 8110


def _make_provider(args: Args):
    """Instantiate the appropriate backend provider for the proxy."""
    _defaults = {
        "openrouter": "https://openrouter.ai/api/v1",
        "openai":     "https://api.openai.com/v1",
        "vllm":       "http://localhost:8000/v1",
    }
    base_url = args.base_url or _defaults.get(args.provider)

    if args.provider in ("openrouter", "openai", "vllm"):
        from guava.llm.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(base_url=base_url, api_key=args.api_key, model=args.model)

    if args.provider == "google":
        from guava.llm.providers.google import GoogleProvider
        return GoogleProvider(api_key=args.api_key, model=args.model)

    if args.provider == "anthropic":
        from guava.llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=args.api_key, model=args.model)

    raise ValueError(f"Unknown provider '{args.provider}'")


def create_app(provider) -> FastAPI:
    app = FastAPI(title="Guava LLM Proxy", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": []}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body: dict[str, Any] = await request.json()
        messages = body.get("messages", [])
        tools    = body.get("tools") or None
        kwargs   = {
            k: body[k]
            for k in ("temperature", "max_tokens", "max_completion_tokens")
            if k in body
        }
        try:
            # Run the (synchronous) provider in a thread so we don't block the loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: provider.complete(messages, tools=tools, **kwargs)
            )
            return JSONResponse(content=response)
        except Exception as exc:
            logger.exception("Provider error")
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def main() -> None:
    args = tyro.cli(Args)
    logger.info(f"Starting guava proxy — provider={args.provider} model={args.model} port={args.port}")
    provider = _make_provider(args)
    app = create_app(provider)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
