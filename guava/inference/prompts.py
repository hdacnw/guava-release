"""Short SFT prompt default for table-aligned RL and inference rollouts.

Benchmark prompts are explicitly selected by study_protocols. Any override here
must also describe table-aligned coordinates and the canonical nine-tool API.
"""
from __future__ import annotations

import os
from pathlib import Path

# guava/guava/inference/prompts.py → up 3 levels lands on the guava repo root.
_GUAVA_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = _GUAVA_ROOT / "configs" / "prompts"
DEFAULT_PROMPT_NAME = "sft_v13b_short"


def resolve(name: str | os.PathLike) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    if p.suffix:        # has extension → relative path
        return PROMPTS_DIR / p
    return PROMPTS_DIR / f"{p}.txt"


def load(name: str | os.PathLike | None = None) -> str:
    """Read a named prompt or an absolute path.
    Falls back to the env var GUAVA_SYSTEM_PROMPT, then DEFAULT_PROMPT_NAME.
    """
    chosen = name or os.environ.get("GUAVA_SYSTEM_PROMPT") or DEFAULT_PROMPT_NAME
    path = resolve(chosen)
    if not path.exists():
        raise FileNotFoundError(
            f"system prompt not found: {path} (resolved from {chosen!r})"
        )
    return path.read_text().rstrip()
