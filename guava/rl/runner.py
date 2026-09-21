"""Multi-process batch rollout for GRPO.

Worker model (mirrors `guava.collection.runner`):
  - mp.Pool(num_workers, initializer=_worker_init).
  - Each worker builds **one** `RolloutEngine` per task_config it ever sees,
    reused across all rollouts that worker handles. This is critical:
    rebuilding the engine per rollout would re-trigger the PyBullet
    client_id leak that bit eval (see fix(eval): pin pybullet client_id).
  - Each worker holds **one** OpenAI client; the inference server
    (e.g. vLLM) handles concurrency on its end.

Job ordering: each (prompt, rollout_idx) is one job. We use
`pool.imap_unordered` so slow rollouts don't block the queue. GRPO grouping
(K rollouts per prompt) is reconstructed by the caller via `prompt_id`.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from guava.rl.result import RolloutResult
from guava.rl.rollout import run_one_rollout


@dataclass(frozen=True)
class GRPOPrompt:
    """A single 'prompt' = sim task + initial seed. K rollouts share these."""
    task_config: str   # path to YAML (e.g. configs/lemon_in_bin.yaml)
    seed: int          # sim seed → deterministic init scene
    prompt_id: str     # human-readable group key, used as folder name


# Per-worker state: one engine per (task_config), one OpenAI client.
_worker: dict = {}


def _worker_init(task_configs: list[str], model_url: str) -> None:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from openai import OpenAI
    from guava.inference.rollout_env import RolloutEngine

    _worker["engines"] = {cfg: RolloutEngine(task_config=cfg) for cfg in task_configs}
    _worker["client"]  = OpenAI(base_url=model_url, api_key="EMPTY")


def _worker_run(args: tuple) -> RolloutResult:
    prompt, rollout_idx, model_name, params = args
    engine = _worker["engines"][prompt.task_config]
    client = _worker["client"]
    return run_one_rollout(
        engine=engine,
        client=client,
        seed=prompt.seed,
        prompt_id=prompt.prompt_id,
        rollout_idx=rollout_idx,
        output_dir=Path(params["output_dir"]),
        model_name=model_name,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        max_turns=params["max_turns"],
        alpha_format=params["alpha_format"],
    )


def run_grpo_batch(
    prompts: list[GRPOPrompt],
    *,
    num_rollouts_per_prompt: int,
    num_workers: int,
    model_url: str,
    model_name: str,
    output_dir: str | Path,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_turns: int = 25,
    alpha_format: float = 0.1,
) -> list[RolloutResult]:
    """Run K rollouts per prompt across `num_workers` parallel workers.

    Returns a flat list of `RolloutResult` (length = len(prompts) * K).
    Group by `result.prompt_id` to recover GRPO groups.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_configs = sorted({p.task_config for p in prompts})
    params = {
        "output_dir":   str(output_dir),
        "temperature":  temperature,
        "max_tokens":   max_tokens,
        "max_turns":    max_turns,
        "alpha_format": alpha_format,
    }
    jobs = [
        (prompt, k, model_name, params)
        for prompt in prompts
        for k in range(num_rollouts_per_prompt)
    ]

    if num_workers <= 1:
        # Single-process path — easier to debug + drops mp overhead.
        _worker_init(task_configs, model_url)
        return [_worker_run(j) for j in jobs]

    ctx = mp.get_context("spawn")  # CUDA-safe; robosuite/mujoco prefer fresh procs
    with ctx.Pool(num_workers, initializer=_worker_init,
                  initargs=(task_configs, model_url)) as pool:
        return list(pool.imap_unordered(_worker_run, jobs))
