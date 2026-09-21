"""Single multi-turn rollout against a `RolloutEngine`.

Uses the shared table-aligned short-SFT rollout interface:
  - returns a `RolloutResult` (no print spam — many rollouts in flight)
  - does not gate behavior on the system prompt name etc. (engine owns that)
  - tolerates client errors (timeouts, bad JSON) without crashing the worker —
    they become `end_reason='client_error'`, reward 0.

The trajectory IS still saved to `output_dir/<prompt_id>/episode_<rollout_idx>/`
because GRPO's policy-gradient step needs to recompute log-probs over the
assistant tokens, and the chat_template depends on the saved per-turn images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from guava.inference.rollout_env import RolloutEngine
from guava.rl.result import RolloutResult, compute_reward


def run_one_rollout(
    engine: RolloutEngine,
    client: Any,                       # openai.OpenAI (or compatible)
    *,
    seed: int,
    prompt_id: str,
    rollout_idx: int,
    output_dir: Path,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_turns: int = 25,
    alpha_format: float = 0.1,
) -> RolloutResult:
    """Run one full episode end-to-end and return its `RolloutResult`."""
    episode_root = Path(output_dir) / prompt_id
    env = engine.episode(episode_id=rollout_idx, output_dir=str(episode_root))
    env.cfg.turn_budget = max_turns

    obs = env.reset(seed=seed)

    end_reason = "max_turns"
    format_ok = True
    client_error: str | None = None

    while not obs.get("done"):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=obs["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = completion.choices[0].message.content or ""
        except Exception as e:
            client_error = f"{type(e).__name__}: {e}"
            end_reason = "client_error"
            format_ok = False
            break

        obs = env.step(response)

    if client_error is None:
        end_reason = obs.get("info", {}).get("end_reason", end_reason)
        if end_reason == "format_error_no_tool_call":
            format_ok = False

    success = (end_reason == "task_success")
    reward = compute_reward(end_reason, alpha_format=alpha_format)

    env.save_trajectory()

    return RolloutResult(
        messages=env.messages,
        reward=reward,
        success=success,
        format_ok=format_ok,
        end_reason=end_reason,
        n_turns=env.turn,
        seed=seed,
        prompt_id=prompt_id,
        rollout_idx=rollout_idx,
        metadata={"coordinate_frame": "table_aligned",
                  "model_z_equals_base_z_plus": engine.ctx.model_z_offset,
                  **({"client_error": client_error} if client_error else {})},
    )
