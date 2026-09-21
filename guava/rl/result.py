"""Result + reward computation for an RL rollout.

`RolloutResult` is the unit GRPO operates on:
    K results per prompt → group-relative advantage = (r_k - mean) / std.

Reward shaping (intentionally minimal at this stage):
    +1.0   task_success
    -alpha format_error_no_tool_call   (penalize parser failures so the
                                        model keeps emitting valid XML)
     0.0   everything else (turn_budget_exceeded, model_declared_done
                            without success, sim_horizon, client_error)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutResult:
    messages: list[dict[str, Any]]   # full chat history; trainer recomputes log-probs over assistant turns
    reward: float                    # post-shaping scalar
    success: bool                    # task reward >= 1.0
    format_ok: bool                  # parser succeeded every assistant turn
    end_reason: str                  # task_success | model_declared_done | turn_budget_exceeded | format_error_no_tool_call | client_error | sim_horizon
    n_turns: int
    seed: int                        # sim seed (so rollout is reproducible)
    prompt_id: str                   # human-readable group key (task_config + seed)
    rollout_idx: int                 # 0..K-1 inside the GRPO group
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_reward(end_reason: str, *, alpha_format: float = 0.1) -> float:
    """Binary task reward + format penalty. See module docstring."""
    if end_reason == "task_success":
        return 1.0
    if end_reason == "format_error_no_tool_call":
        return -alpha_format
    return 0.0
