"""ms-swift multi-turn rollout plugin for the guava sim.

This file is loaded by both `swift rollout` and `swift rlhf` via
`--external_plugins`. It registers:

1. ``guava_sim`` — a gym Env that talks to ``guava.scripts.sim_server`` over
   HTTP. (See run_grpo_native.sh: sim_server is its own .venv process so the
   training stack doesn't have to share mujoco/robosuite/sam3 deps.)

2. ``guava_tool_scheduler`` — a ``MultiTurnScheduler`` subclass that owns the
   per-rollout env lifecycle AND emits explicit ``response_token_ids`` /
   ``response_loss_mask`` per turn. The built-in ``gym_scheduler`` falls back
   to letting the trainer re-tokenize the completion text, which on multi-turn
   multimodal rollouts produces a 1-token shape mismatch in
   ``torch.gather(logits, ..., index)`` (see commit message / docstring of
   GuavaMultiTurnScheduler for details).

Env contract (data_dict.env_config fields)
-----------------------------------------
    task_config:         "configs/can_in_bin.yaml"   (required)
    sim_server_urls:     ["http://127.0.0.1:8200",   (optional; list form for
                          "http://127.0.0.1:8201"]    sharding across N
                                                      processes — picked
                                                      round-robin per episode)
    sim_server_url:      "http://127.0.0.1:8200"     (optional single-URL form,
                                                      kept for legacy callers;
                                                      env var
                                                      GUAVA_SIM_SERVER_URL
                                                      also accepts a comma-
                                                      separated list)
    output_dir:          "/tmp/grpo_rollouts"        (optional)
    system_prompt_name:  null                         (optional)
    seed:                12345                        (optional; if absent we
                                                      derive one from
                                                      RolloutInferRequest.uuid)
"""
from __future__ import annotations

import itertools
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import httpx

from swift.rollout.gym_env import ContextManager, Env, context_managers, envs
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.template.utils import Messages


# Default URL list: comma-separated env var, single URL fallback, or 8200.
def _parse_default_urls() -> List[str]:
    raw = os.environ.get("GUAVA_SIM_SERVER_URLS") or os.environ.get("GUAVA_SIM_SERVER_URL", "")
    if not raw:
        return ["http://127.0.0.1:8200"]
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


DEFAULT_SERVER_URLS: List[str] = _parse_default_urls()
# Round-robin counter shared across all GuavaSimEnv instances in this rollout
# worker process. Each new env grabs `next(_round_robin)` so K parallel
# episodes spread across N sim_server processes.
_round_robin = itertools.count()
_url_cycle = itertools.cycle(DEFAULT_SERVER_URLS)

# 900s (15 min): with MUJOCO_GL=egl and one EGL display per sim_server
# process, ops within a process serialize on its single sim thread. Under
# heavy GRPO concurrency a queued /step can wait for several ahead-of-it
# rollouts to finish, so we keep the generous timeout.
DEFAULT_TIMEOUT_S = float(os.environ.get("GUAVA_SIM_TIMEOUT_S", "900"))

# Successful rollouts keep the dominant success-vs-failure signal, while
# unnecessarily long successful trajectories receive a small terminal
# deduction. Failures remain at zero so the model cannot improve its reward by
# declaring failure early. With the defaults, turns 1-16 are free and a
# success at turn 20 receives 0.90 instead of 1.00.
LENGTH_PENALTY_FREE_TURNS = int(
    os.environ.get("GUAVA_LENGTH_PENALTY_FREE_TURNS", "16")
)
LENGTH_PENALTY_PER_TURN = float(
    os.environ.get("GUAVA_LENGTH_PENALTY_PER_TURN", "0.025")
)
LENGTH_PENALTY_MAX = float(
    os.environ.get("GUAVA_LENGTH_PENALTY_MAX", "0.10")
)


def _apply_success_length_penalty(reward: float, num_turns: int) -> tuple[float, float]:
    """Return (adjusted_reward, penalty), applying length cost to successes only."""
    if reward <= 0.0:
        return reward, 0.0
    excess_turns = max(0, num_turns - LENGTH_PENALTY_FREE_TURNS)
    penalty = min(LENGTH_PENALTY_MAX, excess_turns * LENGTH_PENALTY_PER_TURN)
    return max(0.0, reward - penalty), penalty


# ─────────────────────────────────────────────────────────────────────────── #
# 1. Env — thin HTTP client to guava.scripts.sim_server
# ─────────────────────────────────────────────────────────────────────────── #
class GuavaSimEnv(Env):
    """ms-swift gym env that delegates rollout state to sim_server over HTTP."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        if not env_config.get("task_config"):
            raise ValueError(
                "GuavaSimEnv requires env_config['task_config'] — path to a configs/*.yaml"
            )
        self.task_config = env_config["task_config"]
        self.system_prompt_name = env_config.get("system_prompt_name")
        self.output_dir = env_config.get("output_dir")
        # URL selection priority:
        #   1. env_config['sim_server_urls']  (list → round-robin)
        #   2. env_config['sim_server_url']   (single URL)
        #   3. process-global DEFAULT_SERVER_URLS (round-robin)
        urls_in_cfg = env_config.get("sim_server_urls")
        if isinstance(urls_in_cfg, (list, tuple)) and urls_in_cfg:
            url_list = [str(u).rstrip("/") for u in urls_in_cfg]
            self.server_url = url_list[next(_round_robin) % len(url_list)]
        elif env_config.get("sim_server_url"):
            self.server_url = str(env_config["sim_server_url"]).rstrip("/")
        else:
            self.server_url = next(_url_cycle)
        self.timeout: float = float(env_config.get("timeout_s", DEFAULT_TIMEOUT_S))
        # Seed comes from env_config (so prompts are reproducible). Fall back
        # to hash(uuid) only for ad-hoc test calls without a configured seed.
        self.configured_seed: int | None = (
            int(env_config["seed"]) if "seed" in env_config else None
        )
        # Disable HTTP keep-alive entirely: GRPO rollouts have long gaps
        # between /reset and /step on the same httpx client (the LLM thinks
        # for 5-15s between calls). Persistent connections expire mid-rollout
        # and then httpx surfaces "Server disconnected without sending a
        # response" on the next call. Cost is a TCP handshake per request,
        # but sim_server is localhost so the overhead is negligible.
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._episode_id: str | None = None

    # ─────────────────────────────────────────────────────────────────── #
    async def reset(self, config) -> Tuple[Any, Dict[str, Any], str]:
        if self.configured_seed is not None:
            seed = self.configured_seed
        else:
            seed = abs(hash(getattr(config, "uuid", ""))) % (2**31)

        payload = {
            "task_config": self.task_config,
            "seed": seed,
            "system_prompt_name": self.system_prompt_name,
        }
        if self.output_dir:
            payload["output_dir"] = self.output_dir

        r = await self._client.post(f"{self.server_url}/reset", json=payload)
        if r.status_code >= 400:
            print(
                f"[GuavaSimEnv] /reset {self.server_url} status={r.status_code} body={r.text[:500]!r}",
                flush=True,
            )
        r.raise_for_status()
        data = r.json()
        self._episode_id = data["episode_id"]
        info = {
            "task": data.get("task", ""),
            "episode_id": self._episode_id,
            "seed": seed,
        }
        return data["observation"], info, data["system_message"]

    async def step(self, action: Messages) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if self._episode_id is None:
            raise RuntimeError("GuavaSimEnv.step called before reset()")

        assistant_text = _extract_assistant_text(action)
        if not assistant_text:
            return ("Empty assistant response.", 0.0, True, {"end_reason": "empty_response"})

        # Swallow sim failures so one bad rollout doesn't kill the whole training
        # step. The error observation is NOT appended to the model's context (see
        # GuavaMultiTurnScheduler.run: it only appends next_obs when `not done`),
        # so the model never sees these error tokens. The rollout terminates with
        # reward=0; group-baseline normalization in GRPO then gives this rollout
        # a strongly negative advantage automatically when sibling rollouts in
        # the same K-group succeed — no need to inject an explicit negative
        # reward, which would also punish sim-side noise (MuJoCo NaN, controller
        # divergence) that isn't the model's fault.
        try:
            r = await self._client.post(
                f"{self.server_url}/step",
                json={"episode_id": self._episode_id, "assistant_text": assistant_text},
            )
        except Exception as e:
            print(
                f"[GuavaSimEnv] /step {self.server_url} episode_id={self._episode_id} "
                f"network error: {type(e).__name__}: {e} — terminating episode with reward=0",
                flush=True,
            )
            return ("", 0.0, True, {"end_reason": "sim_unreachable", "error": str(e)})

        if r.status_code >= 400:
            body = r.text[:1000]
            print(
                f"[GuavaSimEnv] /step {self.server_url} episode_id={self._episode_id} "
                f"status={r.status_code} — terminating episode with reward=0\n  body={body!r}",
                flush=True,
            )
            return ("", 0.0, True, {"end_reason": "sim_500", "status": r.status_code, "body": body})

        data = r.json()
        return data["observation"], float(data["reward"]), bool(data["done"]), data.get("info", {})

    async def close(self) -> None:
        if self._episode_id is not None:
            try:
                await self._client.post(
                    f"{self.server_url}/close", json={"episode_id": self._episode_id}
                )
            except Exception:
                # Server may already be gone (shutdown / restart). Don't block.
                pass
            finally:
                self._episode_id = None
        try:
            await self._client.aclose()
        except Exception:
            pass


def _extract_assistant_text(action: Messages) -> str:
    """Find the most recent assistant message and stringify it.

    Some chat templates use list-of-blocks for content; we concatenate the
    text portions so the sim sees the same string a vLLM completion would
    yield in single-turn mode.
    """
    for msg in reversed(action):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(b.get("text", "") for b in content if b.get("type") == "text")
        return str(content)
    return ""


def _extract_text_blocks(observation: Any) -> str:
    """Flatten a multimodal observation (list-of-blocks or str) to plain text.

    Image blocks contribute their placeholder text ("<image>" by default) so
    the rendered chat-template prompt still has an image slot in the right
    place; the rollout server's template applies the real image substitution
    when building the actual model input from the messages list.
    """
    if isinstance(observation, str):
        return observation
    if isinstance(observation, list):
        parts: list[str] = []
        for blk in observation:
            t = blk.get("type") if isinstance(blk, dict) else None
            if t == "text":
                parts.append(blk.get("text", ""))
            elif t in ("image_url", "image"):
                parts.append("<image>")
        return "".join(parts)
    return str(observation)


# ─────────────────────────────────────────────────────────────────────────── #
# 2. MultiTurnScheduler — own the env, emit explicit token_ids + loss_mask
# ─────────────────────────────────────────────────────────────────────────── #
class GuavaMultiTurnScheduler(MultiTurnScheduler):
    """Multi-turn scheduler for guava sim rollouts.

    Why this exists instead of ms-swift's built-in ``gym_scheduler``:
    the built-in scheduler returns ``RolloutOutput`` without explicit
    ``response_token_ids`` / ``response_loss_mask``. The GRPO trainer then
    falls back to re-tokenizing the completed messages list to build its
    loss-mask, which on multi-turn multimodal rollouts produces a 1-token
    drift between the index tensor and the model's logits — manifested as

        RuntimeError: Size does not match at dimension 1, index [B, T+1, 1]
        vs self [B, T, vocab].

    We dodge it entirely by using each turn's ``response_choice.token_ids``
    directly (loss_mask=1) and appending tokenizer-encoded observation
    tokens for masked positions (loss_mask=0).
    """

    def __init__(self, infer_engine, max_turns=None, **kwargs):
        super().__init__(infer_engine, max_turns, **kwargs)
        self.gym_env_name = kwargs.get("gym_env", "guava_sim")

    async def _create_env(self, env_config: Dict[str, Any]) -> GuavaSimEnv:
        env_name = env_config.get("name", self.gym_env_name)
        if env_name not in envs:
            raise ValueError(f"Env '{env_name}' not registered. Available: {list(envs)}")
        return envs[env_name](env_config)

    async def run(self, infer_request, request_config, **kwargs):  # type: ignore[override]
        from swift.utils.utils import remove_response
        from swift.infer_engine.protocol import RolloutOutput

        env_config = (infer_request.data_dict or {}).get("env_config", {})
        env: GuavaSimEnv | None = None

        # Per-turn accumulators (each turn appends one list to these).
        total_response_ids: List[List[int]] = []
        total_response_loss_mask: List[List[int]] = []
        total_reward = 0.0
        trajectory_info: List[Dict[str, Any]] = []
        last_response = None

        try:
            env = await self._create_env(env_config)
            observation, info, system_message = await env.reset(infer_request)
            trajectory_info.append(info)

            messages: Messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": observation})

            current_request = deepcopy(infer_request)
            current_request.messages = messages
            current_turn = 1
            done = False

            # Tokenizer for encoding observation text (for loss_mask=0 padding).
            tokenizer = self.infer_engine.tokenizer

            while not done and current_turn <= (self.max_turns or float("inf")):
                # 1. Model emits an action for the current state.
                remove_response(current_request.messages)
                last_response = await self.infer_engine.infer_async(
                    current_request, request_config, **kwargs
                )
                choice = last_response.choices[0]
                completion_text = choice.message.content or ""
                model_token_ids: List[int] = list(choice.token_ids or [])

                # 2. Update messages with the assistant turn.
                messages.append({"role": "assistant", "content": completion_text})

                # 3. Sim executes the action (parses tool_call from text).
                next_obs, reward, done, step_info = await env.step(deepcopy(messages))
                total_reward += reward
                trajectory_info.append(step_info)

                # 4. Build this turn's response_token_ids + loss_mask.
                turn_ids: List[int] = list(model_token_ids)
                turn_mask: List[int] = [1] * len(turn_ids)

                # Only append next observation if another iteration will run.
                # Tool observations must stay role=tool so the checkpoint's
                # chat template renders them as <tool_response>...</tool_response>,
                # matching the v10 SFT training format.
                will_continue = not done and current_turn < (self.max_turns or float("inf"))
                if will_continue:
                    messages.append({"role": "tool", "content": next_obs})
                    current_request.messages = messages
                    # Tokenize the observation for loss-mask=0 positions so
                    # the trainer's logit index stays aligned with the actual
                    # model output length.
                    obs_text = _extract_text_blocks(next_obs)
                    obs_token_ids = tokenizer.encode(obs_text, add_special_tokens=False)
                    turn_ids.extend(obs_token_ids)
                    turn_mask.extend([0] * len(obs_token_ids))

                total_response_ids.append(turn_ids)
                total_response_loss_mask.append(turn_mask)
                current_turn += 1

            num_turns = current_turn - 1
            base_reward = total_reward
            total_reward, length_penalty = _apply_success_length_penalty(
                base_reward, num_turns,
            )
            if length_penalty:
                print(
                    f"[GuavaMultiTurnScheduler] success length penalty: "
                    f"turns={num_turns} base={base_reward:.3f} "
                    f"penalty={length_penalty:.3f} final={total_reward:.3f}",
                    flush=True,
                )

            return RolloutOutput(
                response=last_response,
                messages=messages,
                response_token_ids=total_response_ids,
                response_loss_mask=total_response_loss_mask,
                rollout_infos={
                    "num_turns": num_turns,
                    "trajectory_id": getattr(infer_request, "uuid", None),
                    "total_reward": total_reward,
                    "base_reward": base_reward,
                    "length_penalty": length_penalty,
                    "trajectory_info": trajectory_info,
                },
            )
        finally:
            if env is not None:
                try:
                    await env.close()
                except Exception:
                    pass


# Registration
envs["guava_sim"] = GuavaSimEnv
envs["guava"] = GuavaSimEnv  # short alias
multi_turns["guava_tool_scheduler"] = GuavaMultiTurnScheduler

print(
    "[guava.rl.swift_gym_env] registered env=guava_sim, "
    "scheduler=guava_tool_scheduler",
    flush=True,
)
