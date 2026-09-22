"""RolloutEnv — multi-turn evaluation env wrapping guava's existing sim + tools.

Layered on top of:
- `guava.sim.tasks.make_env`        (robosuite task)
- `guava.robot.arm.RobotArm`        (IK + gripper control)
- `guava.perception`               (configured position/grasp perception)
- `guava.tools.tools.make_tool_registry`  (nine physical tools)
- `guava.tools.coordinate_frame`   (canonical names and table-frame adapter)

Public surface (designed so it can be wrapped into ms-swift's `Env` ABC later
without changing internals):

    env = RolloutEnv(task_config="configs/lemon_in_bin.yaml",
                     output_dir="runs/eval_<ts>",
                     episode_id=0)
    obs0 = env.reset(seed=0)            # -> dict {messages, image_path, task}
    while not obs0.get("done"):
        response = model.generate(obs0["messages"])
        obs0 = env.step(response)        # -> dict {messages, reward, done, ...}

Each `obs` dict is also written verbatim into a per-episode trajectory.jsonl so
post-hoc analysis can replay everything without re-running sim.
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from guava.config import CollectionConfig
from guava.inference.xml_tool_parser import parse as parse_xml
from guava.inference.prompts import load as load_system_prompt
from guava.tools.tools import ToolContext, _refresh, _gripper_state_text, make_tool_registry
from guava.collection.trajectory import _fmt_tool_result


# Tools that change the visible scene → image must be re-rendered after.
# Queries do not mutate the scene; these actions require a fresh image.
_STATE_MUTATING = {"move", "grasp", "release", "rotate", "align", "close_gripper", "home_pose"}

# Tasks whose terminal success must remain true through short settling windows.
_STABLE_REWARD_TASKS = {"red_objects_in_basket"}


# ─────────────────────────────────────────────────────────────────────────── #
# Config loading — same shape as collection, but we override a few defaults
# for headless eval (no visualize, training-resolution camera, etc.)
# ─────────────────────────────────────────────────────────────────────────── #

def load_eval_config(yaml_path: str | Path) -> CollectionConfig:
    """Load a collection-style YAML and return CollectionConfig with eval overrides."""
    cfg = CollectionConfig()
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    for key, val in raw.items():
        if hasattr(cfg, key) and not isinstance(val, dict):
            setattr(cfg, key, val)
    if "sim" in raw:
        for k, v in raw["sim"].items():
            if k == "camera" and isinstance(v, dict):
                for ck, cv in v.items():
                    if hasattr(cfg.sim.camera, ck):
                        setattr(cfg.sim.camera, ck, cv)
            elif k == "randomize" and isinstance(v, dict):
                for rk, rv in v.items():
                    if hasattr(cfg.sim.randomize, rk):
                        setattr(cfg.sim.randomize, rk, rv)
            elif hasattr(cfg.sim, k):
                setattr(cfg.sim, k, v)
    if "robot" in raw:
        for k, v in raw["robot"].items():
            if hasattr(cfg.robot, k):
                setattr(cfg.robot, k, v)
    # Eval overrides — never visualize.
    # Camera resolution is taken from the YAML — training was at that size, so
    # touching it here causes a distribution shift the model can't recover from.
    # Perception backend is also taken from the YAML so SAM3 / GT is the user's
    # choice; defaulting to GT here would silently break grasp() and align().
    cfg.sim.visualize = False
    cfg.sim.use_mjviewer = False
    # Keep the configured camera distribution. RL task configs deliberately
    # select the view used during training (Shell Game uses agentview).
    if not cfg.sim.camera_views:
        cfg.sim.camera_views = [cfg.sim.camera.name]
    return cfg


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers — image encode/save
# ─────────────────────────────────────────────────────────────────────────── #

def _save_png(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb).save(path)


def _get_image_transport() -> str:
    """Return file (shared filesystem) or base64 (remote API compatibility)."""
    transport = os.environ.get("GUAVA_IMAGE_TRANSPORT", "file").lower()
    if transport not in {"file", "base64"}:
        raise ValueError("GUAVA_IMAGE_TRANSPORT must be 'file' or 'base64'")
    return transport


_INLINE_IMG_SIZE = int(os.environ.get("GUAVA_INLINE_IMAGE_SIZE", "256"))


def _to_data_url(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb)
    if _INLINE_IMG_SIZE > 0 and max(img.size) > _INLINE_IMG_SIZE:
        img = img.resize((_INLINE_IMG_SIZE, _INLINE_IMG_SIZE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _image_message_url(rgb: np.ndarray, saved_path: Path) -> str:
    """Use a short local path for training, or an inline image for remote APIs."""
    if _get_image_transport() == "file":
        return str(saved_path.resolve())
    return _to_data_url(rgb)


# ─────────────────────────────────────────────────────────────────────────── #
# RolloutEnv
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class StepRecord:
    """One step of a rollout — what was attempted and what happened."""
    turn: int
    tool_call: dict[str, Any] | None     # {name, arguments} or None for finished/text-only
    tool_result: str | None
    tool_ok: bool
    image_path: str | None               # set only for state-mutating tools
    task_reward: float
    finished: bool


class RolloutEngine:
    """Persistent rollout components — built once per task, reused across episodes.

    Owns the heavy / stateful resources whose recreation per-episode introduced
    bugs (PyBullet client leak being the canonical example: each new
    IKPoseSolver opened a fresh `p.connect(p.DIRECT)` client and PyBullet's
    `physicsClientId` defaults silently aliased subsequent IK calls onto the
    wrong robot, accumulating quat drift across episodes).

    Mirrors `guava.collection.runner._build_components`: persistent sim/ik/arm/
    perception live here; per-episode state (messages, episode_dir, turn) lives
    in `RolloutEnv`.
    """

    def __init__(
        self,
        task_config: str | Path,
        system_prompt: str | None = None,
        system_prompt_name: str | None = None,
        camera_name: str | None = None,
    ):
        from guava.sim.tasks import make_env
        from guava.robot.arm import RobotArm
        from guava.robot.ik import get_ik_solver
        from guava.perception import get_position_solver

        self.cfg = load_eval_config(task_config)
        self.cfg.model_frame = 'table'
        if camera_name is not None:
            self.cfg.sim.camera_views = [camera_name]
            self.cfg.sim.camera.name = camera_name
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = load_system_prompt(system_prompt_name)

        self.env = make_env(self.cfg.sim)
        ik = get_ik_solver(self.cfg.robot, self.env)
        self.arm = RobotArm(self.env, ik, self.cfg.robot)
        # output_dir for SAM3/bbox debug is set per-episode via update_debug_dir;
        # most eval runs don't enable debug, so None here is fine as a default.
        self.perception = get_position_solver(
            self.cfg.robot, self.cfg.sim, self.env, output_dir=None,
        )
        from guava.perception import get_graspgen_client
        planner = get_graspgen_client(self.cfg.robot, self.cfg.sim.grasp)
        self.ctx = ToolContext(env=self.env, arm=self.arm, perception=self.perception,
                              grasp_planner=planner,
                              graspgen_horizontal=self.cfg.robot.graspgen_horizontal,
                              graspgen_vertical=self.cfg.robot.graspgen_vertical)
        self.registry = make_tool_registry(self.ctx)
        # Keep the engine registry physical for benchmark_engine's own adapter.
        # RolloutEnv wraps it per episode; never replace it in place.

    def episode(self, episode_id: int, output_dir: str | Path) -> "RolloutEnv":
        """Factory: create a per-episode RolloutEnv bound to this engine."""
        return RolloutEnv(engine=self, episode_id=episode_id, output_dir=output_dir)


class RolloutEnv:
    """Per-episode state on top of a `RolloutEngine`.

    Holds messages/history/turn/episode_dir. Sim, IK, arm, perception, and tool
    registry are owned by the engine and shared across episodes.
    """

    def __init__(
        self,
        *,
        engine: RolloutEngine | None = None,
        episode_id: int = 0,
        output_dir: str | Path,
        # Legacy single-episode constructor — kept for callers (smoke tests,
        # probe scripts) that don't want to manage an engine. Prefer
        # `RolloutEngine(...).episode(...)` for multi-episode loops to avoid
        # the per-episode rebuild that hides resource-leak bugs.
        task_config: str | Path | None = None,
        system_prompt: str | None = None,
        system_prompt_name: str | None = None,
    ):
        if engine is None:
            if task_config is None:
                raise ValueError("RolloutEnv: pass either `engine=` or `task_config=`.")
            engine = RolloutEngine(
                task_config=task_config,
                system_prompt=system_prompt,
                system_prompt_name=system_prompt_name,
            )
        self.engine = engine
        # Convenience aliases — keep the old attribute names so tool execution
        # / save_trajectory / smoke tests continue working unchanged.
        self.cfg = engine.cfg
        self.system_prompt = engine.system_prompt
        self.env = engine.env
        self.arm = engine.arm
        self.perception = engine.perception
        self.ctx = engine.ctx
        self.registry = engine.registry

        self.episode_id = episode_id
        self.episode_dir = Path(output_dir) / f"episode_{episode_id:04d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        # Per-episode state
        self.turn = 0
        self.messages: list[dict] = []
        self.history: list[StepRecord] = []
        self.done = False

    def _task_audit(self) -> dict[str, Any]:
        """Return authoritative task-owned success when the task exposes it."""
        scorer = getattr(self.env.rob, "score", None)
        if callable(scorer):
            audit = scorer()
            if not isinstance(audit, dict) or "success" not in audit:
                raise TypeError("task score() must return a dict containing 'success'")
            return audit
        reward = float(self.env.rob.reward(action=None))
        return {
            "version": "legacy-reward-v1",
            "success": reward >= 1.0,
            "legacy_reward": reward,
        }

    @staticmethod
    def _audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
        return {
            key: audit[key]
            for key in (
                "version",
                "success",
                "checks",
                "metrics",
                "legacy_reward",
            )
            if key in audit
        }

    def _stable_task_audit(self) -> tuple[float, dict[str, Any]]:
        audits = [self._task_audit()]
        if self.cfg.sim.task in _STABLE_REWARD_TASKS:
            for _ in range(3):
                self.env._settle(5)
                audits.append(self._task_audit())
        success = all(bool(audit["success"]) for audit in audits)
        return float(success), {
            "success_at_stop": bool(audits[0]["success"]),
            "stable_success": success,
            "stable_audits": [self._audit_summary(audit) for audit in audits],
        }

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed: int = 0) -> dict[str, Any]:
        """Start a new episode. Returns the initial observation dict.

        The dict includes `messages`, ready to be sent to the model as the
        first chat completion request. Images use shared-file transport by
        default to avoid tokenizing a large base64 URL on every turn.
        """
        self.env.reset(seed=seed)
        settle_steps = getattr(self.engine, 'initial_settle_steps', 0)
        if settle_steps:
            self.arm.init_orientation()
            self.env._settle(settle_steps)
        if hasattr(self.env.rob, 'reset_scoring'):
            self.env.rob.reset_scoring()
        from guava.tools.coordinate_frame import measured_offset, model_registry
        from guava.tools.schema import build_tools_schema
        self.ctx.model_z_offset = measured_offset(self.env)
        self.registry = model_registry(self.ctx, self.engine.registry)
        self.tools_schema = build_tools_schema(self.registry)
        (self.episode_dir / 'coordinate_calibration.json').write_text(json.dumps({
            'coordinate_frame': 'table_aligned', 'model_z_equals_base_z_plus': self.ctx.model_z_offset,
            'argument_names': 'canonical-target_name', 'system_prompt': self.system_prompt,
            'tools': self.tools_schema}, indent=2))
        # Force a fresh render
        self.ctx.last_rgb = None
        self.ctx.last_depth = None
        self.ctx.last_K = None
        self.ctx.last_pose_mat = None
        _refresh(self.ctx)

        init_path = self.episode_dir / f"turn_000_init.png"
        _save_png(self.ctx.last_rgb, init_path)
        image_url = _image_message_url(self.ctx.last_rgb, init_path)

        # The image block becomes the single <image> placeholder. Do not add a
        # second literal placeholder to the text.
        gripper_line = _gripper_state_text(self.ctx)
        task = getattr(self.env.rob, "task_instruction", None) or self.cfg.task_name
        first_user_text = f"\nTask: {task}\n\n{gripper_line}"
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": first_user_text},
            ]},
        ]
        self.turn = 0
        self.history = []
        self.done = False

        return {
            "messages": self.messages,
            "image_path": str(init_path),
            "task": self.cfg.task_name,
            "episode_id": self.episode_id,
            "turn": 0,
            "done": False,
        }

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, assistant_response: str) -> dict[str, Any]:
        """Apply one assistant turn. Append messages, execute tool, refresh image.

        Returns:
            {messages, reward, done, image_path, tool_call, tool_result, info}
        """
        if self.done:
            raise RuntimeError("Episode already done — call reset() before step().")
        self.turn += 1

        # Append assistant message verbatim — preserves <think>/<tool_call> XML
        # so the next round of generation sees the model's own prior reasoning.
        self.messages.append({"role": "assistant", "content": assistant_response})

        parsed = parse_xml(assistant_response)
        tool_call = parsed["tool_call"]
        finished = parsed["finished"]

        # Case 1: model says task is done (no tool call, "Task complete." / "Task failed.")
        if finished and tool_call is None:
            return self._finalize(
                tool_call=None,
                tool_result=None,
                tool_ok=True,
                image_path=None,
                end_reason="model_declared_done",
            )

        # Case 2: model emitted no tool call AND no completion marker → format error
        if tool_call is None:
            return self._finalize(
                tool_call=None,
                tool_result="No tool_call and no completion marker emitted.",
                tool_ok=False,
                image_path=None,
                end_reason="format_error_no_tool_call",
            )

        # Case 3: tool execution
        result_text, ok = self._execute_tool(tool_call["name"], tool_call["arguments"])
        is_mutating = tool_call["name"] in _STATE_MUTATING

        image_path = None
        obs_parts: list[dict] = []
        # The image block supplies <image>; the leading newline preserves the
        # SFT tool-turn format without duplicating the placeholder.
        text_payload = f"\n{result_text}" if (ok and is_mutating) else result_text
        if ok and is_mutating:
            _refresh(self.ctx)
            image_path = self.episode_dir / f"turn_{self.turn:03d}_{tool_call['name']}.png"
            _save_png(self.ctx.last_rgb, image_path)
            obs_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": _image_message_url(self.ctx.last_rgb, image_path),
                },
            })
        obs_parts.append({"type": "text", "text": text_payload})

        # The model expects tool returns wrapped in role=tool, but our chat_template
        # renders role=tool as <|im_start|>user + <tool_response>. So we use role=tool
        # so the chat_template can do the wrapping correctly when serving via vllm.
        # When vllm doesn't have the latest qwen template, falling back to user is OK
        # too (the legacy collection code did this). We default to "tool" for fidelity.
        tool_role = "tool" if _get_image_transport() == "file" else "user"
        self.messages.append({"role": tool_role, "content": obs_parts})

        # Task-owned score() is the authoritative binary RL signal. Shaped
        # reward remains available inside the simulator but cannot count as
        # success.
        task_audit = self._task_audit()
        task_reward = float(bool(task_audit["success"]))
        is_success = bool(task_audit["success"])

        # Reach end conditions
        if is_success and self.cfg.stop_on_reward:
            return self._finalize(
                tool_call=tool_call,
                tool_result=result_text,
                tool_ok=ok,
                image_path=str(image_path) if image_path else None,
                end_reason="task_success",
                reward=1.0,
                give_closing_turn=True,
            )

        if self.turn >= self.cfg.turn_budget:
            return self._finalize(
                tool_call=tool_call,
                tool_result=result_text,
                tool_ok=ok,
                image_path=str(image_path) if image_path else None,
                end_reason="turn_budget_exceeded",
                reward=0.0,
            )

        if getattr(self.env, "_episode_done", False):
            return self._finalize(
                tool_call=tool_call,
                tool_result=result_text,
                tool_ok=ok,
                image_path=str(image_path) if image_path else None,
                end_reason="sim_horizon",
                reward=0.0,
            )

        # Step continues — record and return.
        record = StepRecord(
            turn=self.turn, tool_call=tool_call,
            tool_result=result_text, tool_ok=ok,
            image_path=str(image_path) if image_path else None,
            task_reward=task_reward, finished=False,
        )
        self.history.append(record)

        return {
            "messages": self.messages,
            "reward": 0.0,
            "done": False,
            "image_path": str(image_path) if image_path else None,
            "tool_call": tool_call,
            "tool_result": result_text,
            "tool_ok": ok,
            "turn": self.turn,
            "info": {
                "task_reward": task_reward,
                "task_audit": self._audit_summary(task_audit),
            },
        }

    # ------------------------------------------------------------------ #
    # Internal: tool execution mirrors collection/trial.py:_execute
    # ------------------------------------------------------------------ #

    def _execute_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        if name not in self.registry:
            return f"Unknown tool '{name}'. Available: {sorted(self.registry)}", False
        try:
            raw = self.registry[name](**args)
            # Format the raw return into the rich descriptive string the model
            # was trained on (training data goes through guava.collection.trajectory
            # ._fmt_tool_result; eval used to skip it, exposing strings like
            # "grasped"/"closed" the model never saw at training time).
            if isinstance(raw, (list, dict)):
                raw_str = json.dumps(_round_floats(raw))
            else:
                raw_str = str(raw)
            descriptive = _fmt_tool_result(name, args, raw_str)
            gripper_line = _gripper_state_text(self.ctx)
            return f"{descriptive}\n{gripper_line}", True
        except Exception as e:
            return f"Error calling {name}: {e}", False

    def _finalize(
        self,
        tool_call: dict | None,
        tool_result: str | None,
        tool_ok: bool,
        image_path: str | None,
        end_reason: str,
        reward: float | None = None,
        give_closing_turn: bool = False,
    ) -> dict[str, Any]:
        audit_info: dict[str, Any] = {}
        if self.env and self.cfg.sim.task in _STABLE_REWARD_TASKS:
            reward, audit_info = self._stable_task_audit()
        elif self.env:
            audit = self._task_audit()
            if reward is None:
                reward = float(bool(audit["success"]))
            audit_info = {"task_audit": self._audit_summary(audit)}
        elif reward is None:
            reward = 0.0
        self.history.append(StepRecord(
            turn=self.turn, tool_call=tool_call,
            tool_result=tool_result, tool_ok=tool_ok,
            image_path=image_path, task_reward=reward,
            finished=True,
        ))
        self.done = True
        return {
            "messages": self.messages,
            "reward": reward,
            "done": True,
            "image_path": image_path,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "tool_ok": tool_ok,
            "turn": self.turn,
            "info": {
                "end_reason": end_reason,
                "task_reward": reward,
                **audit_info,
            },
        }

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def save_trajectory(self) -> Path:
        """Write a self-contained jsonl of this episode's history."""
        out = self.episode_dir / "trajectory.jsonl"
        with out.open("w") as f:
            for rec in self.history:
                f.write(json.dumps({
                    "turn": rec.turn,
                    "tool_call": rec.tool_call,
                    "tool_result": rec.tool_result,
                    "tool_ok": rec.tool_ok,
                    "image_path": rec.image_path,
                    "task_reward": rec.task_reward,
                    "finished": rec.finished,
                }) + "\n")
        # Also save the full message history (with base64 stripped to file refs
        # for readability — saved images on disk are the source of truth).
        msg_out = self.episode_dir / "messages.json"
        with msg_out.open("w") as f:
            json.dump(_strip_base64(self.messages), f, indent=2, ensure_ascii=False)
        return out


def _round_floats(obj, ndigits: int = 3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def _strip_base64(messages: list[dict]) -> list[dict]:
    """Replace inline base64 data URLs with a `<image_b64>` marker for readable JSON."""
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            new_parts = []
            for p in m["content"]:
                if p.get("type") == "image_url":
                    new_parts.append({"type": "image_url", "image_url": {"url": "<image_b64>"}})
                else:
                    new_parts.append(p)
            out.append({**m, "content": new_parts})
        else:
            out.append(m)
    return out
