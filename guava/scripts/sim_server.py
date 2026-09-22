"""HTTP server exposing guava's multi-turn sim rollout as REST endpoints.

Architecture
------------
    .venv (this server)            .venv-train (swift rollout / rlhf)
    ─────────────────────          ──────────────────────────────────
      RolloutEngine cache    HTTP    GuavaSimEnv (thin httpx client)
      RolloutEnv per         ◄───►   registered as ms-swift gym env
      episode_id

One sim_server PROCESS owns one EGL display (process-global, not
thread-safe) and a single-thread executor. Within a process every /reset
and /step serializes. Parallelism comes from running N independent
sim_server processes on different ports — see scripts/run_grpo_full.sh
and GuavaSimEnv's URL sharding for how the trainer picks one per episode.

Endpoints
---------
    POST /reset → {episode_id, system_message, observation, task}
    POST /step  → {observation, reward, done, info}
    POST /close → {closed}
    GET  /health → {status, engines_cached, episodes_open}

Usage
-----
    .venv/bin/python -m guava.scripts.sim_server --port 8200
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="guava-sim")


# ─────────────────────────────────────────────────────────────────────────── #
# Request / response models
# ─────────────────────────────────────────────────────────────────────────── #

class ResetRequest(BaseModel):
    task_config: str
    seed: int = 0
    episode_id: str | None = None
    output_dir: str | None = None
    system_prompt_name: str | None = None


class ResetResponse(BaseModel):
    episode_id: str
    system_message: str
    observation: Any  # multimodal content list
    task: str


class StepRequest(BaseModel):
    episode_id: str
    assistant_text: str


class StepResponse(BaseModel):
    observation: Any
    reward: float
    done: bool
    info: dict[str, Any]


class CloseRequest(BaseModel):
    episode_id: str


# ─────────────────────────────────────────────────────────────────────────── #
# Server state
# ─────────────────────────────────────────────────────────────────────────── #

_engines: dict[tuple[str, str | None], Any] = {}
_engine_lock = asyncio.Lock()       # protects _engines construction
# Single shared executor (max_workers=1) — all sim ops within this process
# serialize on its one thread. EGL display + mujoco contexts stay bound to
# that thread. Parallelism is achieved across processes, not threads.
_sim_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
_episodes: dict[str, dict[str, Any]] = {}  # episode_id → {env, last_used, task_config}
_episode_lock = asyncio.Lock()       # protects _episodes registry
_rollout_counter = 0                 # monotonic int → RolloutEnv.episode_id (dir name suffix)
# The counter is re-bucketed at main() startup based on this process's port,
# so multiple sim_server replicas writing to a shared output_dir can't
# collide on episode_NNNN/. See main() below.

DEFAULT_OUTPUT_DIR = os.environ.get("GUAVA_SIM_OUTPUT_DIR", "/tmp/guava_sim_rollouts")
EPISODE_TTL_SEC = int(os.environ.get("GUAVA_SIM_TTL_SEC", "1800"))  # 30 min
# Bucket size per sim_server process for episode_id. With BASE_PORT 8200
# and BUCKET 100_000: port 8200 → episode 0..99_999, port 8201 → 100_000..
# 199_999, etc. 100 k headroom per run is more than ample.
_PORT_BASE = int(os.environ.get("GUAVA_SIM_PORT_BASE", "8200"))
_BUCKET_SIZE = int(os.environ.get("GUAVA_SIM_BUCKET_SIZE", "100000"))


async def _get_engine(task_config: str, system_prompt_name: str | None):
    """Build-or-fetch a RolloutEngine on the shared sim thread."""
    key = (task_config, system_prompt_name)
    async with _engine_lock:
        if key not in _engines:
            from guava.inference.rollout_env import RolloutEngine

            loop = asyncio.get_event_loop()
            t0 = time.time()
            engine = await loop.run_in_executor(
                _sim_executor,
                lambda: RolloutEngine(
                    task_config=task_config,
                    system_prompt_name=system_prompt_name,
                ),
            )
            _engines[key] = engine
            print(
                f"[sim_server] engine built for {task_config} (prompt={system_prompt_name}) "
                f"in {time.time() - t0:.1f}s",
                flush=True,
            )
        return _engines[key]


def _save_env_trajectory(env) -> str | None:
    """Run RolloutEnv.save_trajectory() on the sim thread. Best-effort —
    swallows exceptions so a corrupt episode doesn't kill the server.

    Returns the output dir path on success, or None on failure.
    """
    try:
        out = env.save_trajectory()
        return str(out.parent) if hasattr(out, "parent") else str(out)
    except Exception as e:
        print(f"[sim_server] save_trajectory failed: {type(e).__name__}: {e}", flush=True)
        return None


async def _flush_episode(env) -> None:
    """Persist trajectory.jsonl / messages.json / trajectory.md for `env`
    on the shared sim thread. Idempotent — RolloutEnv.save_trajectory()
    rewrites the same files."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_sim_executor, _save_env_trajectory, env)


async def _gc_idle_episodes() -> None:
    """Drop episodes that haven't seen a /step in EPISODE_TTL_SEC.
    Persist their trajectories first so artifacts aren't lost when a
    rollout dies without calling /close.
    """
    now = time.time()
    stale_envs: list[tuple[str, Any]] = []
    async with _episode_lock:
        for eid, ent in list(_episodes.items()):
            if now - ent["last_used"] > EPISODE_TTL_SEC:
                stale_envs.append((eid, ent["env"]))
                _episodes.pop(eid, None)
    for eid, env in stale_envs:
        await _flush_episode(env)
    if stale_envs:
        ids = [eid for eid, _ in stale_envs]
        print(f"[sim_server] GC saved + dropped {len(stale_envs)} idle episode(s): {ids[:5]}…", flush=True)


# ─────────────────────────────────────────────────────────────────────────── #
# Endpoints
# ─────────────────────────────────────────────────────────────────────────── #

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engines_cached": len(_engines),
        "episodes_open": len(_episodes),
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest) -> ResetResponse:
    await _gc_idle_episodes()

    engine = await _get_engine(req.task_config, req.system_prompt_name)
    episode_id = req.episode_id or uuid.uuid4().hex
    output_dir = req.output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Unique int episode index so K rollouts of the same seed don't share an
    # `episode_NNNN/` directory (RolloutEnv formats episode_id as %04d).
    global _rollout_counter
    async with _episode_lock:
        rollout_idx = _rollout_counter
        _rollout_counter += 1

    loop = asyncio.get_event_loop()
    try:
        t0 = time.time()
        env = await loop.run_in_executor(
            _sim_executor,
            lambda: engine.episode(episode_id=rollout_idx, output_dir=output_dir),
        )
        t1 = time.time()
        result = await loop.run_in_executor(_sim_executor, lambda: env.reset(seed=req.seed))
        t2 = time.time()
        print(
            f"[sim_server] /reset {req.task_config} "
            f"episode={t1-t0:.2f}s reset={t2-t1:.2f}s total={t2-t0:.2f}s",
            flush=True,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[sim_server] /reset FAILED task={req.task_config} seed={req.seed}\n{tb}", flush=True)
        raise HTTPException(500, f"engine.reset failed: {type(e).__name__}: {e}")

    msgs = result.get("messages") or []
    if len(msgs) < 2:
        raise HTTPException(500, "RolloutEnv.reset returned malformed messages")
    sys_msg = msgs[0].get("content", "")
    observation = msgs[1].get("content", "")
    task_name = result.get("task", "")

    async with _episode_lock:
        _episodes[episode_id] = {
            "env": env,
            "last_used": time.time(),
            "task_config": req.task_config,
        }

    return ResetResponse(
        episode_id=episode_id,
        system_message=sys_msg if isinstance(sys_msg, str) else str(sys_msg),
        observation=observation,
        task=task_name,
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    async with _episode_lock:
        ent = _episodes.get(req.episode_id)
    if ent is None:
        raise HTTPException(404, f"unknown episode_id: {req.episode_id}")

    ent["last_used"] = time.time()
    env = ent["env"]
    loop = asyncio.get_event_loop()
    try:
        t_queued = time.time()
        result = await loop.run_in_executor(
            _sim_executor, lambda: env.step(req.assistant_text)
        )
        t_done = time.time()
        print(
            f"[sim_server] /step {ent['task_config']} "
            f"wall={t_done-t_queued:.2f}s",
            flush=True,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(
            f"[sim_server] /step FAILED episode_id={req.episode_id} task={ent['task_config']}\n"
            f"assistant_text[:300]={req.assistant_text[:300]!r}\n{tb}",
            flush=True,
        )
        raise HTTPException(500, f"env.step failed: {type(e).__name__}: {e}")

    msgs = result.get("messages") or []
    last_msg = msgs[-1] if msgs else {"content": ""}
    next_obs = last_msg.get("content", "")
    info = {
        "tool_call": result.get("tool_call"),
        "tool_result": result.get("tool_result"),
        "image_path": result.get("image_path"),
        **(result.get("info") or {}),
    }
    return StepResponse(
        observation=next_obs,
        reward=float(result.get("reward", 0.0)),
        done=bool(result.get("done", False)),
        info=info,
    )


@app.post("/close")
async def close(req: CloseRequest) -> dict[str, Any]:
    """End an episode: persist its trajectory artifacts (trajectory.jsonl,
    messages.json, trajectory.md) and drop the env from the registry.

    Idempotent — calling /close on an unknown id is a no-op.
    """
    async with _episode_lock:
        ent = _episodes.pop(req.episode_id, None)
    if ent is None:
        return {"closed": True, "saved": False}
    await _flush_episode(ent["env"])
    return {"closed": True, "saved": True}


@app.on_event("shutdown")
async def _flush_open_episodes_on_shutdown() -> None:
    """On graceful shutdown (SIGTERM via uvicorn), persist any open
    episodes that the trainer never /closed. With the GRPO script's
    bash trap sending SIGTERM, uvicorn's lifespan triggers this hook."""
    async with _episode_lock:
        envs = [ent["env"] for ent in _episodes.values()]
        _episodes.clear()
    if envs:
        print(f"[sim_server] shutdown: flushing {len(envs)} open episode(s)…", flush=True)
        for env in envs:
            await _flush_episode(env)


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point
# ─────────────────────────────────────────────────────────────────────────── #

def main(
    host: str = "127.0.0.1",
    port: int = 8200,
    log_level: str = "info",
    timeout_keep_alive: int = 300,
) -> None:
    """Launch one sim HTTP server process.

    Each process owns one EGL display and one sim thread. To get parallelism,
    launch multiple processes on different ports (the trainer picks one per
    episode). See scripts/run_grpo_full.sh.
    """
    # Re-bucket the episode_id counter so this process's IDs can't collide
    # with sibling sim_server replicas writing into the same output_dir.
    # port 8200 → counter starts at 0, 8201 → 100_000, 8202 → 200_000, …
    # The next /reset call after this gets episode_id == counter, then ++.
    global _rollout_counter
    bucket = max(0, port - _PORT_BASE) * _BUCKET_SIZE
    _rollout_counter = bucket
    print(
        f"[sim_server] starting on {host}:{port}  "
        f"(TTL={EPISODE_TTL_SEC}s, timeout_keep_alive={timeout_keep_alive}s, "
        f"episode_id starts at {bucket})",
        flush=True,
    )
    uvicorn.run(
        app, host=host, port=port, log_level=log_level,
        timeout_keep_alive=timeout_keep_alive,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
