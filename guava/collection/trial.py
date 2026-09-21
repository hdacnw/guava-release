from __future__ import annotations
import json
from pathlib import Path

from guava.tools.tools import _refresh, _encode, _gripper_state_text

_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text().strip()


def run_trial(env, arm, perception, provider, cfg, trial_id: int, *, grasp_planner=None):
    from guava.collection.rollout import configure_context, continue_rollout
    base_seed = cfg.sim.randomize.seed
    episode_seed = trial_id if base_seed is None else base_seed + trial_id
    env.reset(seed=episode_seed)
    arm.init_orientation()
    from guava.sim.video import record_episode
    video_path = Path(cfg.output_dir) / 'videos' / f'trial_{trial_id:04d}.mp4'
    with record_episode(env, video_path, getattr(cfg, 'record_video', False), getattr(cfg, 'video_fps', 20)):
        ctx, registry, system, text, schema, runtime = configure_context(
            env, arm, perception, cfg, grasp_planner)
        runtime['episode_seed'] = episode_seed
        if getattr(cfg, 'record_video', False):
            runtime['video'] = {'path': str(video_path.resolve()), 'fps': cfg.video_fps}
        _refresh(ctx)
        task = getattr(env.rob, 'task_instruction', None) or cfg.task_name
        messages = _init_messages(task, _encode(ctx.last_rgb), text, _gripper_state_text(ctx))
        if cfg.sim.perception == 'gt':
            messages[1]['content'][-1]['text'] += '\n\nAccepted object names: ' + ', '.join(cfg.sim.object_aliases)
        messages[0]['content'] = system
        return continue_rollout(ctx, registry, provider, cfg, trial_id, messages, runtime,
                                tools_text=text, tools_schema=schema, save_checkpoints=True)

# ─────────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _init_messages(task: str, image_b64: str, tools_text: str, gripper_state: str = "") -> list[dict]:
    body = f"Task: {task}\n\nAvailable tools:\n{tools_text}"
    if gripper_state:
        body = f"{body}\n\n{gripper_state}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": body},
        ]},
    ]


def _round_floats(obj, ndigits: int = 3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def _execute(registry: dict, name: str, args: dict) -> tuple[str, bool]:
    if name not in registry:
        return f"Unknown tool '{name}'", False
    try:
        result = registry[name](**args)
        if isinstance(result, (list, dict)):
            return json.dumps(_round_floats(result)), True
        return str(result), True
    except Exception as e:
        return f"Error: {e}", False
