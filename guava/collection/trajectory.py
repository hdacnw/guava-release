from __future__ import annotations
import json
import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np



@dataclass
class Step:
    turn: int
    tool_name: str | None   # None = text-only turn (plan or status)
    tool_args: dict
    tool_result: str
    reasoning: str
    image_b64: str | None
    success: bool
    depth_arr: Any = None    # np.ndarray (H,W) float32 metres, or None
    gripper_state: str = ""  # gripper pose/width text appended to tool result observation
    supervise: bool | None = None  # False for forced/unresolved actions
    reasoning_source: str = 'unknown'
    model_message: dict | None = None  # original API assistant message, if available


@dataclass
class Trajectory:
    trial_id: int
    task_name: str
    plan: list[str]
    steps: list[Step]
    final_reward: float
    success: bool
    turns_used: int
    init_image_b64: str | None = None   # scene image shown to VLM at start
    init_depth_arr: Any = None          # np.ndarray (H,W) float32 metres, or None
    init_gripper_state: str = ""        # gripper state text shown in the initial human turn
    tools_text: str = ""                # human-readable tool descriptions (for human message)
    tools_schema: list[dict] = field(default_factory=list)  # OpenAI-format tool schemas (for "tools" field)
    branch_metadata: dict | None = None
    system_prompt: str | None = None
    runtime_metadata: dict = field(default_factory=dict)
    messages: list[dict] | None = None  # authoritative recorded transcript; required for export
    loss_by_message: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


def save_trajectory(traj: Trajectory, output_dir: str) -> None:
    out = Path(output_dir)
    img_dir   = out / "images"
    depth_dir = out / "depths"
    traj_dir  = out / "trajectories"
    img_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    tag  = "success_1" if traj.success else "success_0"
    path = traj_dir / f"trial_{traj.trial_id:04d}_{tag}.json"
    if list(traj_dir.glob(f'trial_{traj.trial_id:04d}_*.json')):
        raise FileExistsError('Trial ID already exists; refusing to overwrite it or its images')
    record = _to_sharegpt(traj, img_dir, depth_dir)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    # Append to merged fine-tune file (successful trials only)
    if traj.success and traj.quality.get('eligible', True):
        merged = out / "finetune_data.jsonl"
        with merged.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_branch_trajectory(traj: Trajectory, output_dir: str) -> None:
    out = Path(output_dir)
    img_dir   = out / "images_branch"
    depth_dir = out / "depths_branch"
    traj_dir  = out / "trajectories_branch"
    img_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    tag = "success_1" if traj.success else "success_0"
    orig_id = (traj.branch_metadata or {}).get("original_trial_id", 0)
    path = traj_dir / f"trial_{orig_id:04d}_branch_{traj.trial_id:04d}_{tag}.json"
    if list(traj_dir.glob(f'trial_{orig_id:04d}_branch_{traj.trial_id:04d}_*.json')):
        raise FileExistsError('Branch ID already exists; refusing to overwrite it or its images')
    record = _to_sharegpt(traj, img_dir, depth_dir)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    if traj.success and traj.quality.get('eligible', True):
        merged = out / "finetune_data_branch.jsonl"
        with merged.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _to_sharegpt(traj: Trajectory, img_dir: Path, depth_dir: Path) -> dict:
    """Export the exact accepted conversation; never synthesize hidden turns."""
    if not traj.messages or traj.messages[0].get('role') != 'system':
        raise ValueError('Trajectory export requires recorded messages beginning with a system message')
    if not isinstance(traj.messages[0].get('content'), str):
        raise ValueError('System message must contain text')
    branch = traj.branch_metadata
    top_id = branch['original_trial_id'] if branch else traj.trial_id
    prefix = f'trial_{top_id:04d}' + (f'_branch_{traj.trial_id:04d}' if branch else '')
    conversations, images = [], []
    for index, message in enumerate(traj.messages[1:], 1):
        role, content = message['role'], message['content']
        if role not in ('user', 'assistant'):
            raise ValueError(f'Unsupported collection message role: {role}')
        if isinstance(content, str):
            value = content
        else:
            texts, markers = [], []
            for part in content:
                if part['type'] == 'text':
                    texts.append(part['text'])
                elif part['type'] == 'image_url':
                    url = part['image_url']['url']
                    if not url.startswith('data:image/png;base64,'):
                        raise ValueError('Collection images must be embedded PNGs')
                    path = img_dir / f'{prefix}_message{index:04d}_image{len(markers):02d}.png'
                    with path.open('xb') as stream:
                        stream.write(base64.b64decode(url.split(',', 1)[1], validate=True))
                    images.append(str(path.resolve()))
                    markers.append('<image>')
                else:
                    raise ValueError('Unsupported live message content')
            value = '\n'.join(markers + texts)
        entry = {'from': 'gpt' if role == 'assistant' else 'human', 'value': value}
        if role == 'assistant':
            entry['loss'] = traj.loss_by_message.get(index, True)
            if '<tool_call>' in value:
                rationale = re.search(r'<think>(.*?)</think>', value, re.S)
                if not rationale or not rationale[1].strip():
                    raise ValueError('Empty action rationale in accepted transcript')
        conversations.append(entry)
    # Save depth at unique action indices; image filenames are message-indexed.
    for i, step in enumerate(traj.steps):
        if step.depth_arr is not None:
            _save_depth(step.depth_arr, depth_dir / f'{prefix}_step{i:03d}.npy')
    metadata = {'trial_id': top_id, 'task': traj.task_name, 'success': traj.success,
                'final_reward': traj.final_reward, 'turns_used': traj.turns_used,
                'runtime': traj.runtime_metadata, 'quality': traj.quality,
                'reasoning_provenance': [dict(turn=s.turn, source=s.reasoning_source,
                                              model_message=s.model_message) for s in traj.steps]}
    if branch:
        metadata['branch'] = branch
    return {'id': prefix, 'system': traj.messages[0]['content'], 'conversations': conversations,
            'images': images, 'metadata': json.dumps(metadata)}


def _save_depth(arr: np.ndarray, path: Path) -> None:
    np.save(path, arr.astype(np.float32))


def _fmt_tool_result(tool_name: str, tool_args: dict, raw_result: str) -> str:
    """Produce a concise natural-language description of a completed control action."""
    if tool_name in ('grasp', 'close_gripper') and raw_result == 'blocked':
        return 'Gripper closure was obstructed; no object grasp was detected.'
    if tool_name == "move":
        pos = [round(v, 3) for v in tool_args.get("target_position", [])]
        if raw_result == "success":
            return f"Moved gripper to {pos}."
        return f"Move to {pos} failed ({raw_result})."
    if tool_name == "grasp":
        if raw_result == "grasped":
            return "Closed gripper — object grasped."
        return "Closed gripper fully — no object grasped."
    if tool_name == "release":
        return "Opened gripper fully."
    if tool_name == "rotate":
        deg = tool_args.get("angle_deg", 0)
        axis = tool_args.get("axis", "")
        if raw_result == "success":
            return f"Rotated {deg}° around {axis}-axis."
        return f"Rotate {deg}° around {axis}-axis failed ({raw_result})."
    if tool_name == "home_pose":
        return "Reset gripper rotation to face straight vertically down."
    if tool_name == "align":
        target    = tool_args.get("target_name", "object")
        position  = tool_args.get("position", "")
        clearance = tool_args.get("clearance", "")
        base = f"Aligned to {position} of {target} with {clearance} clearance."
        try:
            data = json.loads(raw_result)
            c = data["center"]
            e = data["extents"]
            w, d, h = e[1], e[0], e[2]  # base frame: X=forward(D), Y=lateral(W), Z=up(H)
            return (
                f"{base} {target} is located at {c} and measures approximately "
                f"{w:.2f} x {d:.2f} x {h:.2f} m (W x D x H)."
            )
        except Exception:
            return base
    if tool_name == "get_position":
        obj = tool_args.get("target_name", tool_args.get("object_name", "object"))
        return f"Position of {obj} is at {raw_result}."
    if tool_name == "get_position_and_size":
        obj = tool_args.get("target_name", tool_args.get("object_name", "object"))
        try:
            data = json.loads(raw_result)
            c = data["center"]
            e = data["extents"]
            w, d, h = e[1], e[0], e[2]  # base frame: X=forward(D), Y=lateral(W), Z=up(H)
            return (
                f"{obj.capitalize()} is located at {c} and measures approximately "
                f"{w:.2f} x {d:.2f} x {h:.2f} m (W x D x H)."
            )
        except Exception:
            return raw_result
    return raw_result
