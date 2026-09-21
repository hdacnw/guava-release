from __future__ import annotations
import base64
import json
import pickle
import gzip
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import numpy as np


@dataclass
class Checkpoint:
    turn_idx: int
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    gripper_fraction: float
    current_joints: np.ndarray
    eef_pos: np.ndarray          # base frame; used for displacement calculations
    message_count: int           # len(messages) at save time; indexes into conversation list
    tool_name: str               # action about to execute
    tool_args: dict
    prev_tool_name: str | None
    trial_id: int
    grasped_object: str | None   # object body name nearest to gripper if closed, else None
    version: int = 1
    snapshot: dict | None = None
    messages: list | None = None
    collection_id: str | None = None
    prefix_digest: str | None = None
    assistant_content: str | None = None
    runtime_metadata: dict | None = None


def save(
    env,
    messages: list[dict],
    tool_name: str,
    tool_args: dict,
    prev_tool_name: str | None,
    trial_id: int,
    turn_idx: int,
    object_body_names: list[str] = (),
) -> Checkpoint:
    eef_pos, _ = env.eef_pose_base()
    return Checkpoint(
        turn_idx=turn_idx,
        qpos=env.rob.sim.data.qpos.copy(),
        qvel=env.rob.sim.data.qvel.copy(),
        ctrl=env.rob.sim.data.ctrl.copy(),
        gripper_fraction=env._gripper_fraction,
        current_joints=env._current_joints.copy(),
        eef_pos=eef_pos.copy(),
        message_count=len(messages),
        tool_name=tool_name,
        tool_args=tool_args,
        prev_tool_name=prev_tool_name,
        trial_id=trial_id,
        grasped_object=_find_grasped_object(env, list(object_body_names)),
    )


def save_to_disk(ck: Checkpoint, output_dir: str, trial_id: int) -> Path:
    ck_dir = Path(output_dir) / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    path = ck_dir / f"trial_{trial_id:04d}_turn{ck.turn_idx:03d}.pkl"
    stored = ck
    if getattr(ck, 'snapshot', None):
        snapshot = dict(ck.snapshot)
        payload = {k: snapshot.pop(k) for k in ('xml', 'model_arrays', 'model_binary') if k in snapshot}
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        model_hash = hashlib.sha256(blob).hexdigest()
        model_dir = ck_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        asset = model_dir / f'{model_hash}.pkl.gz'
        try:
            with asset.open('xb') as file:
                with gzip.GzipFile(fileobj=file, mode='wb', mtime=0, compresslevel=3) as stream:
                    stream.write(blob)
        except FileExistsError:
            # Content-addressed immutable asset; readers verify its hash.
            pass
        snapshot['model_sha256'] = model_hash
        stored = replace(ck, snapshot=snapshot)
    with open(path, "xb") as f:
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0, compresslevel=3) as stream:
            pickle.dump(stored, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_from_disk(path: Path) -> Checkpoint:
    from guava.collection.safe_checkpoint import load_checkpoint
    return load_checkpoint(path)


def reconstruct_messages(traj_path: Path, message_count: int) -> list[dict]:
    """Reconstruct OpenAI-format messages from a saved ShareGPT trajectory JSON.

    message_count is len(messages) at checkpoint save time; the conversations list
    has message_count - 1 entries (system is stored separately).
    """
    record = json.loads(traj_path.read_text())
    system = record["system"]
    conversations = record["conversations"]
    image_paths = record["images"]

    messages: list[dict] = [{"role": "system", "content": system}]
    img_idx = 0

    for entry in conversations[: message_count - 1]:
        from_role = entry["from"]
        value = entry["value"]

        if from_role in ("human", "tool"):
            parts: list[dict] = []
            if value.startswith("<image>\n"):
                img_b64 = base64.b64encode(Path(image_paths[img_idx]).read_bytes()).decode()
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
                img_idx += 1
                value = value[len("<image>\n"):]
            parts.append({"type": "text", "text": value})
            messages.append({"role": "user", "content": parts})

        elif from_role == "gpt":
            messages.append({"role": "assistant", "content": value})

    return messages


def _find_grasped_object(env, object_body_names):
    """Use bilateral finger contact, not proximity or a previous grasp command."""
    sim = env.rob.sim
    for name in dict.fromkeys(object_body_names):
        for variant in dict.fromkeys((name, name.replace(' ', '_'))):
            for suffix in ('', '_main', '_body'):
                try:
                    root = sim.model.body_name2id(variant + suffix)
                except (ValueError, KeyError):
                    continue
                bodies = {root}
                for bid in range(sim.model.nbody):
                    if int(sim.model.body_parentid[bid]) in bodies:
                        bodies.add(bid)
                geoms = [sim.model.geom_id2name(g) for g in range(sim.model.ngeom)
                         if int(sim.model.geom_bodyid[g]) in bodies]
                geoms = [g for g in geoms if g]
                if geoms and any(env.rob._check_grasp(robot.gripper, geoms) for robot in env.rob.robots):
                    return name
    return None
