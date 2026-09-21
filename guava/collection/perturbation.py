from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np


def _get_base_to_world(env) -> np.ndarray:
    sim = env.rob.sim
    base_id = sim.model.body_name2id("fixed_mount0_base")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = sim.data.xmat[base_id].reshape(3, 3).copy()
    T[:3, 3] = sim.data.xpos[base_id].copy()
    return T


@dataclass
class SetObjectPose:
    """Move a named object to target_pos (robot base frame) with optional quaternion (wxyz).
    If pos_delta is given instead, the object is displaced by that amount from its current position."""
    object_name: str
    target_pos: list[float] | None = None
    pos_delta: list[float] | None = None
    target_quat: list[float] | None = None

    def apply_to_sim(self, env) -> None:
        sim = env.rob.sim
        joint_name = self._find_free_joint(sim)
        if joint_name is None:
            raise RuntimeError(f"No free joint found for object '{self.object_name}'")
        jid = sim.model.joint_name2id(joint_name)
        if sim.model.jnt_type[jid] != 0:
            raise ValueError('Object perturbation requires a free joint')
        qadr = sim.model.jnt_qposadr[jid]
        T = _get_base_to_world(env)
        if self.target_pos is not None:
            target_world = (T @ np.append(self.target_pos, 1.0))[:3]
        else:
            cur_world = sim.data.qpos[qadr : qadr + 3].copy()
            target_world = cur_world + T[:3, :3] @ np.array(self.pos_delta)
        sim.data.qpos[qadr : qadr + 3] = target_world
        if self.target_quat is not None:
            sim.data.qpos[qadr + 3 : qadr + 7] = self.target_quat
        dof = sim.model.jnt_dofadr[jid]
        sim.data.qvel[dof:dof + 6] = 0
        sim.forward()

    def _find_free_joint(self, sim) -> str | None:
        # Try the name as-given AND with spaces→underscores. Object aliases in
        # yaml configs use spaces for LLM friendliness ("pink bowl"), but the
        # underlying sim joint/body names use underscores ("pink_bowl_joint0").
        name_variants = (self.object_name, self.object_name.replace(" ", "_"))
        for name in name_variants:
            for candidate in (f"{name}_joint0", f"{name}_jnt"):
                try:
                    sim.model.joint_name2id(candidate)
                    return candidate
                except Exception:
                    pass
            for suffix in ("", "_main", "_body"):
                try:
                    body_id = sim.model.body_name2id(f"{name}{suffix}")
                    for j in range(sim.model.njnt):
                        if sim.model.jnt_bodyid[j] == body_id and sim.model.jnt_type[j] == 0:
                            return sim.model.joint_id2name(j)
                except Exception:
                    pass
        return None

    @property
    def detail(self) -> str:
        if self.target_pos is not None:
            return f"{self.object_name} moved to {[round(v, 3) for v in self.target_pos]}"
        return f"{self.object_name} displaced by {self.pos_delta}"


@dataclass
class SetGripperPose:
    """Displace the gripper TCP. target_pos is absolute (base frame); pos_delta is added to
    the current EEF pos at hook time — useful when the final EEF pos isn't known ahead of time."""
    target_pos: list[float] | None = None
    pos_delta: list[float] | None = None
    target_quat: list[float] | None = None

    @property
    def detail(self) -> str:
        if self.target_pos is not None:
            return f"gripper displaced to {[round(v, 3) for v in self.target_pos]}"
        return f"gripper displaced by delta {self.pos_delta}"


@dataclass
class SubstituteArgument:
    """Replace one argument in the tool call dict before execution."""
    arg_name: str
    new_value: Any

    def apply_to_args(self, tool_args: dict) -> dict:
        result = dict(tool_args)
        result[self.arg_name] = self.new_value
        return result

    @property
    def detail(self) -> str:
        return f"argument '{self.arg_name}' → '{self.new_value}'"


Perturbation = SetObjectPose | SetGripperPose | SubstituteArgument
