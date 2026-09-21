from __future__ import annotations
import math
import numpy as np
import viser.transforms as vtf
from guava.config import RobotConfig
from guava.sim.env import RoboEnv


class RobotArm:
    """High-level robot arm control: move, grasp, release, rotate."""

    def __init__(self, env: RoboEnv, ik_solver, cfg: RobotConfig):
        self.env = env
        self.ik = ik_solver
        self.cfg = cfg
        self._tcp_offset = np.array(cfg.tcp_offset, dtype=np.float64)
        self._last_commanded_quat: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move(self, target_pos: list[float], target_quat: np.ndarray | None = None) -> str:
        """Move the EEF to target_pos (robot base frame). Orientation preserved if not given."""
        target = np.asarray(target_pos, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"target_pos must have 3 elements, got {target.shape}")

        current_pos, current_quat = self.env.eef_pose_base()
        if target_quat is None:
            # Use the last explicitly commanded quaternion so the IK solver always
            # targets a known, stable orientation rather than the FK-measured one
            # (which can drift slightly from IK inaccuracy across moves).
            # Seed from FK on first use (sim is fully ready by the time move() is called).
            if self._last_commanded_quat is None:
                self._last_commanded_quat = current_quat.copy()
            quat = self._last_commanded_quat
        else:
            quat = np.asarray(target_quat, dtype=np.float64)
            self._last_commanded_quat = quat.copy()

        # Apply TCP offset
        offset_world = vtf.SO3(wxyz=quat).as_matrix() @ self._tcp_offset
        target_with_offset = target + offset_world

        # print(f"[move] commanded target (fingertip): {target.round(4)}")
        # print(f"[move] tcp offset in world:           {offset_world.round(4)}")
        # print(f"[move] IK target (EEF body):          {target_with_offset.round(4)}")

        import os as _os
        if _os.environ.get("DUMP_IK"):
            print(f"[ik-input] target_pos={np.array(target_with_offset).round(4).tolist()}  "
                  f"target_quat={np.array(quat).round(4).tolist()}  "
                  f"seed_joints={self.env._current_joints.round(4).tolist()}", flush=True)
        joints = self.ik.solve(target_with_offset, quat, self.env._current_joints)
        if _os.environ.get("DUMP_IK"):
            print(f"[ik-output] joints={joints.round(4).tolist()}", flush=True)
        self._move_to_joints(joints)
        if _os.environ.get("DUMP_IK"):
            ap, aq = self.env.eef_pose_base()
            print(f"[ik-after] actual_pos={ap.round(4).tolist()}  actual_quat={aq.round(4).tolist()}", flush=True)

        # actual_eef, _ = self.env.eef_pose_base()
        # print(f"[move] actual EEF body after move:    {actual_eef.round(4)}")                                 
        # print(f"[move] estimated fingertip after move: {(actual_eef - offset_world).round(4)}")

        return "success"

    def _move_to_joints(self, target: np.ndarray, tolerance: float = 0.02, max_steps: int = 100) -> None:
        target = np.asarray(target, dtype=np.float64).reshape(7)
        self.env._current_joints = target

        start = self.env.joint_positions()
        n_waypoints = max(1, int(np.ceil(1.0 / self.env.cfg.motion_speed)))

        for i in range(1, n_waypoints + 1):
            waypoint = start + (i / n_waypoints) * (target - start)
            steps = 0
            while steps < max_steps:
                current = self.env.joint_positions()
                if np.linalg.norm(current - waypoint) < tolerance:
                    break
                self.env.step(self.env.build_action(waypoint))
                steps += 1

        if self.cfg.motion_debug:
            print(f"  [ctrl] requested  joints: {target.round(3)}")
            print(f"  [ctrl] achieved   joints: {self.env.joint_positions().round(3)}")
            print(f"  [ctrl] residual (per j) : {(target - self.env.joint_positions()).round(3)}")

        self.env._settle(10)

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def grasp(self) -> str:
        """Close gripper until stall (object contact) or fully closed.

        Reports "grasped" only when the fingers stopped while still meaningfully
        open (an object is wedged). Reports "closed" when the fingers reached
        the joint limit (empty close) or never moved (already closed).
        """
        cfg = self.cfg
        # Per-finger qpos floor — below this the fingers are at the close limit
        # (no object between them). Tuned for Panda: fully-closed ≈ 0.001 m per
        # finger; thinnest realistic grasp ≈ 0.0025 m.
        _MIN_GRASP_QPOS = 0.0015
        start_qpos = self.env.gripper_qpos()
        prev = start_qpos

        for _ in range(cfg.grasp_max_iters):
            new_frac = max(0.0, self.env._gripper_fraction - 0.05)
            self.env._gripper_fraction = new_frac
            for _ in range(cfg.grasp_steps_per_iter):
                self.env._settle(1)

            current = self.env.gripper_qpos()
            # Early exit: stalled (delta tiny) AND fingers above the close
            # limit AND have actually moved from start → object is wedged.
            if (abs(current - prev) < cfg.grasp_stall_delta
                and current > _MIN_GRASP_QPOS
                and abs(current - start_qpos) > cfg.grasp_stall_delta):
                return "grasped"

            prev = current
            if new_frac <= 0.0:
                break

        # Loop ended without an early grasp. Catches soft objects that
        # compressed slowly without ever triggering the stall test.
        final = self.env.gripper_qpos()
        if (final > _MIN_GRASP_QPOS
            and abs(final - start_qpos) > cfg.grasp_stall_delta):
            return "grasped"
        return "closed"

    def align_down(self) -> str:
        """Orient the gripper to face straight down (roll=180°, pitch=0°), preserving current yaw."""
        current_pos, current_quat = self.env.eef_pose_base()
        w, x, y, z = current_quat
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        straight_down = np.array([0.0, np.cos(yaw/2), np.sin(yaw/2), 0.0])
        return self.move(current_pos.tolist(), straight_down)

    def init_orientation(self) -> None:
        """Seed the orientation reference from current FK after env.reset()."""
        _, _q = self.env.eef_pose_base()
        self._last_commanded_quat = _q.copy()

    def home_pose(self) -> str:
        """Move to the robot's initial home joint configuration."""
        self._move_to_joints(np.array(self.cfg.home_joints, dtype=np.float64))
        _, _q = self.env.eef_pose_base()
        self._last_commanded_quat = _q.copy()
        return "success"

    def release(self) -> str:
        """Open gripper gradually so the held object isn't launched by the
        spring back from a closed → fully-open jump."""
        cfg = self.cfg
        for _ in range(cfg.grasp_max_iters):
            new_frac = min(1.0, self.env._gripper_fraction + 0.05)
            self.env._gripper_fraction = new_frac
            for _ in range(cfg.grasp_steps_per_iter):
                self.env._settle(1)
            if new_frac >= 1.0:
                break
        self.env._settle(10)
        return "released"

    def rotate(self, angle_deg: float, axis: str) -> str:
        """Rotate EEF in place by angle_deg around body-frame axis (x/y/z)."""
        axis = axis.strip().lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be x, y or z, got '{axis}'")

        current_pos, current_quat = self.env.eef_pose_base()
        half = math.radians(angle_deg) / 2.0
        v = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
        dq = np.array([math.cos(half), math.sin(half)*v[0], math.sin(half)*v[1], math.sin(half)*v[2]])

        # Body-frame rotation: new = current * dq
        w0, x0, y0, z0 = current_quat
        w1, x1, y1, z1 = dq
        new_quat = np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
        ])
        new_quat /= np.linalg.norm(new_quat)
        return self.move(current_pos.tolist(), new_quat)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def position(self) -> list[float]:
        pos, _ = self.env.eef_pose_base()
        return pos.tolist()

    def rotation(self) -> list[float]:
        _, quat = self.env.eef_pose_base()
        return quat.tolist()

    def gripper_width(self) -> float:
        return float(self.env.gripper_qpos() * 2.0)
