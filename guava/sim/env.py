import os
os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import replace   

import numpy as np
import viser.transforms as vtf
from guava.config import CameraConfig
from guava.sim.camera import render_camera


class RoboEnv:
    """Thin wrapper adding camera rendering and observation helpers to a robosuite env."""

    def __init__(self, rob_env, cfg: "SimConfig"):
        self.rob = rob_env   # raw robosuite environment
        self.cfg = cfg
        self._cam_cfg: CameraConfig | None = None
        self._gripper_fraction = 1.0  # 1.0=open, 0.0=closed
        self._current_joints = np.zeros(7)
        self._sim_steps = 0
        self._trial_idx = 0  # incremented each reset for auto-seeding when seed=None
        self._episode_done = False

        # Indices into sim.data arrays — set after first reset
        self._base_idx: int | None = None
        self._eef_idx: int | None = None

        # Visual randomizer — created lazily on first randomized reset
        self._randomizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict:
        # Seed the placement sampler's RNG for reproducible object positions.
        # Use an explicit seed if given, otherwise derive one from trial index.
        rand_cfg = self.cfg.randomize
        effective_seed = seed if seed is not None else (rand_cfg.seed + self._trial_idx if rand_cfg.seed is not None else None)
        if effective_seed is not None:
            import random
            np.random.seed(effective_seed)
            random.seed(effective_seed)
        # Hard reset recreates task samplers. Seed the new sampler tree just
        # before placement, not the stale sampler from the previous episode.
        original_reset = self.rob._reset_internal
        previous_override = vars(self.rob).get('_reset_internal')
        def seeded_reset():
            if effective_seed is not None:
                rng = np.random.default_rng(effective_seed)
                self.rob.rng = rng
                def seed_sampler(sampler):
                    if sampler is None:
                        return
                    sampler.rng = rng
                    for child in getattr(sampler, 'samplers', {}).values():
                        seed_sampler(child)
                seed_sampler(getattr(self.rob, 'placement_initializer', None))
            return original_reset()
        self.rob._reset_internal = seeded_reset
        try:
            self.rob.reset()
        finally:
            if previous_override is None:
                del self.rob._reset_internal
            else:
                self.rob._reset_internal = previous_override

        # Apply solver options after every reset — robust to hard resets that
        # recompile the model and wipe programmatic changes made in __init__.
        opt = self.rob.sim.model.opt
        opt.integrator = 3  # implicitfast: semi-implicit, more stable than Euler
        opt.noslip_iterations = 50  # extra contact iterations to resist table penetration

        # Gate placement-task success on an explicit release() call. Reset here
        # so the flag starts False each episode; the release tool sets it True.
        self.rob._release_called = False

        self._episode_done = False
        self._trial_idx += 1

        for _ in range(50):
            self.rob.sim.forward()
            self.rob.sim.step()
        self._gripper_fraction = 1.0
        obs = self.rob._get_observations()
        self._current_joints = np.array(obs["robot0_joint_pos"], dtype=np.float64)
        self._base_idx = self.rob.sim.model.body_name2id("fixed_mount0_base")
        self._eef_idx = self.rob.sim.model.body_name2id("gripper0_right_eef")

        # Apply visual domain randomization (colors / lighting)
        if rand_cfg.colors or rand_cfg.lighting:
            if self._randomizer is None:
                from guava.sim.randomization import VisualRandomizer
                self._randomizer = VisualRandomizer(self.rob)
            vis_rng = np.random.default_rng(
                effective_seed
            )
            self._randomizer.randomize(vis_rng, rand_cfg)

        # Select and apply the active camera before settling so the viewer
        # shows the correct view from the first frame. rob.reset() recreates
        # the viewer (hard_reset=True), so this must come after rob.reset().
        if self.cfg.camera_views:
            cam_seed = effective_seed
            cam_name = str(np.random.default_rng(cam_seed).choice(self.cfg.camera_views))
            self.set_camera(replace(self.cfg.camera, name=cam_name))
        else:
            self.set_camera(self.cfg.camera)

        if self.cfg.visualize:
            self._settle(30)
        if hasattr(self.rob, "reset_scoring"):
            self.rob.reset_scoring()
        return obs

    def step(self, action: np.ndarray) -> None:
        if self._episode_done:
            raise RuntimeError("Episode terminated (horizon reached).")
        _, _, done, _ = self.rob.step(action)   
        if self.cfg.visualize:                                                                              
            self.rob.render()    
        if getattr(self, "_video", None) is not None:
            self._video.on_step()
        self._sim_steps += 1
        if done:
            self._episode_done = True

    # ------------------------------------------------------------------
    # Joint / gripper helpers
    # ------------------------------------------------------------------

    def joint_positions(self) -> np.ndarray:
        # Control must use current physics state, not a sampled observation
        # whose update clock can lag by one control step after restoration.
        return self.rob.sim.data.qpos[self.rob.robots[0]._ref_joint_pos_indexes].copy()

    def gripper_qpos(self) -> float:
        indexes = self.rob.robots[0]._ref_gripper_joint_pos_indexes['right']
        return float(self.rob.sim.data.qpos[indexes[0]])

    def build_action(self, joints: np.ndarray) -> np.ndarray:
        """Build an 8-element action: [joints(7), gripper(1)]. 1.0=open→-1.0, 0.0=closed→+1.0."""
        g = 1.0 - self._gripper_fraction * 2.0
        return np.concatenate([joints, [g]])

    def _settle(self, n: int = 10) -> None:
        """Hold current joints for n steps — syncs viewer, settles dynamics."""
        for _ in range(n):
            self.step(self.build_action(self._current_joints))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def set_camera(self, cfg: CameraConfig) -> None:
        self._cam_cfg = cfg
        self.rob._scoring_camera = cfg.name
        viewer = getattr(self.rob, 'viewer', None)
        if viewer is not None:
            viewer.set_camera(self.rob.sim.model.camera_name2id(cfg.name))

    def capture(self) -> np.ndarray:
        """Render and return (H, W, 3) uint8 RGB using the configured camera."""
        cfg = self._cam_cfg or self.cfg.camera
        return render_camera(self.rob.sim, cfg)

    # ------------------------------------------------------------------
    # Pose helpers (world / base frame)
    # ------------------------------------------------------------------

    def base_wxyz_xyz(self) -> np.ndarray:
        i = self._base_idx
        return np.concatenate([self.rob.sim.data.xquat[i], self.rob.sim.data.xpos[i]])

    def eef_wxyz_xyz(self) -> np.ndarray:
        i = self._eef_idx
        return np.concatenate([self.rob.sim.data.xquat[i], self.rob.sim.data.xpos[i]])

    def eef_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos, quat_wxyz) of EEF in robot base frame."""
        base = vtf.SE3(wxyz_xyz=self.base_wxyz_xyz())
        eef  = vtf.SE3(wxyz_xyz=self.eef_wxyz_xyz())
        tf = base.inverse() @ eef
        return tf.translation(), tf.rotation().wxyz

    def obs(self) -> dict:
        return self.rob._get_observations()
