import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

from guava.sim.tasks.apple_juice_order import (
    AppleObject, JuiceObject, _camera_right_xy,
    _SORT_MIN_GAP,
    _DEFAULT_APPLE_X_RANGE, _DEFAULT_APPLE_Y_RANGE,
    _DEFAULT_JUICE_X_RANGE, _DEFAULT_JUICE_Y_RANGE,
)

_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)


class AppleJuiceReverseOrder(ManipulationEnv):
    """Arrange juice (larger, left) then apple (smaller, right) by decreasing size in camera view."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 apple_x_range=None, apple_y_range=None,
                 juice_x_range=None, juice_y_range=None,
                 **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer
        self._cam_right_xy: np.ndarray | None = None
        self.apple_x_range = apple_x_range if apple_x_range is not None else _DEFAULT_APPLE_X_RANGE
        self.apple_y_range = apple_y_range if apple_y_range is not None else _DEFAULT_APPLE_Y_RANGE
        self.juice_x_range = juice_x_range if juice_x_range is not None else _DEFAULT_JUICE_X_RANGE
        self.juice_y_range = juice_y_range if juice_y_range is not None else _DEFAULT_JUICE_Y_RANGE

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            horizon=horizon,
            renderer=renderer,
            render_camera=render_camera,
            **kwargs,
        )

    def _sort_key(self, pos):
        return float(np.dot(pos[:2], self._cam_right_xy))

    def score(self):
        from guava.sim.scoring import snapshot
        return score_state(snapshot(self, ['apple', 'juice'], geometry=True))

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        if self._cam_right_xy is None:
            return 0.0

        apple_pos = self.sim.data.body_xpos[self.apple_body_id]
        juice_pos = self.sim.data.body_xpos[self.juice_body_id]

        # juice (larger) must be to the left of apple (smaller)
        gap = self._sort_key(apple_pos) - self._sort_key(juice_pos)

        if self.reward_shaping:
            return (1 + np.tanh(5.0 * (gap - _SORT_MIN_GAP))) / 2.0

        return 0.0

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.apple = AppleObject(name="apple")
        self.juice = JuiceObject(name="juice")

        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for obj, x_range, y_range in [
            (self.apple, self.apple_x_range, self.apple_y_range),
            (self.juice, self.juice_x_range, self.juice_y_range),
        ]:
            sampler.append_sampler(UniformRandomSampler(
                name=f"{obj.name}Sampler",
                mujoco_objects=[obj],
                x_range=x_range,
                y_range=y_range,
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))
        self.placement_initializer = sampler

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.apple, self.juice],
        )

    def _setup_references(self):
        super()._setup_references()
        self.apple_body_id = self.sim.model.body_name2id(self.apple.root_body)
        self.juice_body_id = self.sim.model.body_name2id(self.juice.root_body)
        cam_name = self.camera_names[0] if isinstance(self.camera_names, list) else self.camera_names
        cam_id   = self.sim.model.camera_name2id(cam_name)
        self._cam_right_xy = _camera_right_xy(self.sim.model.cam_quat[cam_id])

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def _check_success(self):
        return self.score()['success']


def make_apple_juice_reverse_order(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    apple_rand = getattr(cfg, "apple_randomize", None) or {}
    juice_rand = getattr(cfg, "juice_randomize", None) or {}

    return AppleJuiceReverseOrder(
        robots=["Panda"],
        controller_configs=controller_cfg,
        has_renderer=cfg.visualize,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cfg.camera_views,
        camera_heights=cfg.camera.height,
        camera_widths=cfg.camera.width,
        camera_depths=True,
        reward_shaping=True,
        horizon=500000,
        renderer="mjviewer" if cfg.use_mjviewer else "mujoco",
        render_camera=cfg.camera.name,
        apple_x_range=apple_rand.get("x_range"),
        apple_y_range=apple_rand.get("y_range"),
        juice_x_range=juice_rand.get("x_range"),
        juice_y_range=juice_rand.get("y_range"),
    )


def score_state(state):
    from guava.sim.tasks.apple_juice_order import score_state as ordering_score
    return ordering_score(state, reverse=True)
