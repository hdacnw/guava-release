import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CanObject
from robosuite.models.objects.composite.bin import Bin
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

from guava.sim.randomization import make_placement_sampler

_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)

_BIN_SIZE = (0.15, 0.15, 0.07)
_BIN_X_RANGE = [-0.10, 0.10]
_BIN_Y_RANGE = [0.10, 0.25]


class CanInBin(ManipulationEnv):
    """Place the apple into the open-top bin fixed on the table."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 bin_x_range=None, bin_y_range=None, **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer
        self.bin_x_range = bin_x_range if bin_x_range is not None else _BIN_X_RANGE
        self.bin_y_range = bin_y_range if bin_y_range is not None else _BIN_Y_RANGE

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

    def score(self):
        from guava.sim.scoring import snapshot
        return score_state(snapshot(self, ["apple","bin"]))

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        apple_pos = self.sim.data.body_xpos[self.apple_body_id]
        grasping  = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.apple)

        if self.reward_shaping:
            dist = min(
                np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - apple_pos)
                for arm in self.robots[0].arms
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
            if grasping:
                r_reach += 0.25
            r_lift = 0.25 if apple_pos[2] > self.table_offset[2] + 0.04 else 0.0
            return max(r_reach, r_lift)

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

        self.apple = CanObject(name="can")
        self.bin   = Bin(name="bin", bin_size=_BIN_SIZE, transparent_walls=False, density=5000.0)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects([self.apple])
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.apple],
                x_range=[-0.15, 0.15],
                y_range=[-0.15, 0.0],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.apple, self.bin],
        )

    def _setup_references(self):
        super()._setup_references()
        self.apple_body_id = self.sim.model.body_name2id(self.apple.root_body)
        self.bin_body_id   = self.sim.model.body_name2id(self.bin.root_body)

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        bin_x = np.random.uniform(*self.bin_x_range)
        bin_y = np.random.uniform(*self.bin_y_range)
        bin_pos = np.array([
            self.table_offset[0] + bin_x,
            self.table_offset[1] + bin_y,
            self.table_offset[2] + _BIN_SIZE[2] / 2,
        ])
        self.sim.data.set_joint_qpos(
            self.bin.joints[0],
            np.concatenate([bin_pos, [1, 0, 0, 0]]),
        )

    def _check_success(self):
        return self.score()['success']


def make_can_in_bin(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    bin_rand = getattr(cfg, "bin_randomize", None) or {}
    env = CanInBin(
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
        bin_x_range=bin_rand.get("bin_x_range"),
        bin_y_range=bin_rand.get("bin_y_range"),
    )
    env.placement_initializer = make_placement_sampler(
        objects=[env.apple],
        cfg=cfg.randomize,
        reference_pos=env.table_offset,
    )
    return env


def score_state(state):
    """Authoritative task success criteria, independently testable from physics."""
    from guava.sim.scoring import predicates, result
    p = predicates(state)
    touch, released, supported, upright, inside = p.touch, p.released, p.supported, p.upright, p.inside
    o = state['objects']
    metrics = {}

    checks = {'bin_supported': supported('bin'), 'bin_upright': upright('bin'),
              'can_inside': inside('apple', 'bin', .055, -.035, .055),
              'can_contacts_bin': touch('apple', 'bin'), 'released': released('apple')}
    return result(checks, metrics)
