import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config


_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)
_CUSTOM_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects/robocasa_objects"
)

# tomato_1 native scale (mesh scale=0.06; reg_bbox half-extents: 0.0321x 0.0319y 0.0291z)
_TOMATO_HALF_HEIGHT = 0.029078
_TOMATO_HALF_XY     = 0.032051

# potato_0 native scale (mesh scale=0.09; reg_bbox half-extents: 0.0372x 0.0455y 0.0364z)
_POTATO_HALF_HEIGHT = 0.036363
_POTATO_HALF_XY     = 0.045465

# Success: tomato center within this distance of potato center, resting on table
_NEAR_DIST = 0.15

_POTATO_X_RANGE  = [-0.12, 0.12]
_POTATO_Y_RANGE  = [0.08, 0.20]
_TOMATO_X_RANGE  = [-0.15, 0.15]
_TOMATO_Y_RANGE  = [-0.25, -0.10]


class TomatoObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_CUSTOM_DIR / "tomato_1/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_TOMATO_HALF_HEIGHT])

    @property
    def top_offset(self):
        return np.array([0, 0, _TOMATO_HALF_HEIGHT])

    @property
    def horizontal_radius(self):
        return _TOMATO_HALF_XY


class PotatoObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_CUSTOM_DIR / "potato_0/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_POTATO_HALF_HEIGHT])

    @property
    def top_offset(self):
        return np.array([0, 0, _POTATO_HALF_HEIGHT])

    @property
    def horizontal_radius(self):
        return _POTATO_HALF_XY


class TomatoNearPotato(ManipulationEnv):
    """Place the tomato near the potato."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 potato_x_range=None, potato_y_range=None,
                 tomato_x_range=None, tomato_y_range=None,
                 **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer
        self.potato_x_range = potato_x_range if potato_x_range is not None else _POTATO_X_RANGE
        self.potato_y_range = potato_y_range if potato_y_range is not None else _POTATO_Y_RANGE
        self.tomato_x_range = tomato_x_range if tomato_x_range is not None else _TOMATO_X_RANGE
        self.tomato_y_range = tomato_y_range if tomato_y_range is not None else _TOMATO_Y_RANGE

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

    def reset_scoring(self):
        self._initial_distance = float(np.linalg.norm(
            self.sim.data.body_xpos[self.tomato_body_id][:2] -
            self.sim.data.body_xpos[self.potato_body_id][:2]))

    def score(self):
        tomato = self.sim.data.body_xpos[self.tomato_body_id]
        potato = self.sim.data.body_xpos[self.potato_body_id]
        state = {'objects': {
            'tomato': {'position_world': tomato.tolist(), 'held': bool(self._check_grasp(
                gripper=self.robots[0].gripper, object_geoms=self.tomato))},
            'potato': {'position_world': potato.tolist()}},
            'center_above_table_m': float(tomato[2] - self.table_offset[2])}
        return score_state(state, getattr(self, '_initial_distance', 0.0))

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        tomato_pos = self.sim.data.body_xpos[self.tomato_body_id]
        potato_pos = self.sim.data.body_xpos[self.potato_body_id]

        xy_dist  = np.linalg.norm(tomato_pos[:2] - potato_pos[:2])
        grasping = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.tomato)

        if self.reward_shaping:
            dist = min(
                np.linalg.norm(
                    self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - tomato_pos
                )
                for arm in self.robots[0].arms
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
            if grasping:
                r_reach += 0.25
            r_lift  = 0.25 if tomato_pos[2] > self.table_offset[2] + 0.05 else 0.0
            r_place = 0.25 * max(0.0, 1.0 - xy_dist / _NEAR_DIST) if grasping else 0.0
            return max(r_reach, r_lift + r_place)

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

        self.tomato = TomatoObject(name="tomato")
        self.potato = PotatoObject(name="potato")

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects([self.tomato])
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.tomato],
                x_range=self.tomato_x_range,
                y_range=self.tomato_y_range,
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.tomato, self.potato],
        )

    def _setup_references(self):
        super()._setup_references()
        self.tomato_body_id = self.sim.model.body_name2id(self.tomato.root_body)
        self.potato_body_id = self.sim.model.body_name2id(self.potato.root_body)

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        potato_x = np.random.uniform(*self.potato_x_range)
        potato_y = np.random.uniform(*self.potato_y_range)
        potato_pos = np.array([
            self.table_offset[0] + potato_x,
            self.table_offset[1] + potato_y,
            self.table_offset[2] + _POTATO_HALF_HEIGHT,
        ])
        self.sim.data.set_joint_qpos(
            self.potato.joints[0],
            np.concatenate([potato_pos, [1, 0, 0, 0]]),
        )

    def _check_success(self):
        return self.score()['success']


def make_tomato_near_potato(cfg: "SimConfig"):
    from robosuite.utils.placement_samplers import UniformRandomSampler

    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    potato_rand = getattr(cfg, "potato_randomize", None) or {}
    tomato_rand = getattr(cfg, "tomato_randomize", None) or {}

    env = TomatoNearPotato(
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
        potato_x_range=potato_rand.get("x_range"),
        potato_y_range=potato_rand.get("y_range"),
        tomato_x_range=tomato_rand.get("x_range"),
        tomato_y_range=tomato_rand.get("y_range"),
    )

    rotation = tomato_rand.get("rotation", True)
    env.placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=[env.tomato],
        x_range=tomato_rand.get("x_range", _TOMATO_X_RANGE),
        y_range=tomato_rand.get("y_range", _TOMATO_Y_RANGE),
        rotation=None if rotation else 0,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=env.table_offset,
        z_offset=0.01,
    )
    return env


import math

MAX_DISTANCE_RATIO = 0.75


def distance(state):
    o = state['objects']
    return math.dist(o['tomato']['position_world'][:2],
                     o['potato']['position_world'][:2])


def score_state(state, initial_distance):
    final_distance = distance(state)
    ratio = final_distance / initial_distance if initial_distance > 1e-9 else None
    checks = {
        'distance_reduced_25_percent': ratio is not None and ratio <= MAX_DISTANCE_RATIO,
        'released': not state['objects']['tomato']['held'],
        'table_height': state['center_above_table_m'] < _TOMATO_HALF_HEIGHT + .02,
    }
    return {'version': 'tomato-relative-distance-075-v1', 'success': all(checks.values()), 'checks': checks,
            'initial_distance_m': initial_distance, 'final_distance_m': final_distance,
            'distance_ratio': ratio}
