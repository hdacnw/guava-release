import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)
_ROBOCASA_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects/robocasa_objects"
)

_CUBE_HALF = 0.024

# Tray geometry (tray_1, scale 0.21)
_TRAY_HALF_X = 0.10407
_TRAY_HALF_Y = 0.07390
_TRAY_HALF_Z = 0.00898

_DEFAULT_TRAY_X_RANGE = [-0.15, 0.15]
_DEFAULT_TRAY_Y_RANGE = [0.05, 0.20]


class TrayObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ROBOCASA_DIR / "tray_1/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_TRAY_HALF_Z])

    @property
    def top_offset(self):
        return np.array([0, 0, _TRAY_HALF_Z])

    @property
    def horizontal_radius(self):
        return float(np.linalg.norm([_TRAY_HALF_X, _TRAY_HALF_Y]))


class RemoveCubeFromTray(ManipulationEnv):
    """Red cube starts on the tray. Goal: lift it off and set it on the table."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 tray_x_range=None, tray_y_range=None, **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.tray_x_range    = tray_x_range if tray_x_range is not None else _DEFAULT_TRAY_X_RANGE
        self.tray_y_range    = tray_y_range if tray_y_range is not None else _DEFAULT_TRAY_Y_RANGE

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
        """Authoritative task success, independent of reward shaping."""
        from guava.sim.scoring import result
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        tray_pos = self.sim.data.body_xpos[self.tray_body_id]
        grasping = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

        # Success: cube at table level AND outside the tray footprint, not held.
        # The XY check prevents a false positive when the cube sinks through the
        # thin tray mesh at reset and lands on the table directly below the tray.
        at_table_level = cube_pos[2] < self.table_offset[2] + _CUBE_HALF + 0.015
        outside_tray = (
            abs(cube_pos[0] - tray_pos[0]) > _TRAY_HALF_X - 0.01 or
            abs(cube_pos[1] - tray_pos[1]) > _TRAY_HALF_Y - 0.01
        )
        return result({'at_table_level': at_table_level, 'outside_tray': outside_tray, 'released': not grasping})

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        tray_pos = self.sim.data.body_xpos[self.tray_body_id]
        grasping = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

        if self.reward_shaping:
            dist = min(
                np.linalg.norm(
                    self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - cube_pos
                )
                for arm in self.robots[0].arms
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
            r_grasp = 0.25 if grasping else 0.0
            tray_surface_z = tray_pos[2] + _TRAY_HALF_Z
            lifted = cube_pos[2] > tray_surface_z + _CUBE_HALF + 0.02
            r_lift = 0.25 if lifted else 0.0
            return r_reach + r_grasp + r_lift

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

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redmat = CustomMaterial(texture="WoodRed", tex_name="redwood",
                                mat_name="redwood_mat",
                                tex_attrib=tex_attrib, mat_attrib=mat_attrib)

        self.cube = BoxObject(name="cube", size_min=[_CUBE_HALF] * 3,
                              size_max=[_CUBE_HALF] * 3,
                              rgba=[1, 0, 0, 1], material=redmat)
        self.tray = TrayObject(name="tray")

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, self.tray],
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.tray_body_id = self.sim.model.body_name2id(self.tray.root_body)

    def _reset_internal(self):
        super()._reset_internal()

        rng = getattr(self, 'rng', None)
        if rng is None:
            rng = np.random.default_rng()
        tray_x = rng.uniform(*self.tray_x_range)
        tray_y = rng.uniform(*self.tray_y_range)
        tray_pos = np.array([
            self.table_offset[0] + tray_x,
            self.table_offset[1] + tray_y,
            self.table_offset[2] + _TRAY_HALF_Z + 0.001,
        ])
        self.sim.data.set_joint_qpos(
            self.tray.joints[0],
            np.concatenate([tray_pos, [1, 0, 0, 0]]),
        )

        cube_pos = tray_pos.copy()
        cube_pos[2] = tray_pos[2] + _TRAY_HALF_Z + _CUBE_HALF
        self.sim.data.set_joint_qpos(
            self.cube.joints[0],
            np.concatenate([cube_pos, [1, 0, 0, 0]]),
        )

    def _check_success(self):
        return self.score()['success']


def make_remove_cube_from_tray(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    tray_rand = getattr(cfg, "tray_randomize", None)
    return RemoveCubeFromTray(
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
        tray_x_range=getattr(tray_rand, "x_range", None),
        tray_y_range=getattr(tray_rand, "y_range", None),
    )
