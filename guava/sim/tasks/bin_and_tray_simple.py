import os
os.environ.setdefault("MUJOCO_GL", "egl")

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.objects.composite.bin import Bin
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config


_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)
_ROBOCASA_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects/robocasa_objects"
)

# ----------------------------------------------------------------------------
# Receptacle sizes / fixed positions  (kept identical to bin_and_tray)
# ----------------------------------------------------------------------------
_BIN_SIZE = (0.22, 0.22, 0.08)

_TRAY_HALF_X = 0.07390   # reg_bbox size[0] (body X)
_TRAY_HALF_Y = 0.10407   # reg_bbox size[1] (body Y)
_TRAY_HALF_Z = 0.00898

_BIN_XY  = np.array([-0.18,  0.20])
_TRAY_XY = np.array([ 0.18,  0.18])

# ----------------------------------------------------------------------------
# Per-object geometry (from reg_bbox, same source as bin_and_tray)
# ----------------------------------------------------------------------------
_FORK_THICK_SCALE = 3.0
_FORK_HALF_W = 0.01377   # reg_bbox size[0]
_FORK_HALF_L = 0.08473   # reg_bbox size[1]
_FORK_HALF_T = 0.003265 * _FORK_THICK_SCALE  # reg_bbox size[2] × scale

# banana_1 (scale 0.165): reg_bbox half-extents; banana lies flat via baked-in refquat
_BANANA_HALF_X = 0.06253
_BANANA_HALF_Y = 0.06771
_BANANA_HALF_Z = 0.01606

_PLACEMENT_Z_SLACK = 0.04

# Default per-object spawn ranges
_DEFAULT_FORK_X_RANGE   = [-0.25, 0.25]
_DEFAULT_FORK_Y_RANGE   = [-0.25, -0.02]
_DEFAULT_BANANA_X_RANGE = [-0.25, 0.25]
_DEFAULT_BANANA_Y_RANGE = [-0.25, -0.02]


# ----------------------------------------------------------------------------
# Object wrappers
# ----------------------------------------------------------------------------
class ForkObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ROBOCASA_DIR / "fork_3/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=[1.0, 1.0, _FORK_THICK_SCALE],
        )
        # Replace the 32 VHACD collision hulls with a single thin box so the
        # gripper can close on the handle without hitting invisible geometry.
        inner_body = next(b for b in self.root.iter("body") if b.get("name") == f"{name}_main")
        for geom in list(inner_body.findall("geom")):
            if geom.get("group") == "0":
                inner_body.remove(geom)
        box = ET.SubElement(inner_body, "geom")
        box.set("name", f"{name}_collision_box")
        box.set("type", "box")
        box.set("size", f"{_FORK_HALF_W:.5f} {_FORK_HALF_L:.5f} {_FORK_HALF_T:.5f}")
        box.set("pos", "0 0 0")
        box.set("group", "0")
        box.set("density", "100")
        box.set("friction", "2.0 0.5 0.1")
        box.set("solimp", "0.998 0.998 0.001")
        box.set("solref", "0.001 1")
        box.set("rgba", "0.8 0.8 0.8 0.0")
        self._contact_geoms = ["collision_box"]

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_FORK_HALF_T])

    @property
    def top_offset(self):
        return np.array([0, 0, _FORK_HALF_T])

    @property
    def horizontal_radius(self):
        return float(np.linalg.norm([_FORK_HALF_W, _FORK_HALF_L]))


class BananaObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ROBOCASA_DIR / "banana_1/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_BANANA_HALF_Z])

    @property
    def top_offset(self):
        return np.array([0, 0, _BANANA_HALF_Z])

    @property
    def horizontal_radius(self):
        return float(np.linalg.norm([_BANANA_HALF_X, _BANANA_HALF_Y]))


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


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class BinAndTraySimple(ManipulationEnv):
    """Two items on a cluttered table: place the fork onto the tray and the
    banana into the bin.
    """

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 fork_x_range=None, fork_y_range=None,
                 banana_x_range=None, banana_y_range=None,
                 **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer
        self.fork_x_range   = fork_x_range   if fork_x_range   is not None else _DEFAULT_FORK_X_RANGE
        self.fork_y_range   = fork_y_range   if fork_y_range   is not None else _DEFAULT_FORK_Y_RANGE
        self.banana_x_range = banana_x_range if banana_x_range is not None else _DEFAULT_BANANA_X_RANGE
        self.banana_y_range = banana_y_range if banana_y_range is not None else _DEFAULT_BANANA_Y_RANGE

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

    def _in_bin(self, obj_pos: np.ndarray) -> bool:
        bin_pos = self.sim.data.body_xpos[self.bin_body_id]
        half_xy = np.array(_BIN_SIZE[:2]) / 2 - 0.02
        in_xy = bool(np.all(np.abs(obj_pos[:2] - bin_pos[:2]) < half_xy))
        in_z  = (bin_pos[2] - _BIN_SIZE[2] / 2) < obj_pos[2] < (bin_pos[2] + _BIN_SIZE[2] / 2 + 0.05)
        return in_xy and in_z

    def _on_tray(self, obj_pos: np.ndarray, obj_half_z: float) -> bool:
        tray_pos = self.sim.data.body_xpos[self.tray_body_id]
        half_xy = np.array([_TRAY_HALF_X, _TRAY_HALF_Y]) - 0.01
        in_xy = bool(np.all(np.abs(obj_pos[:2] - tray_pos[:2]) < half_xy))
        expected_z = tray_pos[2] + _TRAY_HALF_Z + obj_half_z
        on_z = abs(obj_pos[2] - expected_z) < _PLACEMENT_Z_SLACK
        return in_xy and on_z

    def score(self):
        """Authoritative task success, independent of reward shaping."""
        from guava.sim.scoring import result
        fork_pos   = self.sim.data.body_xpos[self.fork_body_id]
        banana_pos = self.sim.data.body_xpos[self.banana_body_id]

        fork_done   = self._on_tray(fork_pos, _FORK_HALF_T) and not self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.fork)
        banana_done = self._in_bin(banana_pos) and not self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.banana)

        return result({'fork_placed_and_released': fork_done, 'banana_placed_and_released': banana_done})

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        fork_pos   = self.sim.data.body_xpos[self.fork_body_id]
        banana_pos = self.sim.data.body_xpos[self.banana_body_id]

        fork_done   = self._on_tray(fork_pos, _FORK_HALF_T) and not self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.fork)
        banana_done = self._in_bin(banana_pos) and not self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.banana)


        if self.reward_shaping:
            r = 0.5 * (int(fork_done) + int(banana_done))

            unplaced = []
            if not fork_done:   unplaced.append(fork_pos)
            if not banana_done: unplaced.append(banana_pos)
            if unplaced:
                eef = self.sim.data.site_xpos[
                    self.robots[0].eef_site_id[self.robots[0].arms[0]]
                ]
                nearest = min(np.linalg.norm(eef - p) for p in unplaced)
                r += (1 - np.tanh(5.0 * nearest)) * 0.05
            return r

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

        self.fork   = ForkObject(name="fork")
        self.banana = BananaObject(name="banana")
        self.tray   = TrayObject(name="tray")
        self.bin    = Bin(name="bin", bin_size=_BIN_SIZE, transparent_walls=False,
                         use_texture=False, rgba=(0.82, 0.65, 0.40, 1.0), density=5000.0)

        self.placement_initializer = self._make_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.fork, self.banana, self.tray, self.bin],
        )

    def _make_placement_initializer(self):
        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for obj, x_range, y_range in [
            (self.fork,   self.fork_x_range,   self.fork_y_range),
            (self.banana, self.banana_x_range,  self.banana_y_range),
        ]:
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{obj.name}Sampler",
                    mujoco_objects=[obj],
                    x_range=x_range,
                    y_range=y_range,
                    rotation=None,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )
            )
        return sampler

    def _setup_references(self):
        super()._setup_references()
        self.fork_body_id   = self.sim.model.body_name2id(self.fork.root_body)
        self.banana_body_id = self.sim.model.body_name2id(self.banana.root_body)
        self.tray_body_id   = self.sim.model.body_name2id(self.tray.root_body)
        self.bin_body_id    = self.sim.model.body_name2id(self.bin.root_body)

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        bin_pos = np.array([
            self.table_offset[0] + _BIN_XY[0],
            self.table_offset[1] + _BIN_XY[1],
            self.table_offset[2] + _BIN_SIZE[2] / 2,
        ])
        self.sim.data.set_joint_qpos(
            self.bin.joints[0],
            np.concatenate([bin_pos, [1, 0, 0, 0]]),
        )

        tray_pos = np.array([
            self.table_offset[0] + _TRAY_XY[0],
            self.table_offset[1] + _TRAY_XY[1],
            self.table_offset[2] + _TRAY_HALF_Z + 0.001,
        ])
        self.sim.data.set_joint_qpos(
            self.tray.joints[0],
            np.concatenate([tray_pos, [1, 0, 0, 0]]),
        )

    def _check_success(self):
        return self.score()['success']


def make_bin_and_tray_simple(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)

    fork_rand   = getattr(cfg, "fork_randomize",   None) or {}
    banana_rand = getattr(cfg, "banana_randomize", None) or {}

    env = BinAndTraySimple(
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
        fork_x_range=fork_rand.get("x_range"),
        fork_y_range=fork_rand.get("y_range"),
        banana_x_range=banana_rand.get("x_range"),
        banana_y_range=banana_rand.get("y_range"),
    )
    return env
