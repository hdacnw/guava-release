import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
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
_CUSTOM_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects/custom_objects"
)
_OBJECTS_DIR_STD = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects"
)

# Geometry from each model.xml reg_bbox (half-extents at the model's scale)
_PLATE_HALF_Z  = 0.00555  # plate_1 at scale 0.185 — flat
_PLATE_RADIUS  = 0.09092
# bowl.xml: body origin at the bowl BOTTOM (sites: bottom@0, top@0.044, radius@0.055)
_BOWL_HEIGHT   = 0.044
_BOWL_RADIUS   = 0.055
# custom spoon: reg_bbox size 0.0706 x 0.0178 x 0.0135 (lies flat, long axis = X)
_SPOON_HALF_X  = 0.0706   # long axis
_SPOON_HALF_Y  = 0.0178   # short axis
_SPOON_HALF_Z  = 0.0135

# Success thresholds
_BOWL_ON_PLATE_XY     = 0.052  # bowl XY must be within this of plate centre
_BOWL_PLATE_Z_TOL     = 0.02   # allow bowl bottom up to 2 cm below nominal plate top
_SPOON_NEAR_PLATE_MIN = _PLATE_RADIUS + _SPOON_HALF_Y + 0.01  # spoon centre clears plate edge
_SPOON_NEAR_PLATE_MAX = 0.28                                   # but stays within reach

_DEFAULT_PLATE_X_RANGE = [-0.15, 0.15]
_DEFAULT_PLATE_Y_RANGE = [-0.15, 0.15]
_DEFAULT_BOWL_X_RANGE  = [-0.20, 0.20]
_DEFAULT_BOWL_Y_RANGE  = [-0.20, 0.20]
_DEFAULT_SPOON_X_RANGE = [-0.25, 0.25]
_DEFAULT_SPOON_Y_RANGE = [-0.20, 0.20]


class PlateObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ROBOCASA_DIR / "plate_1/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_PLATE_HALF_Z])

    @property
    def top_offset(self):
        return np.array([0, 0, _PLATE_HALF_Z])

    @property
    def horizontal_radius(self):
        return _PLATE_RADIUS


class BowlObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_OBJECTS_DIR_STD / "bowl.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, 0.0])  # body origin sits at the bowl bottom

    @property
    def top_offset(self):
        return np.array([0, 0, _BOWL_HEIGHT])

    @property
    def horizontal_radius(self):
        return _BOWL_RADIUS


class SpoonObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_CUSTOM_DIR / "spoon/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.1", armature="0.005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_SPOON_HALF_Z])

    @property
    def top_offset(self):
        return np.array([0, 0, _SPOON_HALF_Z])

    @property
    def horizontal_radius(self):
        return _SPOON_HALF_X


class SetTable(ManipulationEnv):
    """Bowl, plate, and spoon scattered on the table. Goal: place the bowl on the
    plate and the spoon next to (but off) the plate.
    """

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 plate_x_range=None, plate_y_range=None,
                 bowl_x_range=None, bowl_y_range=None,
                 spoon_x_range=None, spoon_y_range=None,
                 **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer
        self.plate_x_range = plate_x_range if plate_x_range is not None else _DEFAULT_PLATE_X_RANGE
        self.plate_y_range = plate_y_range if plate_y_range is not None else _DEFAULT_PLATE_Y_RANGE
        self.bowl_x_range  = bowl_x_range  if bowl_x_range  is not None else _DEFAULT_BOWL_X_RANGE
        self.bowl_y_range  = bowl_y_range  if bowl_y_range  is not None else _DEFAULT_BOWL_Y_RANGE
        self.spoon_x_range = spoon_x_range if spoon_x_range is not None else _DEFAULT_SPOON_X_RANGE
        self.spoon_y_range = spoon_y_range if spoon_y_range is not None else _DEFAULT_SPOON_Y_RANGE

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
        plate_pos = self.sim.data.body_xpos[self.plate_body_id]
        bowl_pos  = self.sim.data.body_xpos[self.bowl_body_id]
        spoon_pos = self.sim.data.body_xpos[self.spoon_body_id]

        bowl_xy_dist  = np.linalg.norm(bowl_pos[:2]  - plate_pos[:2])
        spoon_xy_dist = np.linalg.norm(spoon_pos[:2] - plate_pos[:2])

        plate_top = plate_pos[2] + _PLATE_HALF_Z
        # bowl_pos[2] is the body origin which IS the bowl bottom for bowl.xml
        bowl_on_plate_z = bowl_pos[2] > (plate_top - _BOWL_PLATE_Z_TOL)

        bowl_on_plate  = bowl_xy_dist < _BOWL_ON_PLATE_XY and bowl_on_plate_z
        spoon_near     = _SPOON_NEAR_PLATE_MIN < spoon_xy_dist < _SPOON_NEAR_PLATE_MAX

        grasping_any = (
            self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.bowl)
            or self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.spoon)
        )

        return result({'bowl_on_plate': bowl_on_plate, 'spoon_near': spoon_near, 'released': not grasping_any})

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        plate_pos = self.sim.data.body_xpos[self.plate_body_id]
        bowl_pos  = self.sim.data.body_xpos[self.bowl_body_id]
        spoon_pos = self.sim.data.body_xpos[self.spoon_body_id]

        bowl_xy_dist  = np.linalg.norm(bowl_pos[:2]  - plate_pos[:2])
        spoon_xy_dist = np.linalg.norm(spoon_pos[:2] - plate_pos[:2])

        grasping_any = (
            self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.bowl)
            or self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.spoon)
        )


        if self.reward_shaping:
            r_bowl = (1 - np.tanh(5.0 * bowl_xy_dist)) * 0.4
            spoon_target = (_SPOON_NEAR_PLATE_MIN + _SPOON_NEAR_PLATE_MAX) / 2.0
            r_spoon  = (1 - np.tanh(5.0 * abs(spoon_xy_dist - spoon_target))) * 0.4
            r_release = 0.2 if not grasping_any else 0.0
            return r_bowl + r_spoon + r_release

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

        self.plate = PlateObject(name="plate")
        self.bowl  = BowlObject(name="bowl")
        self.spoon = SpoonObject(name="spoon")

        self.placement_initializer = self._make_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.plate, self.bowl, self.spoon],
        )

    def _make_placement_initializer(self):
        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for obj, x_range, y_range in [
            (self.plate, self.plate_x_range, self.plate_y_range),
            (self.bowl,  self.bowl_x_range,  self.bowl_y_range),
            (self.spoon, self.spoon_x_range, self.spoon_y_range),
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
        self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)
        self.bowl_body_id  = self.sim.model.body_name2id(self.bowl.root_body)
        self.spoon_body_id = self.sim.model.body_name2id(self.spoon.root_body)

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


def make_set_table(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)

    plate_rand = getattr(cfg, "plate_randomize", None) or {}
    bowl_rand  = getattr(cfg, "bowl_randomize",  None) or {}
    spoon_rand = getattr(cfg, "spoon_randomize", None) or {}

    env = SetTable(
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
        plate_x_range=plate_rand.get("x_range"),
        plate_y_range=plate_rand.get("y_range"),
        bowl_x_range=bowl_rand.get("x_range"),
        bowl_y_range=bowl_rand.get("y_range"),
        spoon_x_range=spoon_rand.get("x_range"),
        spoon_y_range=spoon_rand.get("y_range"),
    )
    return env
