import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

from guava.sim.randomization import make_placement_sampler
from guava.sim.objects import _BBoxXMLObject

_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)

# Robocasa carrot_6 ships with a 0.14 scale baked into the XML, giving a full
# size of ~3.25 x 3.02 x 14 cm in body frame (body-Z is the length axis after
# the mesh refquat).  Standing up the carrot would be 14 cm tall and tip over,
# so we apply a lay-flat quat below.
_CARROT_SCALE = 1.0

# World-frame half-extents *after* _CARROT_FLAT_QUAT is applied (matches the
# convention used by _BBoxXMLObject for placement math):
#   world X = body -Z (length)       = 0.07
#   world Y = body  Y (cross-section)= 0.01509
#   world Z = body  X (cross-section)= 0.01625  <- vertical, sits on the table
_CARROT_HALF_SIZE = np.array([0.07, 0.01508939, 0.01625231])

# -90° rotation about world Y: body X -> world Z, body Z -> world -X.
# Lays the carrot horizontal with its long axis along world X.
_CARROT_FLAT_QUAT = np.array([np.cos(np.pi / 4), 0.0, -np.sin(np.pi / 4), 0.0])

# Height above the table top (in metres) the carrot must reach to count as
# "picked up".
_LIFT_HEIGHT = 0.08


class PickUpCarrot(ManipulationEnv):
    """Pick up the carrot from the table."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview", **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer

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
        carrot_pos = self.sim.data.body_xpos[self.carrot_body_id]
        grasping   = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.carrot)
        lifted_z   = carrot_pos[2] - self.table_offset[2]

        return result({'grasped': grasping, 'lifted': lifted_z > _LIFT_HEIGHT})

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        carrot_pos = self.sim.data.body_xpos[self.carrot_body_id]
        grasping   = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.carrot)
        lifted_z   = carrot_pos[2] - self.table_offset[2]


        if self.reward_shaping:
            dist = min(
                np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - carrot_pos)
                for arm in self.robots[0].arms
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
            if grasping:
                r_reach += 0.25
            r_lift = 0.25 * min(max(lifted_z, 0.0) / _LIFT_HEIGHT, 1.0)
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

        carrot_xml = xml_path_completion("objects/robocasa_objects/carrot_6/model.xml")
        self.carrot = _BBoxXMLObject(
            fname=carrot_xml,
            name="carrot",
            half_size=_CARROT_HALF_SIZE,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
            scale=_CARROT_SCALE,
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects([self.carrot])
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.carrot],
                x_range=[-0.10, 0.10],
                y_range=[-0.15, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.carrot],
        )

    def _setup_references(self):
        super()._setup_references()
        self.carrot_body_id = self.sim.model.body_name2id(self.carrot.root_body)

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                quat = _CARROT_FLAT_QUAT if obj is self.carrot else np.array(obj_quat)
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), quat]),
                )

    def _check_success(self):
        return self.score()['success']


def make_pick_up_carrot(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    env = PickUpCarrot(
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
    )
    env.placement_initializer = make_placement_sampler(
        objects=[env.carrot],
        cfg=cfg.randomize,
        reference_pos=env.table_offset,
    )
    return env
