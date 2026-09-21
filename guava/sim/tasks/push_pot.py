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
_ROBOCASA_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects/robocasa_objects"
)

# Pot054 reg_bbox (scale 0.65): size=0.12940x0.09020x0.05487, pos z=0.05342
# Mesh origin sits at pot bottom; top of bbox = 0.05342 + 0.05487 = 0.10829
_POT_TOP_Z  = 0.10829
_POT_HALF_X = 0.12940   # widest horizontal half-extent

_POT_X_RANGE = [-0.10, 0.10]
_POT_Y_RANGE = [-0.05, 0.10]

_PUSH_DIST = 0.08  # success threshold (metres)

# Direction → unit vector [x, y] in world frame (robot at -y facing +y)
_DIRECTION_VECS = {
    "left":  np.array([-1.0,  0.0]),
    "right": np.array([ 1.0,  0.0]),
    "front": np.array([ 0.0, -1.0]),  # towards camera / robot
    "back":  np.array([ 0.0,  1.0]),  # away from camera
}


class PotObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ROBOCASA_DIR / "Pot054/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, 0])  # mesh origin sits at pot bottom

    @property
    def top_offset(self):
        return np.array([0, 0, _POT_TOP_Z])

    @property
    def horizontal_radius(self):
        return _POT_HALF_X


class PushPot(ManipulationEnv):
    """Push the pot in a chosen direction."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 pot_x_range=None, pot_y_range=None,
                 push_directions=None,
                 **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))  # Match can_in_bin tabletop height.
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer
        self.pot_x_range     = pot_x_range     if pot_x_range     is not None else _POT_X_RANGE
        self.pot_y_range     = pot_y_range     if pot_y_range     is not None else _POT_Y_RANGE
        self.push_directions = list(push_directions) if push_directions else list(_DIRECTION_VECS)

        self.current_direction: str = self.push_directions[0]
        self._pot_init_pos: np.ndarray | None = None

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

    @property
    def task_instruction(self) -> str:
        return f"Push the pot to the {self.current_direction}."

    def score(self):
        """Authoritative task success, independent of reward shaping."""
        from guava.sim.scoring import result
        pot_pos = self.sim.data.body_xpos[self.pot_body_id]
        if self._pot_init_pos is None:
            return result({'initialized': False, 'pushed_8cm': False})

        from guava.sim.tasks.push_basket import camera_direction_vecs
        camera_name = self.render_camera[0] if isinstance(self.render_camera, (list, tuple)) else self.render_camera
        camera_id = self.sim.model.camera_name2id(camera_name)
        directions = camera_direction_vecs(self.sim.data.cam_xmat[camera_id])
        dir_vec = directions[self.current_direction]
        push_progress = float(np.dot(pot_pos[:2] - self._pot_init_pos[:2], dir_vec))

        return result({'pushed_8cm': push_progress >= _PUSH_DIST})

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        pot_pos = self.sim.data.body_xpos[self.pot_body_id]
        if self._pot_init_pos is None:
            return 0.0

        from guava.sim.tasks.push_basket import camera_direction_vecs
        camera_name = self.render_camera[0] if isinstance(self.render_camera, (list, tuple)) else self.render_camera
        camera_id = self.sim.model.camera_name2id(camera_name)
        directions = camera_direction_vecs(self.sim.data.cam_xmat[camera_id])
        dir_vec = directions[self.current_direction]
        push_progress = float(np.dot(pot_pos[:2] - self._pot_init_pos[:2], dir_vec))


        if self.reward_shaping:
            dist = min(
                np.linalg.norm(
                    self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - pot_pos
                )
                for arm in self.robots[0].arms
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
            r_push  = 0.75 * max(0.0, push_progress / _PUSH_DIST)
            return r_reach + r_push

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

        self.pot = PotObject(name="pot")

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects([self.pot])
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.pot],
                x_range=self.pot_x_range,
                y_range=self.pot_y_range,
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pot],
        )

    def _setup_references(self):
        super()._setup_references()
        self.pot_body_id = self.sim.model.body_name2id(self.pot.root_body)

    def _reset_internal(self):
        self.current_direction = str(np.random.choice(self.push_directions))
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        self.sim.forward()
        self._pot_init_pos = self.sim.data.body_xpos[self.pot_body_id].copy()

    def _check_success(self):
        return self.score()['success']


def make_push_pot(cfg: "SimConfig"):
    from robosuite.utils.placement_samplers import UniformRandomSampler

    controller_cfg  = load_composite_controller_config(controller=_CONTROLLER_CFG)
    pot_rand        = getattr(cfg, "pot_randomize", None) or {}
    push_directions = getattr(cfg, "push_directions", None)

    env = PushPot(
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
        pot_x_range=pot_rand.get("x_range"),
        pot_y_range=pot_rand.get("y_range"),
        push_directions=push_directions,
    )

    rotation = pot_rand.get("rotation", True)
    env.placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=[env.pot],
        x_range=pot_rand.get("x_range", _POT_X_RANGE),
        y_range=pot_rand.get("y_range", _POT_Y_RANGE),
        rotation=None if rotation else 0,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=env.table_offset,
        z_offset=0.01,
    )
    return env
