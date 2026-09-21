import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)
_OBJECTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects/robocasa_objects"
)

# cup_3 at scale 0.105 — same as cube_under_cup
_CUP_HALF_Z  = 0.05250
_CUP_RADIUS  = 0.04330

_CUBE_HALF = 0.015   # small red cube, 3 cm per side

# 180° rotation around world X axis: flips the cup so mouth faces down
_CUP_FLIPPED_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz

_LIFT_HEIGHT    = 0.08   # cube must be lifted ≥ 8 cm above table surface for success
_CUP_CLEAR_DIST = 0.08   # covering cup must move ≥ 8 cm from cube start XY

# Three non-overlapping X zones (gap > 2 * _CUP_RADIUS ≈ 0.087 m between zones)
_ZONE_RANGES = [
    ([-0.25, -0.13], [-0.10, 0.10]),   # left
    ([-0.04,  0.04], [-0.10, 0.10]),   # centre
    ([ 0.13,  0.25], [-0.10, 0.10]),   # right
]


class CupObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_OBJECTS_DIR / "cup_3/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_CUP_HALF_Z])

    @property
    def top_offset(self):
        return np.array([0, 0, _CUP_HALF_Z])

    @property
    def horizontal_radius(self):
        return _CUP_RADIUS


class ShellGame(ManipulationEnv):
    """Three identical upside-down cups on the table; exactly one hides a red cube.
    The robot must check the cups, reveal the cube, and lift it.
    """

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
        self._cube_init_pos: np.ndarray | None = None
        self._covering_cup_body_id: int | None = None

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
        grasping_cube = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        cube_lifted   = cube_pos[2] > self.table_offset[2] + _LIFT_HEIGHT

        return result({'grasped': grasping_cube, 'lifted': cube_lifted})

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        grasping_cube = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)


        if self.reward_shaping:
            # r_clear: covering cup moved away from the cube's starting position
            r_clear = 0.0
            if self._cube_init_pos is not None and self._covering_cup_body_id is not None:
                cup_pos = self.sim.data.body_xpos[self._covering_cup_body_id]
                cup_dist = np.linalg.norm(cup_pos[:2] - self._cube_init_pos[:2])
                r_clear = min(1.0, cup_dist / _CUP_CLEAR_DIST) * 0.25

            eef_pos = self.sim.data.site_xpos[
                self.robots[0].eef_site_id[self.robots[0].arms[0]]
            ]
            r_reach = (1 - np.tanh(10.0 * np.linalg.norm(eef_pos - cube_pos))) * 0.25
            r_grasp = 0.25 if grasping_cube else 0.0
            r_lift  = 0.25 if cube_pos[2] > self.table_offset[2] + 0.04 else 0.0

            return r_clear + r_reach + r_grasp + r_lift

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

        self.cup_a = CupObject(name="cup_a")
        self.cup_b = CupObject(name="cup_b")
        self.cup_c = CupObject(name="cup_c")

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[_CUBE_HALF, _CUBE_HALF, _CUBE_HALF],
            size_max=[_CUBE_HALF, _CUBE_HALF, _CUBE_HALF],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        # Each cup gets its own non-overlapping zone sampler
        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for cup, (x_range, y_range) in zip(
            [self.cup_a, self.cup_b, self.cup_c], _ZONE_RANGES
        ):
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{cup.name}Sampler",
                    mujoco_objects=[cup],
                    x_range=x_range,
                    y_range=y_range,
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )
            )
        self.placement_initializer = sampler

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cup_a, self.cup_b, self.cup_c, self.cube],
        )

    def _setup_references(self):
        super()._setup_references()
        self.cup_a_body_id = self.sim.model.body_name2id(self.cup_a.root_body)
        self.cup_b_body_id = self.sim.model.body_name2id(self.cup_b.root_body)
        self.cup_c_body_id = self.sim.model.body_name2id(self.cup_c.root_body)
        self.cube_body_id  = self.sim.model.body_name2id(self.cube.root_body)

    def _reset_internal(self):
        super()._reset_internal()

        # Sample cup positions from the three zone samplers
        object_placements = self.placement_initializer.sample()
        cup_positions = {}
        for cup in (self.cup_a, self.cup_b, self.cup_c):
            pos, _, _ = object_placements[cup.name]
            cup_positions[cup.name] = pos[:2]

        # Randomly choose which cup hides the cube
        cups = [self.cup_a, self.cup_b, self.cup_c]
        cup_body_ids = [self.cup_a_body_id, self.cup_b_body_id, self.cup_c_body_id]
        cover_idx = np.random.randint(3)
        covering_cup = cups[cover_idx]
        self._covering_cup_body_id = cup_body_ids[cover_idx]
        cover_xy = cup_positions[covering_cup.name]

        # Place all three cups flipped upside-down at table level
        for cup, name in zip(cups, [c.name for c in cups]):
            xy = cup_positions[name]
            cup_pos = np.array([xy[0], xy[1], self.table_offset[2] + _CUP_HALF_Z + 0.001])
            self.sim.data.set_joint_qpos(
                cup.joints[0],
                np.concatenate([cup_pos, _CUP_FLIPPED_QUAT]),
            )

        # Place the cube under the chosen cup
        cube_pos = np.array([cover_xy[0], cover_xy[1],
                             self.table_offset[2] + _CUBE_HALF + 0.001])
        self.sim.data.set_joint_qpos(
            self.cube.joints[0],
            np.concatenate([cube_pos, [1, 0, 0, 0]]),
        )

        self.sim.forward()
        for _ in range(50):
            self.sim.step()

        self._cube_init_pos = self.sim.data.body_xpos[self.cube_body_id].copy()

    def _check_success(self):
        return self.score()['success']


def make_shell_game(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    env = ShellGame(
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
    return env
