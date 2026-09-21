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
_OBJECTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects"
)
_ROBOCASA_DIR = _OBJECTS_DIR / "robocasa_objects"

# apple.xml (mesh scale=2.0): sites → bottom z=-0.08, horiz x=0.08
_APPLE_HALF_Z = 0.08
_APPLE_RADIUS = 0.08

# Minimum gap (m) between apple and juice along the camera left-right axis
_SORT_MIN_GAP = 0.08

_DEFAULT_APPLE_X_RANGE = [-0.22, 0.22]
_DEFAULT_APPLE_Y_RANGE = [-0.20, -0.05]
_DEFAULT_JUICE_X_RANGE = [-0.22, 0.22]
_DEFAULT_JUICE_Y_RANGE = [-0.20, -0.05]


def _camera_right_xy(cam_quat_wxyz: np.ndarray) -> np.ndarray:
    """Return the unit vector pointing camera-right projected to world XY."""
    w, x, y, z = cam_quat_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    v = R[:2, 0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


class AppleObject(MujocoXMLObject):
    # apple.xml has a baked-in <freejoint> that robosuite doesn't track.
    # We strip it in _get_object_subtree and let robosuite add its own.
    def __init__(self, name):
        super().__init__(
            str(_OBJECTS_DIR / "apple.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    def _get_object_subtree(self):
        obj = super()._get_object_subtree()
        for child in list(obj):
            if child.tag == "freejoint":
                obj.remove(child)
        return obj

    @property
    def bottom_offset(self):
        return np.array([0, 0, -_APPLE_HALF_Z])

    @property
    def top_offset(self):
        return np.array([0, 0, _APPLE_HALF_Z])

    @property
    def horizontal_radius(self):
        return _APPLE_RADIUS


class JuiceObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ROBOCASA_DIR / "Juice001/model.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        # Use the custom robosuite asset as-is: no evaluation-side resizing.
        self._asset_bounds()

    def _asset_bounds(self):
        bbox = self.get_obj().find(f'.//geom[@name="{self.naming_prefix}reg_bbox"]')
        if bbox is None or bbox.get('type') != 'box':
            raise ValueError('Juice asset requires a box reg_bbox for placement bounds')
        center = np.fromstring(bbox.get('pos', '0 0 0'), sep=' ')
        half = np.fromstring(bbox.get('size', ''), sep=' ')
        if center.shape != (3,) or half.shape != (3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(half)) or np.any(half <= 0):
            raise ValueError('Invalid Juice reg_bbox placement bounds')
        return center, half

    @property
    def bottom_offset(self):
        center, half = self._asset_bounds()
        return np.array([0, 0, center[2] - half[2]])

    @property
    def top_offset(self):
        center, half = self._asset_bounds()
        return np.array([0, 0, center[2] + half[2]])

    @property
    def horizontal_radius(self):
        center, half = self._asset_bounds()
        return float(np.max(np.abs(center[:2]) + half[:2]))


class AppleJuiceOrder(ManipulationEnv):
    """Arrange the apple (smaller, left) and juice (larger, right) by size in camera view."""

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
        return score_state(snapshot(self, ["apple","juice"], geometry=True))

    def reward(self, action=None):
        return float(self.score()['success'])

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

        self.placement_initializer = self._make_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.apple, self.juice],
        )

    def _make_placement_initializer(self):
        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for obj, x_range, y_range in [
            (self.apple, self.apple_x_range, self.apple_y_range),
            (self.juice, self.juice_x_range, self.juice_y_range),
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


def make_apple_juice_order(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    apple_rand = getattr(cfg, "apple_randomize", None) or {}
    juice_rand = getattr(cfg, "juice_randomize", None) or {}

    return AppleJuiceOrder(
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


def score_state(state, *, reverse=False):
    """Camera-projected geometry ordering and release; no support requirement."""
    from guava.sim.scoring import predicates, result
    released = predicates(state).released
    o = state['objects']
    metrics = {}

    right = np.array(state['camera_right'], dtype=float)[:2]
    right /= np.linalg.norm(right)
    centers = {n: np.array(o[n]['geometry_center']) for n in ('apple', 'juice')}
    gap = float(np.dot(centers['juice'][:2] - centers['apple'][:2], right))
    metrics['camera_right_gap_m'] = gap
    def projected_x(n):
        delta = centers[n] - np.array(state['camera_position'])
        depth = -float(np.dot(delta, state['camera_front']))
        return float(np.dot(delta, state['camera_right'])) / depth if depth > 0 else None
    apple_x, juice_x = projected_x('apple'), projected_x('juice')
    image_gap = juice_x - apple_x if apple_x is not None and juice_x is not None else None
    metrics['projected_horizontal_gap'] = image_gap
    metrics['ordering_reference'] = 'camera_projection_of_geometry_centers'
    metrics['apple_projected_x'] = apple_x
    metrics['juice_projected_x'] = juice_x
    if reverse:
        image_gap = -image_gap if image_gap is not None else None
        metrics['projected_horizontal_gap'] = image_gap
        metrics['camera_right_gap_m'] = -gap
        metrics['expected_left_to_right'] = ['juice', 'apple']
    # About 3 pixels at 512px / 45-degree FOV: reject indistinguishable ties,
    # without adding a task requirement for 8cm physical separation.
    # User-selected apple/juice criterion: ordering and release only.
    # Table support remains required for the other task scorers.
    checks = {'ordering': image_gap is not None and image_gap > .005,
              'released': all(released(n) for n in o)}

    return result(checks, metrics)
