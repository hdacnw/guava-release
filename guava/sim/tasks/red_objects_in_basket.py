import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject, CanObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config


_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)
_CUSTOM_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/robosuite/robosuite/models/assets/objects/robocasa_objects"
)

# ── Object dimensions (from reg_bbox in each model.xml) ──────────────────────
# Basket045: reg_bbox pos_z=0.054196, size=(0.19974, 0.19974, 0.055817)
_BASKET_TOP_Z  = 0.11001   # 0.054196 + 0.055817
_BASKET_HALF_X = 0.19974
_BASKET_INT_X  = 0.16    # interior half-extents for "in basket" check
_BASKET_INT_Y  = 0.16

# tomato_1 (scale 0.06): reg_bbox half ≈ (0.032, 0.032, 0.029)
_TOMATO_HALF_Z = 0.029
_TOMATO_RADIUS = 0.032

# CanObject (robosuite built-in): bottom_site z=-0.06, top_site z=0.04, radius≈0.035
_CAN_HALF_Z = 0.05    # (0.06 + 0.04) / 2
_CAN_RADIUS = 0.035

# potato_0 (scale 0.09): reg_bbox half ≈ (0.037, 0.045, 0.036)
_POTATO_HALF_Z = 0.036
_POTATO_RADIUS = 0.045

# pear_1 (scale 0.12): lies on its side; z half ≈ 0.029, horizontal max ≈ 0.060
_PEAR_HALF_Z   = 0.029
_PEAR_RADIUS   = 0.060

_BASKET_X_RANGE = [-0.08, 0.08]
_BASKET_Y_RANGE = [ 0.10, 0.20]
_SMALL_X_RANGE  = [-0.25, 0.25]
_SMALL_Y_RANGE  = [-0.25, -0.05]


# ── Object classes ────────────────────────────────────────────────────────────

class BasketObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_CUSTOM_DIR / "Basket045/model.xml"),
            name=name, joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self): return np.array([0, 0, 0])

    @property
    def top_offset(self): return np.array([0, 0, _BASKET_TOP_Z])

    @property
    def horizontal_radius(self): return _BASKET_HALF_X


class TomatoObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_CUSTOM_DIR / "tomato_1/model.xml"),
            name=name, joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self): return np.array([0, 0, -_TOMATO_HALF_Z])

    @property
    def top_offset(self): return np.array([0, 0, _TOMATO_HALF_Z])

    @property
    def horizontal_radius(self): return _TOMATO_RADIUS



class PotatoObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_CUSTOM_DIR / "potato_0/model.xml"),
            name=name, joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self): return np.array([0, 0, -_POTATO_HALF_Z])

    @property
    def top_offset(self): return np.array([0, 0, _POTATO_HALF_Z])

    @property
    def horizontal_radius(self): return _POTATO_RADIUS


class PearObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_CUSTOM_DIR / "pear_1/model.xml"),
            name=name, joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=False,
        )

    @property
    def bottom_offset(self): return np.array([0, 0, -_PEAR_HALF_Z])

    @property
    def top_offset(self): return np.array([0, 0, _PEAR_HALF_Z])

    @property
    def horizontal_radius(self): return _PEAR_RADIUS


# ── Environment ───────────────────────────────────────────────────────────────

class RedObjectsInBasket(ManipulationEnv):
    """Place the tomato and can into the basket; potato and pear are distractors."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True, placement_initializer=None,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 basket_x_range=None, basket_y_range=None,
                 small_x_range=None, small_y_range=None,
                 **kwargs):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.placement_initializer = placement_initializer
        self.basket_x_range = basket_x_range if basket_x_range is not None else _BASKET_X_RANGE
        self.basket_y_range = basket_y_range if basket_y_range is not None else _BASKET_Y_RANGE
        self.small_x_range  = small_x_range  if small_x_range  is not None else _SMALL_X_RANGE
        self.small_y_range  = small_y_range  if small_y_range  is not None else _SMALL_Y_RANGE

        super().__init__(
            robots=robots, controller_configs=controller_configs,
            has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs, camera_names=camera_names,
            camera_heights=camera_heights, camera_widths=camera_widths,
            camera_depths=camera_depths, horizon=horizon, renderer=renderer,
            render_camera=render_camera, **kwargs,
        )

    def _in_basket(self, obj_body_id: int) -> bool:
        basket_pos = self.sim.data.body_xpos[self.basket_body_id]
        obj_pos    = self.sim.data.body_xpos[obj_body_id]
        dx = abs(obj_pos[0] - basket_pos[0])
        dy = abs(obj_pos[1] - basket_pos[1])
        dz = obj_pos[2] - basket_pos[2]
        return dx < _BASKET_INT_X and dy < _BASKET_INT_Y and -0.02 < dz < 0.19

    def score(self):
        from guava.sim.scoring import snapshot
        return score_state(snapshot(self, ["basket","tomato","can","potato","pear"]))

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        t_done = self._in_basket(self.tomato_body_id) and not self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.tomato)
        c_done = self._in_basket(self.can_body_id) and not self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.can)

        # Partial credit only for objects already released inside the basket.
        r = 0.5 * (int(t_done) + int(c_done))

        if self.reward_shaping:
            for body_id, obj_geoms, done in [
                (self.tomato_body_id, self.tomato, t_done),
                (self.can_body_id, self.can, c_done),
            ]:
                if done:
                    continue
                obj_pos  = self.sim.data.body_xpos[body_id]
                eef_dist = min(
                    np.linalg.norm(
                        self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - obj_pos
                    )
                    for arm in self.robots[0].arms
                )
                r += (1 - np.tanh(10.0 * eef_dist)) * 0.10

        return r

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

        self.basket = BasketObject(name="basket")
        self.tomato = TomatoObject(name="tomato")
        self.can    = CanObject(name="can")
        self.potato = PotatoObject(name="potato")
        self.pear   = PearObject(name="pear")

        # SequentialCompositeSampler places the 4 small objects with collision avoidance.
        # Basket is placed manually in _reset_internal.
        small_sampler = SequentialCompositeSampler(name="SmallObjectSampler")
        for obj in [self.tomato, self.can, self.potato, self.pear]:
            small_sampler.append_sampler(UniformRandomSampler(
                name=f"{obj.name}Sampler",
                mujoco_objects=[obj],
                x_range=self.small_x_range,
                y_range=self.small_y_range,
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))
        self.placement_initializer = small_sampler

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.basket, self.tomato, self.can, self.potato, self.pear],
        )

    def _setup_references(self):
        super()._setup_references()
        self.basket_body_id = self.sim.model.body_name2id(self.basket.root_body)
        self.tomato_body_id = self.sim.model.body_name2id(self.tomato.root_body)
        self.can_body_id    = self.sim.model.body_name2id(self.can.root_body)
        self.potato_body_id = self.sim.model.body_name2id(self.potato.root_body)
        self.pear_body_id   = self.sim.model.body_name2id(self.pear.root_body)

    def _reset_internal(self):
        super()._reset_internal()

        # Basket at a random back-of-table position
        bx = self.table_offset[0] + np.random.uniform(*self.basket_x_range)
        by = self.table_offset[1] + np.random.uniform(*self.basket_y_range)
        self.sim.data.set_joint_qpos(
            self.basket.joints[0],
            np.concatenate([[bx, by, self.table_offset[2] + 0.01], [1, 0, 0, 0]]),
        )

        # Small objects placed sequentially with collision avoidance
        placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
            )

    def _check_success(self):
        return self.score()['success']


def make_red_objects_in_basket(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    basket_rand    = getattr(cfg, "basket_randomize", None) or {}
    small_rand     = getattr(cfg, "small_randomize",  None) or {}

    return RedObjectsInBasket(
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
        basket_x_range=basket_rand.get("x_range"),
        basket_y_range=basket_rand.get("y_range"),
        small_x_range=small_rand.get("x_range"),
        small_y_range=small_rand.get("y_range"),
    )


def score_state(state):
    """Authoritative task success criteria, independently testable from physics."""
    from guava.sim.scoring import predicates, result
    p = predicates(state)
    touch, released, supported, upright, inside = p.touch, p.released, p.supported, p.upright, p.inside
    o = state['objects']
    metrics = {}

    checks = {'basket_supported': supported('basket'), 'basket_upright': upright('basket')}
    for n in ('tomato', 'can'):
        checks[n + '_inside'] = inside(n, 'basket', .16, -.02, .14)
        checks[n + '_supported_in_basket'] = touch(n, 'basket') or (
            touch(n, 'can' if n == 'tomato' else 'tomato') and
            touch('can' if n == 'tomato' else 'tomato', 'basket'))
        checks[n + '_released'] = released(n)
    for n in ('potato', 'pear'):
        checks[n + '_not_in_basket'] = not inside(n, 'basket', .16, -.02, .14)
        checks[n + '_on_table'] = supported(n)
    return result(checks, metrics)
