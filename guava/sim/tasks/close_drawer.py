import os
os.environ.setdefault("MUJOCO_GL", "egl")

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import ManipulationTask
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)

# Cabinet placed against the -Y edge of the table, oriented so its drawers
# face (and open toward) +Y. The drawer joint axis is local +Y; with the 180°
# Z rotation in _reset_internal, local +Y maps to world -Y, so negative qpos
# (drawer "out") slides toward world +Y.
_CABINET_XY    = np.array([0.0, -0.35])
_INIT_OPEN_POS = -0.14   # drawer starts near fully open (range: -0.16 to 0.01)
_CLOSE_THRESHOLD = -0.06  # drawer is "closed enough" when qpos > this


class WoodenCabinetObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/articulated_objects/wooden_cabinet.xml"),
            name=name,
            joints=[dict(type="free", damping="0")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    def _get_object_subtree(self):
        obj = super()._get_object_subtree()
        _g = dict(
            type="box",
            solimp="0.998 0.998 0.001", solref="0.001 1",
            density="100", friction="0.95 0.3 0.1", group="0",
        )

        # ── Cabinet base: disable solid mesh, replace with hollow outer shell ──────
        # Measured AABB: x=[-0.106,0.114], y=[-0.089,0.146], z=[-0.022,0.234]
        base = obj.find("./body[@name='base']")
        if base is not None:
            for geom in base:
                if geom.tag == "geom" and geom.get("group") == "0" and geom.get("type") != "box":
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                    break
            for name, pos, size in [
                ("shell_bottom", "0.0038 0.0287 -0.010",  "0.11 0.1174 0.012"),
                ("shell_top",    "0.0038 0.0287 0.222",   "0.11 0.1174 0.012"),
                ("shell_left",   "-0.094 0.0287 0.1058",  "0.012 0.1174 0.1281"),
                ("shell_right",  "0.102 0.0287 0.1058",   "0.012 0.1174 0.1281"),
                ("shell_back",   "0.0038 0.134 0.1058",   "0.11 0.012 0.1281"),
            ]:
                base.append(ET.Element("geom", attrib={**_g, "name": name, "pos": pos, "size": size}))

        # ── Top drawer: disable solid mesh; add thin front-face slab ────────────
        # The handle box (g5) stays active. A front-face slab gives the gripper
        # something physical to contact when grasping the drawer face directly.
        top = obj.find("./body[@name='base']/body[@name='cabinet_top']")
        if top is not None:
            for geom in top:
                if geom.tag == "geom" and geom.get("group") == "0" and geom.get("type") != "box":
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                    break
            # drawer face-plate slab — collision enabled, red tint for visual verification
            top.append(ET.Element("geom", attrib={
                **_g, "name": "drawer_front",
                "pos": "0.007 -0.073 0.182", "size": "0.080 0.005 0.036"
            }))

        return obj


class CloseDrawer(ManipulationEnv):
    """Top drawer starts open. Goal: push it closed."""

    def __init__(self, robots, controller_configs=None, has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 reward_shaping=True,
                 camera_names="frontview", camera_heights=512, camera_widths=512,
                 camera_depths=False, horizon=500000, renderer="mujoco",
                 render_camera="frontview",
                 cab_x_range=None, cab_y_range=None, **kwargs):
        self.table_full_size = (1.0, 1.0, 0.05)
        self.table_friction  = (1.0, 5e-3, 1e-4)
        self.table_offset    = np.array((0, 0, 0.932))
        self.reward_shaping  = reward_shaping
        self.cab_x_range     = cab_x_range
        self.cab_y_range     = cab_y_range

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
        drawer_pos = self.sim.data.qpos[self._top_drawer_qadr]

        return result({'drawer_closed': drawer_pos > _CLOSE_THRESHOLD})

    def reward(self, action=None):
        from guava.sim.scoring import shaped_reward
        return shaped_reward(self, action)

    def _shaped_reward(self, action=None):
        drawer_pos = self.sim.data.qpos[self._top_drawer_qadr]


        if self.reward_shaping:
            site_pos = self.sim.data.site_xpos[self.top_region_id]
            dist = min(
                np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - site_pos)
                for arm in self.robots[0].arms
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
            # close_frac: 0 when fully open, 1 when at threshold
            close_frac = max(0.0, (drawer_pos - _INIT_OPEN_POS) / (_CLOSE_THRESHOLD - _INIT_OPEN_POS))
            return r_reach + close_frac * 0.75

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

        self.cabinet = WoodenCabinetObject(name="cabinet")

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cabinet],
        )

    def _setup_references(self):
        super()._setup_references()
        self.top_region_id = self.sim.model.site_name2id("cabinet_top_region")
        jid = self.sim.model.joint_name2id("cabinet_top_level")
        self._top_drawer_qadr = self.sim.model.jnt_qposadr[jid]

        # Make the cabinet body effectively immovable so gripper contact
        # doesn't knock it off the table. Drawers (cabinet_top/middle/bottom)
        # have their own slide joints and keep their original ~3 kg mass, so
        # they remain easy to push closed.
        root_id = self.sim.model.body_name2id(self.cabinet.root_body)
        self.sim.model.body_mass[root_id] = 100.0
        self.sim.model.body_inertia[root_id] = [100.0, 100.0, 100.0]

        # Crank friction on drawer-body collision geoms so the gripper doesn't
        # slip off the handle during interaction.
        for jname in ("cabinet_top_level", "cabinet_middle_level", "cabinet_bottom_level"):
            drawer_jid = self.sim.model.joint_name2id(jname)
            drawer_bid = self.sim.model.jnt_bodyid[drawer_jid]
            for gid in range(self.sim.model.ngeom):
                if (self.sim.model.geom_bodyid[gid] == drawer_bid
                        and self.sim.model.geom_group[gid] == 0):  # collision geoms only
                    self.sim.model.geom_friction[gid] = [5.0, 1.0, 0.1]

    def _reset_internal(self):
        super()._reset_internal()

        rng = getattr(self, 'rng', None)
        if rng is None:
            rng = np.random.default_rng()
        x_off = rng.uniform(*self.cab_x_range) if self.cab_x_range else 0.0
        y_off = rng.uniform(*self.cab_y_range) if self.cab_y_range else 0.0
        cab_pos = np.array([
            self.table_offset[0] + _CABINET_XY[0] + x_off,
            self.table_offset[1] + _CABINET_XY[1] + y_off,
            self.table_offset[2],
        ])
        self.sim.data.set_joint_qpos(
            f"{self.cabinet.name}_joint0",
            # wxyz = 180° about Z → cabinet's front (drawer side) faces +Y.
            np.concatenate([cab_pos, [0.0, 0.0, 0.0, 1.0]]),
        )

        # Middle and bottom drawers closed; top drawer open
        for jname in ["cabinet_middle_level", "cabinet_bottom_level"]:
            jid  = self.sim.model.joint_name2id(jname)
            qadr = self.sim.model.jnt_qposadr[jid]
            self.sim.data.qpos[qadr] = 0.0

        self.sim.data.qpos[self._top_drawer_qadr] = _INIT_OPEN_POS

    def _check_success(self):
        return self.score()['success']


def make_close_drawer(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    rand = getattr(cfg, "randomize", None)
    return CloseDrawer(
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
        cab_x_range=getattr(rand, "x_range", None),
        cab_y_range=getattr(rand, "y_range", None),
    )
