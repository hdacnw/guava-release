import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np
import robosuite.environments.manipulation.stack as _stack
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from guava.sim.randomization import make_placement_sampler

_CONTROLLER_CFG = str(
    Path(__file__).resolve().parent.parent / "controllers" / "panda_joint_ctrl.json"
)


class StackReverse(_stack.Stack):
    """Stack the green cube (cubeB) on top of the red cube (cubeA)."""

    def score(self):
        """Green cube lifted, released, and in contact with the red cube."""
        from guava.sim.scoring import result
        height = self.sim.data.body_xpos[self.cubeB_body_id][2]
        return result({
            'lifted': height > self.table_offset[2] + 0.04,
            'released': not self._check_grasp(
                gripper=self.robots[0].gripper, object_geoms=self.cubeB),
            'contact': self.check_contact(self.cubeB, self.cubeA),
        })

    def _check_success(self):
        return self.score()['success']

    def reward(self, action=None):
        # Keep robosuite's optional reward scaling; success never depends on it.
        reach, lift, stack = self.staged_rewards()
        value = max(reach, lift, stack) if self.reward_shaping else stack
        return value if self.reward_scale is None else value * self.reward_scale / 2.0

    def staged_rewards(self):
        # Same logic as Stack.staged_rewards() but cubeB is picked up and placed on cubeA.
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]

        dist = min(
            np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - cubeB_pos)
            for arm in self.robots[0].arms
        )
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        grasping_cubeB = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeB)
        if grasping_cubeB:
            r_reach += 0.25

        cubeB_height = cubeB_pos[2]
        table_height = self.table_offset[2]
        cubeB_lifted = cubeB_height > table_height + 0.04
        r_lift = 1.0 if cubeB_lifted else 0.0

        if cubeB_lifted:
            horiz_dist = np.linalg.norm(cubeB_pos[:2] - cubeA_pos[:2])
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        r_stack = 2.0 if self.score()['success'] else 0.0

        return r_reach, r_lift, r_stack


def make_cube_stack_reverse(cfg: "SimConfig"):
    controller_cfg = load_composite_controller_config(controller=_CONTROLLER_CFG)
    env = StackReverse(
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
        objects=[env.cubeA, env.cubeB],
        cfg=cfg.randomize,
        reference_pos=env.table_offset,
    )
    return env
