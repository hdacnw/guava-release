import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from guava.config import CameraConfig


def render_camera(sim, cfg: CameraConfig) -> np.ndarray:
    """Render an RGB image from a named camera using robosuite's offscreen context."""
    ctx = sim._render_context_offscreen
    if ctx is None:
        raise RuntimeError("No offscreen render context — env must have has_offscreen_renderer=True")
    rgb = sim.render(camera_name=cfg.name, width=cfg.width, height=cfg.height, depth=False)
    return rgb[::-1].copy()
