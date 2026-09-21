"""Randomization utilities for guava tasks.

Two concerns live here:
1. make_placement_sampler() — task-agnostic factory for UniformRandomSampler.
   Every task calls this so placement config (x_range, y_range, rotation) is
   controlled centrally via RandomizationConfig rather than per-task hardcoding.

2. VisualRandomizer — applies color and lighting randomization directly to
   MuJoCo model arrays after each reset (no h5py / DomainRandomizationWrapper).

Uses no external wrappers (no h5py dependency). Saves original values on first
call and restores them before each new randomization so drift does not accumulate
across trials.
"""
from __future__ import annotations
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Placement sampler factory (task-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def make_placement_sampler(objects: list, cfg, reference_pos, z_offset: float = 0.01):
    """Return a UniformRandomSampler configured from RandomizationConfig.

    Args:
        objects:       List of MuJoCo object instances to place (task-specific).
        cfg:           The RandomizationConfig from SimConfig.
        reference_pos: Origin for placement, typically env.table_offset.
        z_offset:      Height above reference_pos to place objects.

    Every task should call this instead of constructing a sampler directly,
    so that x_range, y_range, and rotation are controlled by the config.
    """
    from robosuite.utils.placement_samplers import UniformRandomSampler

    # rotation=None → uniform random z-rotation; rotation=0 → fixed upright
    rotation = None if cfg.rotation else 0

    return UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=objects,
        x_range=cfg.x_range,
        y_range=cfg.y_range,
        rotation=rotation,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=reference_pos,
        z_offset=z_offset,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Visual randomizer
# ──────────────────────────────────────────────────────────────────────────────

class VisualRandomizer:
    """Randomizes non-robot geom colors and light properties per trial."""

    def __init__(self, rob_env):
        self._rob = rob_env
        # Saved originals — populated on first call to randomize()
        self._orig_geom_rgba: np.ndarray | None = None
        self._orig_light_pos: np.ndarray | None = None
        self._orig_light_diffuse: np.ndarray | None = None
        self._orig_light_specular: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def randomize(self, rng: np.random.Generator, cfg) -> None:
        """Apply randomization according to RandomizationConfig."""
        model = self._rob.sim.model

        if self._orig_geom_rgba is None:
            self._save_defaults(model)
        else:
            self._restore_defaults(model)

        if cfg.colors:
            self._randomize_colors(model, rng)
        if cfg.lighting:
            self._randomize_lighting(model, rng)

        # Propagate model changes into the simulation state
        self._rob.sim.forward()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_defaults(self, model) -> None:
        self._orig_geom_rgba = model.geom_rgba.copy()
        if model.nlight > 0:
            self._orig_light_pos = model.light_pos.copy()
            self._orig_light_diffuse = model.light_diffuse.copy()
            self._orig_light_specular = model.light_specular.copy()

    def _restore_defaults(self, model) -> None:
        model.geom_rgba[:] = self._orig_geom_rgba
        if self._orig_light_pos is not None:
            model.light_pos[:] = self._orig_light_pos
            model.light_diffuse[:] = self._orig_light_diffuse
            model.light_specular[:] = self._orig_light_specular

    def _randomize_colors(self, model, rng: np.random.Generator) -> None:
        """Shift the hue/brightness of non-robot, non-invisible geoms."""
        skip_keywords = ("robot", "gripper", "hand", "mount", "base")
        for gid in range(model.ngeom):
            body_id = model.geom_bodyid[gid]
            try:
                body_name = model.body_id2name(body_id).lower()
            except Exception:
                continue
            if any(kw in body_name for kw in skip_keywords):
                continue
            rgba = model.geom_rgba[gid]
            if rgba[3] < 0.05:   # skip invisible geoms
                continue
            noise = rng.uniform(-0.25, 0.25, 3)
            model.geom_rgba[gid, :3] = np.clip(rgba[:3] + noise, 0.05, 0.95)

    def _randomize_lighting(self, model, rng: np.random.Generator) -> None:
        """Perturb light positions and diffuse intensities."""
        for lid in range(model.nlight):
            # Horizontal position jitter (keep lights above scene)
            model.light_pos[lid, 0] += rng.uniform(-0.4, 0.4)
            model.light_pos[lid, 1] += rng.uniform(-0.4, 0.4)
            model.light_pos[lid, 2] = float(np.clip(
                self._orig_light_pos[lid, 2] + rng.uniform(-0.3, 0.3), 0.3, 3.5
            ))
            # Diffuse intensity: keep bright enough to be usable
            intensity = rng.uniform(0.55, 1.0, 3)
            model.light_diffuse[lid] = np.clip(intensity, 0, 1)
            model.light_specular[lid] = np.clip(intensity * 0.5, 0, 1)
