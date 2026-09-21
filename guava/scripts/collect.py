"""Main data-collection entry point.

Usage:
    python -m guava.scripts.collect
    python -m guava.scripts.collect --config configs/can_in_bin.yaml
    python -m guava.scripts.collect \\
        --cfg.llm.provider google --cfg.llm.model gemini-2.5-pro-preview \\
        --cfg.total-trials 200 --cfg.num-workers 4
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import yaml
import tyro
from guava.config import CollectionConfig
from guava.collection.runner import run_collection

# Guava root = the directory that contains configs/, guava/, README.md, etc.
# Resolved from this file: guava/guava/scripts/collect.py → up 3 levels.
_GUAVA_ROOT = Path(__file__).resolve().parent.parent.parent


def main(
    config: str = "configs/can_in_bin.yaml",
    cfg: CollectionConfig = CollectionConfig(),
) -> None:
    """Run VLM data collection.

    Args:
        config: Path to the YAML config file (relative to the guava root, or absolute).
                Loaded first; --cfg.* flags override individual fields.
        cfg:    Collection config — all fields overridable via --cfg.<field> flags.
    """
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = _GUAVA_ROOT / config
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    _apply_yaml(cfg, raw)

    if cfg.sim.visualize:
        cfg.num_workers = 1

    run_collection(cfg)


def _apply_yaml(cfg: CollectionConfig, raw: dict) -> None:
    for key, val in raw.items():
        if hasattr(cfg, key) and not isinstance(val, dict):
            setattr(cfg, key, val)
    if "llm" in raw:
        for k, v in raw["llm"].items():
            if hasattr(cfg.llm, k):
                setattr(cfg.llm, k, v)
    if "sim" in raw:
        for k, v in raw["sim"].items():
            if k == "camera" and isinstance(v, dict):
                for ck, cv in v.items():
                    if hasattr(cfg.sim.camera, ck):
                        setattr(cfg.sim.camera, ck, cv)
            elif k == "randomize" and isinstance(v, dict):
                for rk, rv in v.items():
                    if hasattr(cfg.sim.randomize, rk):
                        setattr(cfg.sim.randomize, rk, rv)
            elif hasattr(cfg.sim, k):
                setattr(cfg.sim, k, v)
    if "robot" in raw:
        for k, v in raw["robot"].items():
            if hasattr(cfg.robot, k):
                setattr(cfg.robot, k, v)
    # Pass through any other top-level dict keys (e.g. bin_randomize, bowl_randomize,
    # tomato_randomize) as dynamic attributes on cfg.sim so task factories can read them.
    _KNOWN = {"llm", "sim", "robot"}
    for key, val in raw.items():
        if key not in _KNOWN and isinstance(val, (dict, list)):
            setattr(cfg.sim, key, val)


if __name__ == "__main__":
    tyro.cli(main)
