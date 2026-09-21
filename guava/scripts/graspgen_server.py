"""Guava GraspGen server launcher.

Finds the GraspGen .venv and re-execs the upstream server script inside it,
so callers only need guava's Python (no GraspGen deps required in this process).

Usage:
    uv run python -m guava.scripts.graspgen_server \\
        --port 5556 \\
        --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_franka_panda.yml \\
        [--host 127.0.0.1] \\
        [--venv /path/to/GraspGen/.venv]

The venv defaults to third_party/GraspGen/.venv relative to the guava repo root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_default_venv() -> Path:
    guava_root = Path(__file__).resolve().parents[2]
    return guava_root / "third_party" / "GraspGen" / ".venv"


def _find_server_script() -> Path:
    # Use guava's extended server (collision-aware) instead of the upstream script.
    return Path(__file__).resolve().parent / "_graspgen_extended_server.py"


def main(
    port: int = 5556,
    host: str = "127.0.0.1",
    gripper_config: str = "",
    venv: str | None = None,
) -> None:
    if not gripper_config:
        print("Error: --gripper_config is required.", file=sys.stderr)
        print("  Example: --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_franka_panda.yml",
              file=sys.stderr)
        sys.exit(1)

    venv_path = Path(venv) if venv else _find_default_venv()
    python = venv_path / "bin" / "python"
    if not python.exists():
        print(f"Error: GraspGen Python not found at {python}", file=sys.stderr)
        print("  Set --venv to the GraspGen .venv directory, or create one at:", file=sys.stderr)
        print(f"  {_find_default_venv()}", file=sys.stderr)
        sys.exit(1)

    server_script = _find_server_script()
    if not server_script.exists():
        print(f"Error: GraspGen server script not found at {server_script}", file=sys.stderr)
        print("  Ensure the third_party/GraspGen submodule is initialised:", file=sys.stderr)
        print("  git submodule update --init third_party/GraspGen", file=sys.stderr)
        sys.exit(1)

    # Replace the current process with the GraspGen server running inside its venv.
    os.execv(str(python), [
        str(python), str(server_script),
        "--gripper_config", gripper_config,
        "--host", host,
        "--port", str(port),
    ])


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
