from __future__ import annotations
import atexit
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from guava.config import RobotConfig, SimConfig

_sam3_proc: subprocess.Popen | None = None
_graspgen_proc: subprocess.Popen | None = None


def _tcp_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


def _start_sam3(url: str, timeout: float = 120.0) -> None:
    """Spawn the SAM3 server as a background subprocess and wait for it to be ready."""
    global _sam3_proc
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8114

    print(f"[perception] Starting SAM3 server on {host}:{port}...")
    _sam3_proc = subprocess.Popen(
        [sys.executable, "-m", "guava.scripts.sam3_server",
         "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    atexit.register(_stop_sam3)

    deadline = time.monotonic() + timeout
    interval = 1.0
    while time.monotonic() < deadline:
        if _sam3_proc.poll() is not None:
            try:
                _, err = _sam3_proc.communicate(timeout=5.0)
                err_text = err.decode("utf-8", errors="replace").strip()
            except Exception:
                err_text = "(could not read server output)"
            raise RuntimeError(
                f"[perception] SAM3 server process exited early (code {_sam3_proc.returncode}).\n"
                f"Server output:\n{err_text}"
            )
        if _tcp_ready(host, port):
            print(f"[perception] SAM3 server ready at {url}")
            return
        time.sleep(min(interval, deadline - time.monotonic()))
        interval = min(interval * 1.5, 5.0)

    _sam3_proc.terminate()
    raise RuntimeError(
        f"[perception] SAM3 server did not become ready within {timeout:.0f}s"
    )


def _stop_sam3() -> None:
    global _sam3_proc
    if _sam3_proc is not None and _sam3_proc.poll() is None:
        print("[perception] Stopping SAM3 server...")
        _sam3_proc.terminate()
        try:
            _sam3_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _sam3_proc.kill()
        _sam3_proc = None


def _graspgen_zmq_ready(host: str, port: int, timeout_ms: int = 2000) -> bool:
    """Return True if the GraspGen ZMQ server responds to a health check."""
    try:
        import zmq
        import msgpack
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{host}:{port}")
        sock.send(msgpack.packb({"action": "health"}, use_bin_type=True))
        raw = sock.recv()
        resp = msgpack.unpackb(raw, raw=False)
        sock.close()
        ctx.term()
        return resp.get("status") == "ok"
    except Exception:
        return False


def _start_graspgen(host: str, port: int, gripper_config: str, venv: str, timeout: float = 120.0) -> None:
    """Spawn the GraspGen ZMQ server using its own .venv and wait for it to be ready."""
    global _graspgen_proc
    from pathlib import Path as _Path

    python = _Path(venv) / "bin" / "python"
    server_script = _Path(__file__).resolve().parents[2] / "third_party" / "GraspGen" / "client-server" / "graspgen_server.py"

    print(f"[perception] Starting GraspGen server on {host}:{port}...")
    _graspgen_proc = subprocess.Popen(
        [str(python), str(server_script),
         "--gripper_config", gripper_config,
         "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    atexit.register(_stop_graspgen)

    deadline = time.monotonic() + timeout
    interval = 2.0
    while time.monotonic() < deadline:
        if _graspgen_proc.poll() is not None:
            try:
                _, err = _graspgen_proc.communicate(timeout=5.0)
                err_text = err.decode("utf-8", errors="replace").strip()
            except Exception:
                err_text = "(see /tmp/anygrasp_server.log)"
            raise RuntimeError(
                f"[perception] GraspGen server exited early "
                f"(code {_graspgen_proc.returncode}).\n"
                f"Server output:\n{err_text}"
            )
        if _graspgen_zmq_ready(host, port):
            print(f"[perception] GraspGen server ready at tcp://{host}:{port}")
            return
        time.sleep(min(interval, deadline - time.monotonic()))
        interval = min(interval * 1.5, 10.0)

    _graspgen_proc.terminate()
    raise RuntimeError(
        f"[perception] GraspGen server did not become ready within {timeout:.0f}s"
    )


def _stop_graspgen() -> None:
    global _graspgen_proc
    if _graspgen_proc is not None and _graspgen_proc.poll() is None:
        print("[perception] Stopping GraspGen server...")
        _graspgen_proc.terminate()
        try:
            _graspgen_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _graspgen_proc.kill()
        _graspgen_proc = None


def _resolve_graspgen_venv(robot_cfg) -> str:
    """Return the GraspGen .venv path, falling back to third_party/GraspGen/.venv."""
    from pathlib import Path as _Path
    configured = getattr(robot_cfg, "graspgen_venv", None)
    if configured:
        return configured
    return str(_Path(__file__).resolve().parents[2] / "third_party" / "GraspGen" / ".venv")


def get_graspgen_client(robot_cfg, grasp_mode: str = "pca"):
    """Return a GraspGenClient if grasp_mode is 'graspgen', auto-starting the server if needed."""
    if grasp_mode != "graspgen":
        return None

    url = getattr(robot_cfg, "graspgen_url", None)
    if not url:
        raise RuntimeError("[perception] sim.grasp='graspgen' but robot.graspgen_url is not set")

    addr = url.replace("tcp://", "")
    parts = addr.rsplit(":", 1)
    host = parts[0] if len(parts) == 2 else "127.0.0.1"
    port = int(parts[1]) if len(parts) == 2 else 5556

    from guava.perception.graspgen import GraspGenClient
    client = GraspGenClient(host=host, port=port)
    if client.is_available():
        print(f"[perception] Using GraspGen at {url}")
        return client

    gripper_config = getattr(robot_cfg, "graspgen_config", None)
    if not gripper_config:
        raise RuntimeError(
            "[perception] GraspGen server not reachable and robot.graspgen_config is not set.\n"
            "  Either start the server manually:\n"
            "    uv run python -m guava.scripts.graspgen_server "
            "--gripper_config /path/to/graspgen_franka_panda.yml --port 5556\n"
            "  Or set robot.graspgen_config in your YAML config."
        )

    venv = _resolve_graspgen_venv(robot_cfg)
    _start_graspgen(host, port, gripper_config, venv)
    if client.is_available():
        return client
    raise RuntimeError(f"[perception] GraspGen started but still unreachable at {url}")


def get_position_solver(robot_cfg: RobotConfig, sim_cfg: SimConfig, env, output_dir: str | None = None):
    """Return the configured perception solver.

    sim_cfg.perception controls the backend:
      'gt'   — ground-truth positions from sim state (no service required)
      'sam3' — SAM3 segmentation service; auto-starts if url is set and server is not running
      'auto' — SAM3 if reachable, else ground-truth
    """
    mode = sim_cfg.perception

    if mode in ("sam3", "auto"):
        if robot_cfg.sam3_url:
            from guava.perception.sam3 import SAM3Solver
            debug_dir = (
                str(__import__("pathlib").Path(output_dir) / "debug_sam3")
                if robot_cfg.sam3_debug and output_dir
                else None
            )
            solver = SAM3Solver(robot_cfg.sam3_url, debug_dir=debug_dir)
            if solver.is_available():
                print(f"[perception] Using SAM3 at {robot_cfg.sam3_url}")
                return solver

            if mode == "sam3":
                # Auto-start and retry
                _start_sam3(robot_cfg.sam3_url)
                if solver.is_available():
                    return solver
                raise RuntimeError(
                    f"[perception] SAM3 started but still unreachable at {robot_cfg.sam3_url}"
                )
        elif mode == "sam3":
            raise RuntimeError(
                "[perception] SAM3 requested but robot.sam3_url is not set"
            )
        print("[perception] SAM3 unavailable — falling back to ground-truth positions")

    aliases = sim_cfg.object_aliases or {}
    from guava.perception.gt import GTSolver
    if aliases:
        print(f"[perception] GT solver with aliases: {aliases}")
    else:
        print("[perception] GT solver (no aliases)")
    return GTSolver(env, aliases=aliases)
