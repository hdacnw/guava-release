from __future__ import annotations
import multiprocessing as mp
import re
from pathlib import Path
from guava.collection.trajectory import save_trajectory


def _next_trial_id(output_dir: str) -> int:
    """Return one past the highest trial ID already saved in output_dir."""
    ids = [
        int(m.group(1))
        for subdir in ("trajectories", "videos")
        for f in (Path(output_dir) / subdir).glob('trial_*')
        if (m := re.match(r"trial_(\d+)[_.]", f.name))
    ]
    return max(ids) + 1 if ids else 1


def run_collection(cfg) -> None:
    """Run total_trials episodes, using num_workers parallel processes."""
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    start_id = _next_trial_id(cfg.output_dir)
    trial_ids = range(start_id, start_id + cfg.total_trials)
    print(f"[collection] Starting from trial_id={start_id} ({cfg.total_trials} trials).")

    if cfg.num_workers == 1 or cfg.sim.visualize:
        # Single process — simpler to debug
        env, arm, perception, grasp_planner, provider = _build_components(cfg)
        for trial_id in trial_ids:
            from guava.collection.trial import run_trial
            traj = run_trial(env, arm, perception, provider, cfg, trial_id, grasp_planner=grasp_planner)
            save_trajectory(traj, cfg.output_dir)
    else:
        # Start external servers in the main process before spawning workers.
        # Without this, every worker races to start the same server simultaneously,
        # causing CUDA OOM when multiple model loads hit the same GPU.
        _warmup_servers(cfg)
        with mp.Pool(cfg.num_workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            for traj in pool.imap_unordered(_worker_run, trial_ids):
                save_trajectory(traj, cfg.output_dir)


def _warmup_servers(cfg) -> None:
    """Start SAM3 server in the main process before workers are spawned."""
    from urllib.parse import urlparse
    from guava.perception import _tcp_ready

    # GraspGen (must be started manually — see guava/scripts/graspgen_server.py)
    if getattr(cfg.sim, "grasp", "pca") == "graspgen":
        from guava.perception import get_graspgen_client
        get_graspgen_client(cfg.robot, cfg.sim.grasp)

    # SAM3
    perception_mode = getattr(cfg.sim, "perception", "auto")
    sam3_url = getattr(cfg.robot, "sam3_url", None)
    if perception_mode in ("sam3", "auto") and sam3_url:
        parsed = urlparse(sam3_url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 8114
        if not _tcp_ready(host, port):
            from guava.perception import _start_sam3
            _start_sam3(sam3_url)


_worker_state: dict = {}


def _worker_init(cfg) -> None:
    _worker_state["components"] = _build_components(cfg)
    _worker_state["cfg"] = cfg


def _worker_run(trial_id: int) -> object:
    env, arm, perception, grasp_planner, provider = _worker_state["components"]
    cfg = _worker_state["cfg"]
    from guava.collection.trial import run_trial
    return run_trial(env, arm, perception, provider, cfg, trial_id, grasp_planner=grasp_planner)


def _build_components(cfg):
    from guava.sim.tasks import make_env
    from guava.robot.arm import RobotArm
    from guava.robot.ik import get_ik_solver
    from guava.perception import get_position_solver
    from guava.llm.providers import get_provider

    env = make_env(cfg.sim)
    ik  = get_ik_solver(cfg.robot, env)
    arm = RobotArm(env, ik, cfg.robot)
    perception = get_position_solver(cfg.robot, cfg.sim, env, output_dir=cfg.output_dir)
    from guava.perception import get_graspgen_client
    grasp_planner = get_graspgen_client(cfg.robot, cfg.sim.grasp)
    provider      = get_provider(cfg.llm)
    return env, arm, perception, grasp_planner, provider
