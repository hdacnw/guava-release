"""Explicit Ubuntu collection CLI; YAML loads before command-line overrides."""
import argparse
import os
from pathlib import Path


def main():
    from guava.task_catalog import TRAINING_TASKS, ALL_TASKS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=ALL_TASKS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-turns', type=int, default=31)
    p.add_argument('--camera', default='frontview')
    p.add_argument('--grasp', choices=('pca', 'graspgen'), default='pca')
    p.add_argument('--graspgen-url', default='tcp://127.0.0.1:5556')
    p.add_argument('--frame', choices=('table',), default='table',
                   help='Collection uses the calibrated table-aligned frame (z=0 at tabletop).')
    from guava.llm.provider_settings import add_provider_arguments, resolve_provider_arguments
    add_provider_arguments(p)
    p.add_argument('--dry-run', action='store_true')
    from guava.sim.video import add_video_arguments, validate_video_fps
    add_video_arguments(p)
    args = p.parse_args()
    try:
        validate_video_fps(args.video_fps)
    except ValueError as exc:
        p.error(str(exc))
    resolve_provider_arguments(args)
    if args.trials < 1 or args.max_turns < 1:
        p.error('Trials and turn budget must be positive')
    if args.dry_run:
        print(vars(args))
        print('Training task:', args.task in TRAINING_TASKS)
        return
    if not os.environ.get(args.api_key_env):
        p.error(f'Set {args.api_key_env} in the environment')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import yaml
    from guava.config import CollectionConfig
    from guava.scripts.collect import _apply_yaml
    from guava.collection.runner import run_collection
    root = Path(__file__).resolve().parents[2]
    name = args.task
    cfg = CollectionConfig()
    _apply_yaml(cfg, yaml.safe_load((root / 'configs' / f'{name}.yaml').read_text()))
    cfg.output_dir = str(args.output.resolve())
    cfg.total_trials, cfg.num_workers, cfg.turn_budget = args.trials, 1, args.max_turns
    cfg.stop_on_reward, cfg.model_frame = False, args.frame
    cfg.sim.perception, cfg.sim.grasp, cfg.sim.motion_speed = 'sam3', args.grasp, 1.0
    cfg.sim.visualize = cfg.sim.use_mjviewer = False
    cfg.sim.randomize.seed = args.seed
    cfg.sim.camera_views = [args.camera]
    cfg.sim.camera.name = args.camera
    cfg.robot.graspgen_url = args.graspgen_url
    cfg.record_video, cfg.video_fps = args.record_video, args.video_fps
    cfg.llm.base_url, cfg.llm.model, cfg.llm.api_key_env = args.base_url, args.model, args.api_key_env
    cfg.llm.max_tokens = 8192
    run_collection(cfg)


if __name__ == '__main__':
    main()
