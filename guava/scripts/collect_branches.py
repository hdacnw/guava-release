#!/usr/bin/env python
"""Collect counterfactual branch trajectories from saved baseline checkpoints.

Example:
    uv run python guava/scripts/collect_branches.py \\
        --config configs/can_in_bin.yaml \\
        --baseline_dir data/collect/can_in_bin \\
        --trial_ids 1 2 3 \\
        --turns 3 5 \\
        --perturbations grasp_failure object_moved_align \\
        --pass_history true
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to task YAML config")
    p.add_argument("--baseline_dir", required=True, help="Directory containing baseline trajectories and checkpoints")
    p.add_argument("--output_dir", default=None, help="Output directory (defaults to baseline_dir)")
    p.add_argument("--trial_ids", nargs="+", type=int, default=None, help="Trial IDs to branch from (default: all)")
    p.add_argument("--turns", nargs="+", type=int, default=None, help="Turn indices to branch from (default: all)")
    p.add_argument("--perturbations", nargs="+", default=None,
                   help="Perturbation types to apply (default: all). Choices: "
                        "grasp_failure, wrong_object, drop_during_transport, "
                        "object_moved_align, wrong_approach_side, wrong_orientation")
    p.add_argument("--dryrun", action="store_true",
                   help="Print which perturbations apply to which turns without running rollouts.")
    p.add_argument("--pass_history", type=lambda x: x.lower() == "true", default=True,
                   help="Pass prior conversation history to LLM (true/false). "
                        "false = fresh VLM start from mid-task state.")
    from guava.llm.provider_settings import add_provider_arguments, resolve_provider_arguments
    add_provider_arguments(p)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--grasp', choices=('pca', 'graspgen'), default='pca')
    p.add_argument('--graspgen-url', default='tcp://127.0.0.1:5556')
    p.add_argument('--max-branches', type=int, default=1, help='Explicit cap for bounded pilots')
    p.add_argument('--seed', type=int, default=7)
    from guava.sim.video import add_video_arguments, validate_video_fps
    add_video_arguments(p)
    args = p.parse_args()
    try:
        validate_video_fps(args.video_fps)
    except ValueError as exc:
        p.error(str(exc))
    resolve_provider_arguments(args)
    if args.max_branches < 1:
        p.error('--max-branches must be positive')

    import yaml
    from guava.config import CollectionConfig
    from guava.scripts.collect import _apply_yaml
    cfg = CollectionConfig()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent.parent / args.config
    with open(config_path) as f:
        _apply_yaml(cfg, yaml.safe_load(f) or {})
    cfg.record_video, cfg.video_fps = args.record_video, args.video_fps
    cfg.llm.base_url, cfg.llm.model = args.base_url, args.model
    cfg.llm.api_key_env, cfg.llm.max_tokens = args.api_key_env, args.max_tokens
    cfg.sim.perception, cfg.sim.grasp = 'sam3', args.grasp
    cfg.sim.visualize = cfg.sim.use_mjviewer = False
    cfg.robot.graspgen_url = args.graspgen_url
    cfg.num_workers = 1
    import os
    if not args.dryrun and not os.environ.get(args.api_key_env):
        p.error(f'Set {args.api_key_env} before collection')

    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir or args.baseline_dir)
    cfg.output_dir = str(output_dir)
    ck_dir = baseline_dir / "checkpoints"
    traj_dir = baseline_dir / "trajectories"

    if not ck_dir.exists():
        print(f"[branches] No checkpoints directory found at {ck_dir}. Run baseline collection first.")
        return

    from guava.collection.checkpoint import load_from_disk
    from guava.collection.branch import make_branch_specs, run_branch
    from guava.collection.trajectory import save_branch_trajectory

    object_aliases = (getattr(cfg.sim, "object_aliases", None) or {})
    object_vocab = list(object_aliases.keys())
    all_specs = make_branch_specs(object_vocab, object_aliases)
    specs = (
        {k: v for k, v in all_specs.items() if k in args.perturbations}
        if args.perturbations
        else {k: v for k, v in all_specs.items() if k != 'fresh_start'}
    )
    if args.perturbations and set(args.perturbations) - set(all_specs):
        p.error('Unknown perturbation name')
    if not specs:
        print(f"[branches] No matching perturbation types. Available: {list(all_specs)}")
        return

    if "fresh_start" in specs and args.pass_history:
        raise SystemExit(
            "[branches] 'fresh_start' requires --pass_history false. "
            "Re-run with: --pass_history false --perturbations fresh_start"
        )

    # Discover trial IDs from checkpoint dir if not specified
    if args.trial_ids:
        trial_ids = args.trial_ids
    else:
        trial_ids = sorted({
            int(m.group(1))
            for f in ck_dir.iterdir()
            if (m := re.match(r"trial_(\d+)_turn\d+\.pkl", f.name))
        })

    if not trial_ids:
        print("[branches] No trials found.")
        return

    # Drop trials whose baseline trajectory ended in failure (success_0.json) —
    # branching from a failed run isn't useful. Applies even to ids passed via
    # --trial_ids so the user can't accidentally branch off a failure.
    filtered, skipped_failed, skipped_missing = [], [], []
    for tid in trial_ids:
        traj_files = sorted(traj_dir.glob(f"trial_{tid:04d}_*.json"))
        if len(traj_files) > 1:
            raise ValueError(f'Ambiguous parent trial {tid}: multiple JSONs; choose a unique baseline directory')
        if not traj_files:
            skipped_missing.append(tid)
            continue
        if any(f.name.endswith("_success_0.json") for f in traj_files):
            skipped_failed.append(tid)
            continue
        filtered.append(tid)
    if skipped_failed:
        print(f"[branches] Skipping {len(skipped_failed)} failed trial(s) (success_0): {skipped_failed}")
    if skipped_missing:
        print(f"[branches] Skipping {len(skipped_missing)} trial(s) with no trajectory: {skipped_missing}")
    trial_ids = filtered

    if not trial_ids:
        print("[branches] No successful trials to branch from.")
        return

    if args.dryrun:
        _dryrun(ck_dir, trial_ids, args.turns, specs)
        return

    # Add parent cameras before environment construction (observation sensors
    # are built from camera_views). Restoring an XML alone cannot add sensors.
    parent_cameras = set()
    for tid in trial_ids:
        first = next(iter(sorted(ck_dir.glob(f'trial_{tid:04d}_turn*.pkl'))), None)
        if first is not None:
            checkpoint = load_from_disk(first)
            if getattr(checkpoint, 'snapshot', None):
                parent_cameras.add(checkpoint.snapshot['camera']['name'])
    if not parent_cameras:
        raise ValueError('No version-2 parent cameras found; legacy checkpoints need verified reconstruction')
    cfg.sim.camera_views = sorted(parent_cameras)
    cfg.sim.camera.name = cfg.sim.camera_views[0]

    from guava.collection.runner import _build_components
    env, arm, perception, grasp_planner, provider = _build_components(cfg)

    from guava.tools.schema import build_tools_schema, tools_to_prompt
    from guava.tools.tools import ToolContext, make_tool_registry

    bbox_debug_dir = (
        output_dir / "debug_bbox"
        if getattr(cfg.robot, "bbox_debug", False)
        else None
    )
    ctx = ToolContext(env=env, arm=arm, perception=perception,
                      grasp_planner=grasp_planner, bbox_debug_dir=bbox_debug_dir)
    registry = make_tool_registry(ctx)
    tools_schema = build_tools_schema(registry)
    tools_text = tools_to_prompt(registry)

    total = 0

    for trial_id in trial_ids:
        branch_id = _next_branch_id(str(output_dir), trial_id)
        traj_files = sorted(traj_dir.glob(f"trial_{trial_id:04d}_*.json"))
        if not traj_files:
            print(f"[branches] No trajectory for trial {trial_id}, skipping.")
            continue
        traj_path = traj_files[0]

        ck_files = sorted(ck_dir.glob(f"trial_{trial_id:04d}_turn*.pkl"))
        if not ck_files:
            print(f"[branches] No checkpoints for trial {trial_id}, skipping.")
            continue

        # Load all, patch grasped_object from grasp/release history, then filter
        all_cks: dict[int, object] = {}
        for ck_file in ck_files:
            m = re.match(r"trial_\d+_turn(\d+)\.pkl", ck_file.name)
            if m:
                all_cks[int(m.group(1))] = load_from_disk(ck_file)
        _patch_grasped_objects(list(all_cks.values()))

        for turn, ck in sorted(all_cks.items()):
            if args.turns and turn not in args.turns:
                continue

            for spec_name, spec in specs.items():
                if total >= args.max_branches:
                    env.rob.close()
                    print(f'[branches] Reached requested cap: {total}')
                    return
                if ck.tool_name not in spec.trigger_actions:
                    continue
                if not spec.trigger_condition(ck):
                    continue

                try:
                    import numpy as np
                    from guava.collection.snapshot import digest
                    np.random.seed(int(digest([args.seed, trial_id, turn, spec_name])[:8], 16))
                    perturbation = spec.make_perturbation(ck)
                except Exception as e:
                    print(f"[branches] Skip {spec_name} trial={trial_id} turn={turn}: {e}")
                    continue

                print(f"\n[branches] trial={trial_id} turn={turn} spec={spec_name}")
                print(f"  perturbation: {perturbation.detail if perturbation is not None else 'none (fresh_start)'}")
                env.reset(seed=trial_id)

                try:
                    traj = run_branch(
                        env=env, arm=arm, ctx=ctx, registry=registry,
                        tools_text=tools_text, tools_schema=tools_schema,
                        provider=provider, cfg=cfg,
                        ck=ck, spec=spec, perturbation=perturbation,
                        branch_id=branch_id,
                        pass_history=args.pass_history,
                        traj_path=traj_path,
                        base_trial_id=trial_id,
                    )
                except ValueError as exc:
                    import json
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with (output_dir / 'held_branches.jsonl').open('a') as stream:
                        stream.write(json.dumps({'trial_id': trial_id, 'turn': turn, 'spec': spec_name,
                                                 'reason': str(exc), 'seed': args.seed}) + '\n')
                    print(f'[branches] HELD trial={trial_id} turn={turn} spec={spec_name}: {exc}')
                    if cfg.record_video:
                        branch_id += 1  # Preserve held-attempt videos without reusing their filename.
                    continue
                if traj is not None:
                    import hashlib
                    checkpoint_file = ck_dir / f'trial_{trial_id:04d}_turn{turn:03d}.pkl'
                    traj.branch_metadata['generation_seed'] = args.seed
                    traj.branch_metadata['checkpoint_sha256'] = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
                    save_branch_trajectory(traj, str(output_dir))
                    branch_id += 1
                    total += 1

    env.rob.close()
    print(f"\n[branches] Done. Created {total} branch trajectories → {output_dir}/trajectories_branch/")


def _dryrun(ck_dir: Path, trial_ids: list[int], turns_filter: list[int] | None,
            specs: dict) -> None:
    from guava.collection.checkpoint import load_from_disk

    spec_names = list(specs.keys())

    for trial_id in trial_ids:
        ck_files = sorted(ck_dir.glob(f"trial_{trial_id:04d}_turn*.pkl"))
        if not ck_files:
            print(f"\ntrial={trial_id:04d}  (no checkpoints)")
            continue

        # Load all checkpoints (needed for full grasp history), then patch
        all_cks: dict[int, object] = {}
        for ck_file in ck_files:
            m = re.match(r"trial_\d+_turn(\d+)\.pkl", ck_file.name)
            if m:
                all_cks[int(m.group(1))] = load_from_disk(ck_file)
        _patch_grasped_objects(list(all_cks.values()))

        rows: list[tuple[int, str, list[str]]] = []
        for turn, ck in sorted(all_cks.items()):
            if turns_filter and turn not in turns_filter:
                continue
            cells = [
                _dryrun_cell(spec, ck) if (
                    ck.tool_name in spec.trigger_actions and spec.trigger_condition(ck)
                ) else "—"
                for spec in specs.values()
            ]
            rows.append((turn, _fmt_action(ck), cells))

        if not rows:
            continue

        print(f"\ntrial={trial_id:04d}")
        _print_table(rows, spec_names)


def _fmt_action(ck) -> str:
    tool = ck.tool_name or "(text)"
    if tool == "move":
        tgt = ck.tool_args.get("target_position", [])
        if len(tgt) >= 3:
            dz = tgt[2] - ck.eef_pos[2]
            if dz > 0.05:
                return "move (lift)"
            if dz < -0.03:
                return "move (lower)"
    return tool


def _dryrun_cell(spec, ck) -> str:
    name = spec.name
    if name == "drop_during_transport":
        return f"✓ — holding {ck.grasped_object}"
    if name == "object_moved_align":
        obj = ck.tool_args.get("target_name", "")
        return f"✓ — displaces {obj}"
    if name == "wrong_approach_side":
        pos = ck.tool_args.get("position", "")
        return f"✓ — side={pos}"
    if name == "wrong_object":
        _OBJECT_ARG = {"grasp": "target", "align": "target_name",
                       "get_position": "object_name", "get_position_and_size": "object_name"}
        arg = _OBJECT_ARG.get(ck.tool_name, "object_name")
        return f"✓ — {ck.tool_args.get(arg, '')}"
    if name == "wrong_orientation":
        return f"✓ — angle={ck.tool_args.get('angle_deg', 0)}"
    return "✓ "


def _print_table(rows: list, spec_names: list[str]) -> None:
    headers = ["Turn", "Action"] + spec_names
    col_data = [
        [str(r[0]) for r in rows],
        [r[1] for r in rows],
        *[[r[2][i] for r in rows] for i in range(len(spec_names))],
    ]
    widths = [max(len(headers[i]), max(len(v) for v in col)) for i, col in enumerate(col_data)]

    def sep(l, m, r):
        return l + m.join("─" * (w + 2) for w in widths) + r

    def row(vals):
        return "│" + "│".join(f" {v:<{widths[i]}} " for i, v in enumerate(vals)) + "│"

    print(sep("┌", "┬", "┐"))
    print(row(headers))
    for r in rows:
        print(sep("├", "┼", "┤"))
        print(row([str(r[0]), r[1]] + r[2]))
    print(sep("└", "┴", "┘"))


def _patch_grasped_objects(cks: list) -> None:
    """Back-fill grasped_object=None checkpoints using the grasp/release history.

    Checkpoints are saved BEFORE each action executes, so grasped_object is often
    None for move/align turns that happen after a successful grasp (the proximity
    check in _find_grasped_object uses the wrist body which is ~20 cm from the
    fingertips). Scanning the grasp/release sequence recovers the correct value.
    """
    held = None
    for ck in sorted(cks, key=lambda c: c.turn_idx):
        if getattr(ck, 'version', 1) >= 2:
            continue  # v2 uses measured bilateral contact, never inferred action success.
        if ck.grasped_object is None and ck.gripper_fraction < 0.9:
            ck.grasped_object = held
        # Update held AFTER reading (checkpoint saved before action executes)
        if ck.tool_name == "grasp":
            held = ck.tool_args.get("object_name") or ck.tool_args.get("target")
        elif ck.tool_name == "release":
            held = None


def _next_branch_id(output_dir: str, trial_id: int) -> int:
    all_ids = []
    for subdir in ("trajectories_branch", "images_branch", "videos_branch"):
        d = Path(output_dir) / subdir
        if not d.exists():
            continue
        all_ids.extend(
            int(m.group(1))
            for f in d.iterdir()
            if (m := re.match(rf"trial_{trial_id:04d}_branch_(\d+)[_.]", f.name))
        )
    return max(all_ids) + 1 if all_ids else 1


if __name__ == "__main__":
    main()
