from __future__ import annotations
import json
import hashlib
from copy import deepcopy
from guava.collection.snapshot import restore_snapshot, digest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from guava.collection.checkpoint import Checkpoint, reconstruct_messages
from guava.collection.perturbation import Perturbation, SetObjectPose, SetGripperPose, SubstituteArgument
from guava.collection.reasoning import checkpoint_reasoning


_STATE_MUTATING = {"move", "grasp", "align", "close_gripper", "release", "rotate", "home_pose"}

_ALIGN_DISPLACE_RANGE  = (0.07, 0.14)   # metres — object displacement after align
_GRASP_DISPLACE_RANGE  = (0.03, 0.06)   # metres — gripper/object displacement for grasp/drop
_ORIENTATION_OFFSET_RANGE = (30.0, 90.0)  # degrees — magnitude of wrong-orientation offset
_APPROACH_SIDES = ("left", "right", "front", "back")


@dataclass
class BranchSpec:
    name: str
    trigger_actions: list[str]
    trigger_condition: Callable[[Checkpoint], bool]
    make_perturbation: Callable[[Checkpoint], Perturbation]
    hook_timing: str  # "pre" | "post" | "pre_close"


def make_branch_specs(object_vocab: list[str], object_aliases: dict[str, str] | None = None) -> dict[str, BranchSpec]:
    """Return predefined BranchSpecs for all variation types, parameterised by the task's object vocab."""

    def _rand_xy(mag_range: tuple) -> list:
        angle = np.random.uniform(0, 2 * np.pi)
        mag = np.random.uniform(*mag_range)
        return [mag * np.cos(angle), mag * np.sin(angle), 0.0]

    def _grasp_failure(ck: Checkpoint) -> Perturbation:
        d = _rand_xy(_GRASP_DISPLACE_RANGE)
        return SetGripperPose(pos_delta=d)

    _OBJECT_ARG = {"grasp": "target", "align": "target_name",
                   "get_position": "object_name", "get_position_and_size": "object_name"}

    def _wrong_object(ck: Checkpoint) -> Perturbation:
        # Preserve the key used by a canonical or historical checkpoint.
        arg = "target_name" if "target_name" in ck.tool_args else _OBJECT_ARG.get(ck.tool_name, "object_name")
        current = ck.tool_args.get(arg, "")
        aliases = object_aliases or {}
        others = [v for v in object_vocab if aliases.get(v, v) != aliases.get(current, current)]
        if not others:
            raise ValueError('No distinct physical object available')
        return SubstituteArgument(arg_name=arg, new_value=np.random.choice(others))

    def _drop_transport(ck: Checkpoint) -> Perturbation:
        if ck.grasped_object is None:
            raise ValueError("No grasped object in checkpoint")
        d = _rand_xy(_GRASP_DISPLACE_RANGE)
        return SetObjectPose(object_name=ck.grasped_object, pos_delta=d)

    def _object_moved_align(ck: Checkpoint) -> Perturbation:
        target_body = ck.tool_args.get("target_name", "")
        aliases = object_aliases or {}
        if target_body in aliases:
            target_body = aliases[target_body]
        elif target_body:
            # LLM may use a single token of a multi-word alias key
            # (e.g. "loaf" when yaml has "brown loaf: bread"). Resolve only
            # when the token match is unambiguous.
            matches = {v for k, v in aliases.items() if target_body in k.split()}
            if len(matches) == 1:
                target_body = matches.pop()
        return SetObjectPose(object_name=target_body, pos_delta=_rand_xy(_ALIGN_DISPLACE_RANGE))

    def _wrong_approach(ck: Checkpoint) -> Perturbation:
        current = ck.tool_args.get("position", "")
        options = [s for s in _APPROACH_SIDES if s != current]
        return SubstituteArgument(arg_name="position", new_value=np.random.choice(options))

    def _wrong_orientation(ck: Checkpoint) -> Perturbation:
        mag = np.random.uniform(*_ORIENTATION_OFFSET_RANGE)
        offset = mag * np.random.choice([-1.0, 1.0])
        current = float(ck.tool_args.get("angle_deg", 0.0))
        return SubstituteArgument(arg_name="angle_deg", new_value=current + offset)

    return {
        "fresh_start": BranchSpec(
            name="fresh_start",
            trigger_actions=list(_STATE_MUTATING),
            trigger_condition=lambda ck: ck.turn_idx > 1,
            make_perturbation=lambda ck: None,
            hook_timing="post",
        ),
        "grasp_failure": BranchSpec(
            name="grasp_failure",
            trigger_actions=["grasp"],
            trigger_condition=lambda ck: True,
            make_perturbation=_grasp_failure,
            hook_timing="pre_close",
        ),
        "wrong_object": BranchSpec(
            name="wrong_object",
            trigger_actions=["grasp", "align", "get_position", "get_position_and_size"],
            trigger_condition=lambda ck: len(object_vocab) > 1,
            make_perturbation=_wrong_object,
            hook_timing="pre",
        ),
        "drop_during_transport": BranchSpec(
            name="drop_during_transport",
            trigger_actions=["move", "align", "rotate", "home_pose"],
            trigger_condition=lambda ck: ck.gripper_fraction < 0.9 and ck.grasped_object is not None,
            make_perturbation=_drop_transport,
            hook_timing="pre",
        ),
        "object_moved_align": BranchSpec(
            name="object_moved_align",
            trigger_actions=["align"],
            trigger_condition=lambda ck: True,
            make_perturbation=_object_moved_align,
            hook_timing="post",
        ),
        "wrong_approach_side": BranchSpec(
            name="wrong_approach_side",
            trigger_actions=["align"],
            trigger_condition=lambda ck: ck.tool_args.get("position", "top") in ("left", "right", "front", "back"),
            make_perturbation=_wrong_approach,
            hook_timing="pre",
        ),
        "wrong_orientation": BranchSpec(
            name="wrong_orientation",
            trigger_actions=["rotate"],
            trigger_condition=lambda ck: True,
            make_perturbation=_wrong_orientation,
            hook_timing="pre",
        ),
    }




def validate_parent(ck, traj_path, task_key):
    if getattr(ck, 'version', 1) != 2 or not ck.snapshot or not ck.messages:
        raise ValueError('Legacy checkpoint lacks camera, controller and exact history state. '
                         'Use a verified version-2 parent; legacy warm replay is not collection-safe.')
    if traj_path is None:
        raise ValueError('Parent JSON required for provenance validation')
    record = json.loads(Path(traj_path).read_text())
    metadata = record['metadata']
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    runtime = metadata.get('runtime', {})
    if runtime.get('collection_id') != ck.collection_id or runtime.get('task_key') != task_key:
        raise ValueError('Checkpoint belongs to a different parent run or task')
    if metadata.get('trial_id') != ck.trial_id:
        raise ValueError('Parent trial ID mismatch')
    if not metadata.get('quality', {}).get('eligible'):
        raise ValueError('Parent did not pass collection quality checks')
    messages = reconstruct_messages(Path(traj_path), ck.message_count)
    if digest(messages) != ck.prefix_digest or digest(ck.messages) != ck.prefix_digest:
        raise ValueError('Parent history/image content changed or checkpoint index does not match')
    index = ck.message_count - 1
    if index >= len(record['conversations']):
        raise ValueError('Checkpoint action absent from parent')
    entry = record['conversations'][index]
    if entry['from'] != 'gpt' or entry['value'] != ck.assistant_content or entry.get('loss') is False:
        raise ValueError('Parent action or reasoning mismatch')
    if not checkpoint_reasoning(record, index, ck.tool_name, ck.tool_args)[1]:
        raise ValueError('Parent decision lacks compatible reasoning')
    binding = runtime.get('checkpoints', {}).get(str(ck.turn_idx), {})
    if binding != {'message_count': ck.message_count, 'prefix_digest': ck.prefix_digest,
                   'action_digest': digest(ck.assistant_content)}:
        raise ValueError('Parent checkpoint binding mismatch')
    return record, metadata


def run_branch(*, env, arm, ctx, registry, tools_text, tools_schema, provider, cfg,
               ck, spec, perturbation, branch_id, pass_history, traj_path, base_trial_id):
    from guava.collection.rollout import continue_rollout, execute_observe, append_observation
    from guava.collection.trial import _init_messages
    from guava.collection.trajectory import Step
    from guava.tools.tools import make_tool_registry, _refresh, _encode, _gripper_state_text
    from guava.tools.coordinate_frame import model_registry
    parent, parent_md = validate_parent(ck, traj_path, cfg.sim.task)
    for key, actual in (('ik', cfg.robot.ik_backend), ('perception', cfg.sim.perception),
                        ('grasp', cfg.sim.grasp)):
        if ck.runtime_metadata.get(key) != actual:
            raise ValueError(f'Parent execution backend differs: {key}')
    if base_trial_id != ck.trial_id:
        raise ValueError('Requested parent ID differs from checkpoint')
    if spec is None or ck.tool_name not in spec.trigger_actions or not spec.trigger_condition(ck):
        raise ValueError('Perturbation is ineligible at this checkpoint')
    if spec.name != 'fresh_start' and perturbation is None:
        raise ValueError('A perturbation is required')
    if spec.name == 'fresh_start' and pass_history:
        raise ValueError('fresh_start requires pass_history=false')
    check = restore_snapshot(ck.snapshot, env, arm, ctx)
    from guava.sim.video import record_episode
    video_path = Path(cfg.output_dir) / 'videos_branch' / f'trial_{base_trial_id:04d}_branch_{branch_id:04d}.mp4'
    with record_episode(env, video_path, getattr(cfg, 'record_video', False), getattr(cfg, 'video_fps', 20)):
        # Parent observations stay immutable; the requested speed applies from the
        # branch intervention onward and is recorded separately from parent history.
        env.cfg.motion_speed = float(cfg.sim.motion_speed)
        registry = model_registry(ctx, make_tool_registry(ctx))
        from guava.tools.schema import tools_to_prompt, build_tools_schema
        tools_text, tools_schema = tools_to_prompt(registry), build_tools_schema(registry)
        runtime = deepcopy(ck.runtime_metadata)
        runtime.pop('video', None)  # Parent video is not this branch's recording.
        if getattr(cfg, 'record_video', False):
            runtime['video'] = {'path': str(video_path.resolve()), 'fps': cfg.video_fps}
        runtime['model'] = cfg.llm.model
        runtime['motion_speed'] = float(env.cfg.motion_speed)
        runtime['checkpoints'] = {}
        messages = deepcopy(ck.messages)
        steps = []
        effect = None
        intended = deepcopy(ck.tool_args)
        executed = deepcopy(intended)
        ctx.grasp_displace_to = ctx.grasp_displace_delta = None
        ctx.grasp_perturbation_applied = False
        try:
            if spec.name != 'fresh_start':
                if spec.hook_timing == 'pre':
                    if isinstance(perturbation, SetObjectPose):
                        perturbation.apply_to_sim(env)
                        if spec.name == 'drop_during_transport':
                            env._settle(30)
                            ctx.was_holding = False
                        _refresh(ctx)
                    elif isinstance(perturbation, SubstituteArgument):
                        executed = perturbation.apply_to_args(executed)
                    else:
                        raise ValueError('Unsupported pre-action perturbation')
                elif spec.hook_timing == 'pre_close':
                    if not isinstance(perturbation, SetGripperPose):
                        raise ValueError('Expected a gripper displacement')
                    ctx.grasp_displace_to = perturbation.target_pos
                    ctx.grasp_displace_delta = perturbation.pos_delta
                result, ok, image, gs, text = execute_observe(ctx, registry, ck.tool_name, executed)
                if spec.hook_timing == 'pre_close' and not ctx.grasp_perturbation_applied:
                    raise ValueError('Grasp failed before perturbation applied; branch held')
                if spec.name == 'grasp_failure':
                    # Judge injected grasp failure independently of the merged API label.
                    effect = (getattr(ctx, 'last_grasp_feedback_raw', None) or result) in ('closed', 'blocked')
                elif spec.name == 'drop_during_transport':
                    from guava.collection.checkpoint import _find_grasped_object
                    effect = _find_grasped_object(env, [ck.grasped_object]) is None
                if spec.hook_timing == 'post':
                    if not isinstance(perturbation, SetObjectPose):
                        raise ValueError('Expected an object displacement')
                    perturbation.apply_to_sim(env)
                    env._settle(30)
                    _refresh(ctx)
                    image, gs = _encode(ctx.last_rgb), _gripper_state_text(ctx)
                if pass_history:
                    # Faults change execution, not the teacher's intended action or rationale.
                    messages.append({'role': 'assistant', 'content': ck.assistant_content})
                    if executed != intended:
                        text = ('Execution deviated from the requested arguments; actual arguments: '
                                + json.dumps(executed) + '. ' + text)
                    append_observation(messages, text, image, gs)
                    reasoning = checkpoint_reasoning(parent, ck.message_count - 1, ck.tool_name, intended)[0]
                    steps.append(Step(0, ck.tool_name, intended, result, reasoning, image, ok,
                                      depth_arr=ctx.last_depth.copy(), gripper_state=gs,
                                      reasoning_source='verified_parent_intended_action'))
            if not pass_history:
                _refresh(ctx)
                messages = _init_messages(parent_md['task'], _encode(ctx.last_rgb), tools_text,
                                          _gripper_state_text(ctx))
                if cfg.sim.perception == 'gt':
                    messages[1]['content'][-1]['text'] += '\n\nAccepted object names: ' + ', '.join(cfg.sim.object_aliases)
                messages[0]['content'] = parent['system']
        finally:
            ctx.grasp_displace_to = ctx.grasp_displace_delta = None
        if effect is False:
            raise ValueError('Requested failure did not occur; skip continuation without API calls')
        branch = {'original_trial_id': ck.trial_id, 'branch_turn': ck.turn_idx,
                  'pass_history': pass_history, 'perturbation_type': spec.name,
                  'perturbation_detail': perturbation.detail if perturbation else 'fresh_start',
                  'parent_path': str(Path(traj_path).resolve()),
                  'parent_sha256': hashlib.sha256(Path(traj_path).read_bytes()).hexdigest(),
                  'parent_collection_id': ck.collection_id, 'parent_prefix_digest': ck.prefix_digest,
                  'intended_action': {'name': ck.tool_name, 'arguments': intended},
                  'executed_action': None if spec.name == 'fresh_start' else {'name': ck.tool_name, 'arguments': executed},
                  'pre_close_perturbation_applied': ctx.grasp_perturbation_applied,
                  'failure_observed': effect,
                  'execution_motion_speed': float(env.cfg.motion_speed),
                  'restore_check': check}
        from dataclasses import asdict
        branch['perturbation_parameters'] = asdict(perturbation) if perturbation else None
        trajectory = continue_rollout(ctx, registry, provider, cfg, branch_id, messages, runtime,
                                steps=steps, branch_metadata=branch, tools_text=tools_text,
                                tools_schema=tools_schema)
        return trajectory
