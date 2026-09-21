"""Shared collection loop: one decision, one execution, one saved observation."""
from copy import deepcopy
import inspect
import json
import re
import uuid
import numpy as np
from guava.collection.trajectory import Step, Trajectory, _fmt_tool_result
from guava.collection.snapshot import capture, digest
from guava.llm.client import query_tool_call

MUTATING = {'move', 'grasp', 'align', 'close_gripper', 'release', 'rotate', 'home_pose'}


def configure_context(env, arm, perception, cfg, grasp_planner=None):
    from guava.tools.tools import ToolContext, make_tool_registry
    from guava.tools.coordinate_frame import measured_offset, model_registry
    from guava.collection.trial import _SYSTEM_PROMPT
    from guava.tools.schema import tools_to_prompt, build_tools_schema
    if cfg.model_frame != 'table':
        raise ValueError('The collection prompt requires model_frame=table; base-frame collection is no longer supported.')
    ctx = ToolContext(env, arm, perception, grasp_planner=grasp_planner,
                      graspgen_horizontal=getattr(cfg.robot, 'graspgen_horizontal', False),
                      graspgen_vertical=getattr(cfg.robot, 'graspgen_vertical', False))
    system = _SYSTEM_PROMPT
    ctx.model_z_offset = measured_offset(env)
    registry = model_registry(ctx, make_tool_registry(ctx))
    runtime = {'collector_version': 2, 'grasp_feedback_policy': 'blocked_merged_into_grasped',
               'collection_id': str(uuid.uuid4()),
               'task_key': cfg.sim.task, 'coordinate_frame': 'table_aligned',
               'offset_added_to_absolute_z': ctx.model_z_offset, 'model': cfg.llm.model,
               'ik': cfg.robot.ik_backend, 'perception': cfg.sim.perception,
               'grasp': cfg.sim.grasp, 'motion_speed': float(env.cfg.motion_speed), 'checkpoints': {}}
    return ctx, registry, system, tools_to_prompt(registry), build_tools_schema(registry), runtime


def validate_decision(tc, registry):
    """Validate the public completion, never execute a rationale example or partial JSON."""
    raw = tc.raw_content or ''
    if tc.tool_name is None:
        outside = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
        if outside not in ('Task complete.', 'Task failed.'):
            raise ValueError('Expected a single action or an explicit terminal message')
        return
    if tc.tool_name not in registry:
        raise ValueError('Unknown tool')
    m = re.fullmatch(r'\s*<think>\s*(.+?)\s*</think>\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*', raw, re.S)
    if not m or not m[1].strip() or '<tool_call>' in m[1] or '</think>' in m[1]:
        raise ValueError('Missing public action rationale or ambiguous action format')
    call = json.loads(m[2])  # strict; reject trailing garbage and extra calls
    if call != {'name': tc.tool_name, 'arguments': tc.tool_args}:
        raise ValueError('Parsed action differs from public completion')
    inspect.signature(registry[tc.tool_name]).bind(**tc.tool_args)
    def finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError('Nonfinite action argument')
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        if isinstance(value, list):
            for v in value: finite(v)
    finite(tc.tool_args)
    width = tc.tool_args.get('gripper_width')
    if width is not None and (isinstance(width, bool) or not isinstance(width, (int, float))
                              or not 0 <= width <= 100):
        raise ValueError('Gripper width outside [0,100]')


def query_checked(provider, messages, cfg, registry, diagnostics):
    # Formatting retries never execute an action or mutate the accepted transcript.
    for attempt in range(3):
        try:
            tc = query_tool_call(provider, messages, cfg)
            validate_decision(tc, registry)
            return tc
        except (ValueError, KeyError, TypeError) as exc:
            diagnostics.append({'kind': 'rejected_completion', 'attempt': attempt + 1,
                                'error_type': type(exc).__name__})
    raise ValueError('Three invalid completions; collection stopped without executing them')


def execute_observe(ctx, registry, name, args):
    from guava.collection.trial import _execute
    from guava.tools.tools import _refresh, _encode, _gripper_state_text
    result, ok = _execute(registry, name, args)
    # A failed motion may still have moved the robot. Always observe its result.
    _refresh(ctx)
    image = _encode(ctx.last_rgb)
    gs = _gripper_state_text(ctx)
    text = _fmt_tool_result(name, args, result) if ok else result
    return result, ok, image, gs, text


def append_observation(messages, text, image, gripper):
    content = []
    if image:
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}})
    content.append({'type': 'text', 'text': text + ('\n' + gripper if gripper else '')})
    messages.append({'role': 'user', 'content': content})


def continue_rollout(ctx, registry, provider, cfg, trial_id, messages, runtime,
                     *, steps=None, branch_metadata=None, tools_text='', tools_schema=None,
                     save_checkpoints=False, loss_by_message=None):
    from guava.collection import checkpoint as checkpoints
    env, arm = ctx.env, ctx.arm
    steps = [] if steps is None else steps
    losses = {} if loss_by_message is None else loss_by_message
    diagnostics = []
    terminal = None
    end_reason = 'turn_budget'
    prev_tool = steps[-1].tool_name if steps else None
    # Count total history actions as well as continuation actions: no >30 exports.
    prior_actions = sum(m['role'] == 'assistant' and '<tool_call>' in str(m['content']) for m in messages)
    budget = min(cfg.turn_budget, max(0, 30 - prior_actions) + 1)
    for _ in range(budget):
        try:
            tc = query_checked(provider, messages, cfg, registry, diagnostics)
        except Exception as exc:
            end_reason = 'query_error'
            diagnostics.append({'kind': end_reason, 'error_type': type(exc).__name__})
            break
        index = len(messages)
        turn = len(steps)
        if tc.tool_name is None:
            messages.append({'role': 'assistant', 'content': tc.raw_content})
            steps.append(Step(turn, None, {}, tc.raw_content, tc.reasoning, None, True,
                              reasoning_source=tc.reasoning_source, model_message=tc.model_message))
            terminal = re.sub(r'<think>.*?</think>', '', tc.raw_content, flags=re.S).strip()
            end_reason = 'terminal'
            break
        if prior_actions >= 30:
            end_reason = 'action_limit'
            break
        if save_checkpoints:
            ck = checkpoints.save(env, messages, tc.tool_name, tc.tool_args, prev_tool,
                                  trial_id, turn, list(cfg.sim.object_aliases.values()))
            ck.version = 2
            ck.snapshot = capture(env, arm, ctx)
            ck.messages = deepcopy(messages)
            ck.collection_id = runtime['collection_id']
            ck.prefix_digest = digest(messages)
            ck.assistant_content = tc.raw_content
            ck.runtime_metadata = {k: deepcopy(v) for k, v in runtime.items() if k != 'checkpoints'}
            checkpoints.save_to_disk(ck, cfg.output_dir, trial_id)
            runtime['checkpoints'][str(turn)] = {'message_count': len(messages),
                                                 'prefix_digest': ck.prefix_digest,
                                                 'action_digest': digest(tc.raw_content)}
        result, ok, image, gs, text = execute_observe(ctx, registry, tc.tool_name, tc.tool_args)
        messages.append({'role': 'assistant', 'content': tc.raw_content})
        append_observation(messages, text, image, gs)
        steps.append(Step(turn, tc.tool_name, tc.tool_args, result, tc.reasoning, image, ok,
                          depth_arr=ctx.last_depth.copy(), gripper_state=gs,
                          reasoning_source=tc.reasoning_source, model_message=tc.model_message))
        print(f'[collect {trial_id:04d}] action {turn}: {tc.tool_name} ok={ok}', flush=True)
        prev_tool = tc.tool_name
        prior_actions += 1
        if env._episode_done:
            end_reason = 'simulator_horizon'
            break
    # Shaped rewards can exceed 1 before success (notably Stack). Use the task predicate.
    reward = float(env.rob.reward(action=None))
    success = bool(env.rob._check_success())
    quality = {'eligible': success and terminal == 'Task complete.',
               'end_reason': end_reason, 'terminal': terminal, 'diagnostics': diagnostics,
               'task_success': success, 'action_count': prior_actions}
    task = getattr(env.rob, 'task_instruction', None) or cfg.task_name
    return Trajectory(trial_id, task, [], steps, reward, success, len(steps),
                      tools_text=tools_text, tools_schema=tools_schema or [],
                      system_prompt=messages[0]['content'], runtime_metadata=runtime,
                      branch_metadata=branch_metadata, messages=deepcopy(messages),
                      loss_by_message=losses, quality=quality)
