"""Full-harness benchmark engine, preserving the tasks-branch physical scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
from guava.inference.rollout_env import RolloutEngine, _save_png, _to_data_url
from guava.inference.xml_tool_parser import parse
from guava.tools.tools import _refresh, _gripper_state_text
from guava.collection.trajectory import _fmt_tool_result

ROOT = Path(__file__).resolve().parents[2]
ALL_MODES = ('full',)
PHYSICAL_TASKS = frozenset(('apple_juice_order', 'apple_juice_reverse_order', 'can_in_bin',
                           'push_basket', 'red_objects_in_basket'))
OBJECTS = {
    'can_in_bin': ('apple', 'bin'), 'apple_juice_order': ('apple', 'juice'),
    'remove_cube_from_tray': ('cube', 'tray'), 'push_basket': ('basket',),
    'close_drawer': (), 'pick_up_carrot': ('carrot',),
    'tomato_near_potato': ('tomato', 'potato'), 'lemon_in_bin': ('lemon', 'bin'),
    'push_pot': ('pot',), 'cube_stack_reverse': ('cubeB', 'cubeA'),
    'apple_juice_reverse_order': ('apple', 'juice'),
    'bin_and_tray_simple': ('fork', 'banana', 'bin', 'tray'),
    'set_table': ('bowl', 'spoon', 'plate'), 'shell_game': ('cube', 'cup_a', 'cup_b', 'cup_c'),
    'red_objects_in_basket': ('tomato', 'can', 'potato', 'pear', 'basket'),
}

def build_prompt(mode='full'):
    if mode != 'full':
        raise ValueError('This release supports full-harness evaluation only')
    return (ROOT / 'guava/collection/system_prompt.txt').read_text()

def observation(mode, image_url, gripper, result='', initial=False,
                task_name='Pick up the orange from the table.', object_names=('orange',)):
    parts = [{'type': 'image_url', 'image_url': {'url': image_url}}]
    lines = ['<image>']
    if initial:
        lines.append('Task: ' + task_name + '\nObject names: ' + ', '.join(object_names) + '.')
    if result:
        lines.append(result)
    lines.append(gripper)
    parts.append({'type': 'text', 'text': '\n\n'.join(lines)})
    return parts

def decision(content, mode):
    if mode != 'full':
        raise ValueError('Only full-harness actions are supported')
    # Permit ONE leading reasoning block (including Qwen's implicit opening).
    # Never discard earlier actions by splitting at the LAST closing tag.
    answer = content.strip()
    if '</think>' in answer:
        prefix, answer = answer.split('</think>', 1)
        if '<think>' in prefix and (not prefix.startswith('<think>') or prefix.count('<think>') != 1):
            raise ValueError('Malformed leading reasoning block')
        answer = answer.strip()
    if '<think>' in answer or '</think>' in answer:
        raise ValueError('Multiple or unclosed reasoning blocks')
    from guava.inference.terminal_prose import terminal_with_prose
    terminal = terminal_with_prose(answer)
    if terminal:
        return 'terminal', terminal
    match = re.fullmatch(r'<tool_call>(.*?)</tool_call>', answer, re.S)
    if match and answer.count('<tool_call>') == 1 and answer.count('</tool_call>') == 1:
        body = match[1].strip()
        if body.startswith('{'):
            value = json.loads(body)
            if (not isinstance(value, dict) or set(value) != {'name', 'arguments'}
                    or not isinstance(value['name'], str) or not value['name'].strip()
                    or not isinstance(value['arguments'], dict)):
                raise ValueError('Invalid tool-call schema')
        else:
            value = parse(answer)['tool_call']
        if value and isinstance(value['arguments'], dict):
            return 'tool', value
    raise ValueError('Expected one mode-compatible action or a terminal declaration')

def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def state(engine):
    rob = engine.env.rob
    score = rob.score()
    if engine.cfg.sim.task == 'close_drawer':
        return {'held':None,
                'drawer_slide_qpos_m':float(rob.sim.data.qpos[rob._top_drawer_qadr]),
                'drawer_region_position_world':rob.sim.data.site_xpos[rob.top_region_id].tolist(),
                'success':score['success'], 'scorer_audit':score,
                'gripper':_gripper_state_text(engine.ctx)}
    names = OBJECTS[engine.cfg.sim.task]
    objects = {}
    for name in names:
        pos = rob.sim.data.body_xpos[getattr(rob, name+'_body_id')].copy()
        objects[name] = {'position_world':pos.tolist(),
                         'held':bool(rob._check_grasp(gripper=rob.robots[0].gripper,
                                                    object_geoms=getattr(rob,name)))}
    target = objects[names[0]]
    pos = target['position_world']
    result = {'held':target['held'], 'objects':objects,
            'position_world': pos, 'center_above_table_m': float(pos[2]-rob.table_offset[2]),
            'success': score['success'], 'gripper': _gripper_state_text(engine.ctx)}
    result['scorer_audit'] = score
    result['success'] = result['scorer_audit']['success']
    if engine.cfg.sim.task in PHYSICAL_TASKS:
        result['qpos'] = rob.sim.data.qpos.tolist()
        result['qvel'] = rob.sim.data.qvel.tolist()
    return result

def json_value(raw):
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, (dict, list)):
                return decoded
        except ValueError:
            pass
    return raw

def run_episode(engine, client, mode, index, root, seed):
    folder = root/mode/f'episode_{index:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    if engine.cfg.sim.task in ('push_pot', 'push_basket'):
        engine.env.rob.push_directions = ['left' if index % 2 == 0 else 'right']
    episode = engine.episode(index, folder)
    episode.reset(seed=seed)
    from guava.sim.video import record_episode
    with record_episode(engine.env, folder / 'episode.mp4',
                        getattr(engine, 'record_video', False), getattr(engine, 'video_fps', 20)):
        if engine.cfg.sim.task in ('push_pot', 'push_basket'):
            engine.cfg.task_name = engine.env.rob.task_instruction
        table_frame = getattr(engine, 'table_frame', False)
        if table_frame:
            from guava.inference import sft_frame
            offset = sft_frame.measured_offset(engine.env)
            write(folder/'coordinate_calibration.json', {'model_z_equals_base_z_plus':offset,
                  'audit_states_frame':'original robot base; object positions explicitly world'})
        registry = engine.registry
        actions, responses, audit_messages = [], [], []
        messages = [{'role': 'system', 'content': engine.system_prompt}]
        terminal = None
        started = time.time()

        def observe(result='', initial=False):
            _refresh(engine.ctx)
            image_path = folder/f'observation_{len(responses):03d}.png'
            _save_png(engine.ctx.last_rgb, image_path)
            gripper = _gripper_state_text(engine.ctx)
            if table_frame:
                gripper = sft_frame.gripper_text(engine.ctx, gripper, offset)
            parts = observation(mode, _to_data_url(engine.ctx.last_rgb), gripper, result, initial,
                                engine.cfg.task_name, tuple(engine.cfg.sim.object_aliases))
            message = {'role': 'user' if initial else 'tool', 'content': parts}
            messages.append(message)
            saved = {'role': message['role'], 'content': [
                {'type': 'image_url', 'image_url': {'url': str(image_path.resolve())}}, parts[1]]}
            audit_messages.append(saved)

        initial_state = state(engine)
        observe(initial=True)

        def invoke(name, *args, **kwargs):
            if len(actions) >= getattr(engine, 'max_actions', 25):
                raise RuntimeError('Tool-call budget exhausted')
            if name not in registry:
                raise ValueError(f'Unknown tool: {name}')
            before = state(engine)
            entry = {'index': len(actions), 'name': name, 'args': args, 'kwargs': kwargs, 'before': before}
            actions.append(entry)
            try:
                physical_kwargs = sft_frame.physical_arguments(name, kwargs, offset) if table_frame else kwargs
                entry['physical_kwargs'] = physical_kwargs
                raw = registry[name](*args, **physical_kwargs)
                if table_frame:
                    entry['physical_result'] = json_value(raw)
                    raw = sft_frame.model_result(name, raw, offset)
                entry.update(ok=True, result=json_value(raw))
                return json_value(raw)
            except Exception as exc:
                entry.update(ok=False, error=str(exc))
                raise
            finally:
                _refresh(engine.ctx)
                entry['after'] = state(engine)
                img = folder/f"action_{entry['index']:03d}_{name}.png"
                _save_png(engine.ctx.last_rgb, img)
                entry['image'] = str(img.resolve())
                write(folder/'actions.json', actions)

        end_reason = 'action_limit'
        try:
            while len(actions) < getattr(engine, 'max_actions', 25) and len(responses) < getattr(engine, 'max_decisions', 25):
                response = client.chat.completions.create(model=getattr(engine, 'model_name', 'Qwen/Qwen3.5-4B'), messages=messages,
                                                           temperature=0, max_tokens=8192)
                text = response.choices[0].message.content or ''
                responses.append({'content': text, 'finish_reason': response.choices[0].finish_reason,
                                  'usage': response.usage.model_dump() if response.usage else None})
                if getattr(response, 'response_audit', None) is not None:
                    responses[-1]['response_audit'] = response.response_audit
                write(folder/'responses.json', responses)
                if response.choices[0].finish_reason == 'length':
                    end_reason = 'incomplete_model_output'
                    break
                try:
                    kind, value = decision(text, mode)
                except ValueError:
                    end_reason = 'invalid_model_output'
                    break
                message = {'role': 'assistant', 'content': text}
                messages.append(message)
                audit_messages.append(message)
                if kind == 'terminal':
                    terminal = value
                    end_reason = 'model_terminal'
                    break
                feedback = ''
                try:
                    raw = invoke(value['name'], **value['arguments'])
                    feedback_args = dict(value['arguments'])
                    if table_frame and value['name'] in ('get_position', 'get_position_and_size'):
                        feedback_args['object_name'] = feedback_args.get('target_name', feedback_args.get('object_name', 'object'))
                    feedback = _fmt_tool_result(value['name'], feedback_args,
                                                json.dumps(raw) if isinstance(raw, (dict,list)) else str(raw))
                except Exception as exc:
                    feedback = f"Error calling {value['name']}: {exc}"
                observe(feedback)
                write(folder/'messages.json', [{'role':'system','content':engine.system_prompt}] + audit_messages)
        except Exception as exc:
            end_reason = 'infrastructure_error'
            write(folder/'error.json', {'type': type(exc).__name__, 'error': str(exc)})
        final = state(engine)
        _refresh(engine.ctx)
        _save_png(engine.ctx.last_rgb, folder/'final.png')
        final_at_stop = final
        stable_audits = []
        if engine.cfg.sim.task in PHYSICAL_TASKS:
            stable_audits.append(final)
            for _ in range(3):
                engine.env._settle(5)
                stable_audits.append(state(engine))
            final = stable_audits[-1]
            _refresh(engine.ctx)
            _save_png(engine.ctx.last_rgb, folder/'final_settled.png')
        write(folder/'messages.json', [{'role':'system','content':engine.system_prompt}] + audit_messages)
        result = {'task':engine.cfg.sim.task, 'mode':mode, 'episode':index, 'seed_label':seed, 'initial':initial_state,
                  'final':final, 'success':all(s['success'] for s in stable_audits) if stable_audits else final['success'],
                  'success_at_stop':final_at_stop['success'], 'stable_final_audits':stable_audits,
                  'ever_success': initial_state['success'] or any(a['after']['success'] for a in actions),
                  'end_reason':end_reason, 'terminal':terminal, 'actions':len(actions),
                  'decisions':len(responses), 'seconds':time.time()-started}
        write(folder/'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('initial','final')}), flush=True)
        return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--motion-speed', type=float, choices=(0.5, 1.0), default=0.5)
    ap.add_argument('--task', choices=tuple(OBJECTS), default='can_in_bin')
    ap.add_argument('--modes', nargs='+', choices=ALL_MODES, default=['full'])
    ap.add_argument('--seed-base', type=int, default=None)
    ap.add_argument('--camera', choices=('frontview', 'sideview', 'ftview'), default='frontview')
    ap.add_argument('--max-actions', type=int, default=25)
    ap.add_argument('--max-decisions', type=int, default=25)
    ap.add_argument('--model', default=None)
    ap.add_argument('--backend', choices=('local', 'openrouter', 'openai', 'nvidia'), default='openrouter')
    ap.add_argument('--model-revision', default=None)
    ap.add_argument('--system-prompt', type=Path)
    ap.add_argument('--table-frame', action='store_true')
    ap.add_argument('--grasp', choices=('pca', 'graspgen'), default='pca')
    ap.add_argument('--graspgen-url', default='tcp://127.0.0.1:5556')
    from guava.sim.video import add_video_arguments, validate_video_fps
    add_video_arguments(ap)
    args = ap.parse_args()
    validate_video_fps(args.video_fps)
    from guava.llm.provider_settings import DEFAULTS
    args.model = args.model or ('Qwen/Qwen3.5-4B' if args.backend == 'local' else DEFAULTS[args.backend][2])
    if args.seed_base is None:
        import secrets
        args.seed_base = secrets.randbelow(2**31)
    if args.backend == 'nvidia' and args.model != 'openai/openai/gpt-5.4':
        raise ValueError('NVIDIA backend requires the explicit GPT-5.4 model')
    if args.max_actions <= 0 or args.max_decisions <= 0:
        raise ValueError('Action and decision caps must be positive')
    if (args.table_frame or args.system_prompt) and args.modes != ['full']:
        raise ValueError('Custom SFT interface currently supports full mode only')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    if len(set(args.modes)) != len(args.modes) or args.episodes <= 0:
        raise ValueError('Select unique modes and a positive episode count')
    prompts = {mode:build_prompt(mode) for mode in args.modes}
    if args.system_prompt:
        prompts = {mode:args.system_prompt.read_text() for mode in args.modes}
    else:
        # The built-in collection prompt now always describes table coordinates.
        args.table_frame = True
    for mode, text in prompts.items():
        (root/f'prompt_{mode}.txt').write_text(text)
    files = ['guava/inference/benchmark_engine.py', 'guava/sim/video.py',
             'guava/inference/rollout_env.py',f'configs/{args.task}.yaml',
             'guava/collection/system_prompt.txt','guava/tools/tools.py',
             'guava/robot/arm.py','guava/robot/ik/ik_pose.py','guava/sim/env.py',
             f'guava/sim/tasks/{args.task}.py','guava/config.py']
    if args.table_frame:
        files.append('guava/inference/sft_frame.py')
    if args.backend == 'nvidia':
        files.extend(['guava/inference/nvidia_eval_client.py', 'guava/llm/providers/responses.py'])
    if args.backend in ('openrouter', 'openai'):
        files.extend(['guava/inference/remote_eval_client.py', 'guava/llm/provider_settings.py',
                      'guava/llm/providers/__init__.py', 'guava/llm/providers/openai_compat.py',
                      'guava/llm/providers/responses.py'])
    if args.camera == 'ftview':
        files.append('third_party/robosuite/robosuite/models/assets/arenas/table_arena.xml')
    files.append('guava/sim/scoring.py')
    if args.task == 'apple_juice_reverse_order':
        files.append('guava/sim/tasks/apple_juice_order.py')
    scorer_version = 'task-owned-v1'
    write(root/'protocol.json', {'task':args.task,'modes':args.modes,'episodes_per_mode':args.episodes,
          'model':args.model,'model_revision':args.model_revision,
          'prompt_sha256':{m:hashlib.sha256(p.encode()).hexdigest() for m,p in prompts.items()},
          'backend':args.backend, 'record_video':args.record_video, 'video_fps':args.video_fps,
          'endpoint':DEFAULTS[args.backend][0] if args.backend != 'local' else 'http://127.0.0.1:8000/v1',
          'reasoning_effort':'medium' if args.backend in ('nvidia', 'openai') else None,
          'native_thinking_default':args.backend == 'local','temperature':None if args.backend != 'local' else 0,
          'max_tokens':8192,'max_actions':args.max_actions,'max_decisions':args.max_decisions,'format_attempts':1,
          'perception':'sam3','grasp':args.grasp, 'ik':'pybullet',
          'camera':args.camera,
          'motion_speed':args.motion_speed,'frame':'table-aligned' if args.table_frame else 'original robot base','auto_stop_success':False,
          'scorer':scorer_version,
          'post_stop_stability_steps':[5,5,5] if args.task in PHYSICAL_TASKS else [],
          'seed_base':args.seed_base,
          'initial_settle_steps':20,
          'seeding':'eval-ubuntu effective seed: Python/NumPy, recreated nested samplers, camera and visual randomization; reproducibility requires identical simulator/configuration',
          'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}})
    from openai import OpenAI
    if args.backend == 'nvidia':
        from guava.inference.nvidia_eval_client import NvidiaEvalClient
        client = NvidiaEvalClient(ROOT)
    elif args.backend in ('openrouter', 'openai'):
        from guava.inference.remote_eval_client import RemoteEvalClient
        client = RemoteEvalClient(args.backend, args.model)
    else:
        client = OpenAI(base_url='http://127.0.0.1:8000/v1',api_key='EMPTY',timeout=180,max_retries=2)
        if args.model not in [m.id for m in client.models.list().data]:
            raise ValueError('Requested model is not served')
    import yaml
    config = yaml.safe_load((ROOT/f'configs/{args.task}.yaml').read_text())
    config['grasp'] = args.grasp
    config['stop_on_reward'] = False
    config['sim'].update(perception='sam3', grasp=args.grasp, motion_speed=args.motion_speed,
                         visualize=False, camera_views=[args.camera])
    config['sim']['camera'] = {'width':512,'height':512,'name':args.camera}
    config.setdefault('robot',{}).update(sam3_url='http://127.0.0.1:8114',sam3_debug=False,bbox_debug=False, graspgen_url=args.graspgen_url)
    effective_config = root/'effective-config.yaml'
    effective_config.write_text(yaml.safe_dump(config,sort_keys=False))
    engine = RolloutEngine(effective_config,system_prompt=prompts[args.modes[0]],camera_name=args.camera)
    engine.record_video, engine.video_fps = args.record_video, args.video_fps
    engine.model_name = args.model
    engine.initial_settle_steps = 20
    engine.max_actions = args.max_actions
    engine.max_decisions = args.max_decisions
    engine.table_frame = args.table_frame
    engine.cfg.stop_on_reward = False
    engine.cfg.sim.motion_speed = args.motion_speed
    engine.env.cfg.motion_speed = args.motion_speed
    results=[]
    for mi, mode in enumerate(args.modes):
        engine.system_prompt = prompts[mode]
        for index in range(args.episodes):
            result=run_episode(engine,client,mode,index,root,args.seed_base+mi*100+index)
            results.append(result)
            write(root/'summary.json',results)
            if result['end_reason']=='infrastructure_error':
                raise RuntimeError('Stopping campaign after infrastructure error; preserve failed scene for diagnosis')

if __name__ == '__main__':
    main()
