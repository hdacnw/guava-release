"""Versioned task/model catalog and explicit experiment interface selection.

Default full-mode interface is the latest Guava SFT evaluation protocol:
first-call compatibility parsing, dynamic tools, and explicit model termination.
No ablation or inference-efficiency runner is included.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'pre-release-v9-native-model-ids'
BENCHMARK = ['pick_up_carrot', 'tomato_near_potato', 'lemon_in_bin', 'push_pot',
             'cube_stack_reverse', 'apple_juice_reverse_order', 'bin_and_tray_simple',
             'set_table', 'can_in_bin', 'apple_juice_order', 'remove_cube_from_tray',
             'push_basket', 'close_drawer', 'shell_game', 'red_objects_in_basket']
UNSEEN = BENCHMARK[:8]
CAPS = {t: (30 if t in ('bin_and_tray_simple', 'set_table', 'shell_game',
                       'red_objects_in_basket') else 25) for t in BENCHMARK}
MODELS = {
    'openai/gpt-5.4': ('openai/gpt-5.4', None),
    'Qwen/Qwen3.5-4B': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    'AIcell/guava-v13b-qwen3.5-4b': ('AIcell/guava-v13b-qwen3.5-4b', '19a8c77544cf7c161966f49bbb2654b85c6b2fa3'),
    'AIcell/guava-v13b-qwen3.5-4b-no-counterfactual': ('AIcell/guava-v13b-qwen3.5-4b-no-counterfactual', '37e7805eaf5dcf39be4292f732afe3041a50fb23'),
    'AIcell/guava-v13b-qwen3.5-4b-matched-nocf': ('AIcell/guava-v13b-qwen3.5-4b-matched-nocf', '405bd0ea8760d1c1abd96109b7a60c37d92195ef'),
}


def interface(study, model, mode, prompt_profile='auto'):
    if study != 'benchmark' or mode != 'full':
        raise ValueError('Only the full benchmark protocol is supported')
    if prompt_profile not in ('auto', 'long', 'sft'):
        raise ValueError('Unknown prompt profile')
    short = prompt_profile == 'sft' or (prompt_profile == 'auto' and model.startswith('AIcell/guava-'))
    prompt = 'configs/prompts/sft_v13b_short.txt' if short else 'guava/collection/system_prompt.txt'
    return {'runner': 'guava.inference.benchmark_compatible_tasks',
            'prompt': prompt, 'frame': 'table-aligned',
            'parser': 'first-call-compatibility-audited'}


def episode_command(python, study, model, task, mode, episodes, cap, output, catalog=None, seed_base=None, provider='openrouter', prompt_profile='auto'):
    model_id, revision = (catalog or {}).get(model, (model, None))
    spec = interface(study, model, mode, prompt_profile)
    cmd = [str(python), '-u', '-m', spec['runner'], '--backend',
           provider, '--model', model_id,
           '--task', task, '--modes', mode, '--episodes', str(episodes),
           '--max-actions', str(cap), '--max-decisions', str(cap),
           '--camera', 'sideview' if task == 'shell_game' else 'frontview',
           '--motion-speed', '1', '--output-dir', str(output)]
    if seed_base is not None:
        cmd += ['--seed-base', str(seed_base)]
    if revision:
        cmd += ['--model-revision', revision]
    if spec['prompt']:
        cmd += ['--system-prompt', str(ROOT / spec['prompt'])]
    if spec['frame'] == 'table-aligned':
        cmd += ['--table-frame']
    return cmd


def model_directory(model):
    """A readable collision-resistant directory, never an unchecked model path."""
    import hashlib
    import re
    label = re.sub(r'[^A-Za-z0-9_-]+', '--', model).strip('-')[:100] or 'model'
    return label + '-' + hashlib.sha256(model.encode()).hexdigest()[:12]
