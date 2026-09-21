"""Portable 15-task benchmark launcher for GPT, Qwen and Guava.

Run with the evaluation Python. --dry-run prints a plan without making requests.
Each run samples fresh initialization seeds and records them for reproducibility.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
import secrets
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from guava.inference.study_protocols import (
    ROOT, VERSION, BENCHMARK, UNSEEN, CAPS, MODELS,
    interface, episode_command, model_directory,
)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', help='Exact provider model IDs, Hugging Face IDs, or local checkpoint directories.')
    p.add_argument('--provider', choices=('openrouter', 'openai', 'local', 'nvidia'), default='openrouter',
                   help='Provider for all selected models; use local for Hugging Face/vLLM.')
    p.add_argument('--prompt-profile', choices=('auto', 'long', 'sft'), default='auto',
                   help='Auto: short for AIcell/guava-*; long otherwise. Use sft for custom Guava checkpoint paths.')
    p.add_argument('--tasks', nargs='+', choices=list(CAPS))
    p.add_argument('--unseen', action='store_true')
    p.add_argument('--episodes', type=int)
    p.add_argument('--max-turns', type=int, help='Override BOTH action and decision caps for every task.')
    p.add_argument('--output', type=Path, required=True, help='New directory under data/ or runs/.')
    p.add_argument('--seed', type=int, help='Optional reproducible seed; default samples a fresh seed per run.')
    p.add_argument('--serve-python', type=Path, default=ROOT / '.venv-serve/bin/python')
    p.add_argument('--vllm', type=Path, default=ROOT / '.venv-serve/bin/vllm')
    p.add_argument('--sam-python', type=Path, default=ROOT / '.venv-sam3/bin/python')
    p.add_argument('--external-servers', action='store_true', help='Single-model runs only; operator owns servers.')
    p.add_argument('--grasp', choices=['pca', 'graspgen'], default='pca')
    p.add_argument('--graspgen-url', default='tcp://127.0.0.1:5556')
    p.add_argument('--revision', help='Revision for one local-provider Hugging Face model; known models have pinned defaults.')
    p.add_argument('--dry-run', action='store_true')
    from guava.sim.video import add_video_arguments, validate_video_fps
    add_video_arguments(p)
    args = p.parse_args(argv)
    try:
        validate_video_fps(args.video_fps)
    except ValueError as exc:
        p.error(str(exc))
    args.study = 'benchmark'
    args.modes = ['full']
    from guava.llm.provider_settings import DEFAULTS
    args.models = args.models or ([DEFAULTS[args.provider][2]] if args.provider != 'local'
                                  else ['AIcell/guava-v13b-qwen3.5-4b'])
    if args.revision and (args.provider != 'local' or len(args.models) != 1):
        p.error('--revision requires exactly one model with --provider local')
    retired = {'gpt54', 'qwen4b', 'guava-cf', 'guava-nocf', 'guava-matched-nocf'}
    if any(not m.strip() or m in retired for m in args.models):
        p.error('Use exact provider model IDs instead of model aliases')
    args.model_catalog = {}
    for model in args.models:
        revision = None
        if args.provider == 'local':
            if Path(model).is_dir():
                model_id = str(Path(model).resolve())
                if args.revision:
                    p.error('--revision does not apply to a checkpoint directory')
            else:
                model_id = model
                if len(model.split('/')) != 2 or any(x in ('', '.', '..') for x in model.split('/')):
                    p.error('Local models require an existing directory or a Hugging Face organization/model ID')
                revision = args.revision or MODELS.get(model, (model, None))[1]
                if not revision:
                    p.error('An unpinned Hugging Face model requires --revision for reproducibility')
        else:
            model_id = model
        if args.provider == 'nvidia' and model_id != DEFAULTS['nvidia'][2]:
            p.error('The internal adapter only supports its configured model ID')
        args.model_catalog[model] = (model_id, revision)
    args.model_directories = {model: model_directory(model) for model in args.models}
    args.tasks = args.tasks or (UNSEEN if args.unseen else BENCHMARK)
    if args.unseen and args.tasks != UNSEEN:
        p.error('--unseen cannot be combined with a different --tasks list')
    args.episodes = args.episodes if args.episodes is not None else 15
    if args.episodes <= 0 or (args.max_turns is not None and args.max_turns <= 0):
        p.error('Episode and turn counts must be positive')
    if any(len(v) != len(set(v)) for v in (args.models, args.tasks, args.modes)):
        p.error('Duplicate models, tasks or modes are not allowed')
    if args.external_servers and len(args.models) != 1:
        p.error('External servers require exactly one model')
    args.output = args.output.resolve()
    if not any(args.output.is_relative_to(ROOT / d) and args.output != ROOT / d for d in ('data', 'runs')):
        p.error('--output must be a new subdirectory under ignored data/ or runs/')
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)
    if not 0 <= args.seed < 2**32 - len(BENCHMARK) * args.episodes:
        p.error('--seed and episode count exceed the supported 32-bit seed range')
    args.seed_bases = {task: args.seed + BENCHMARK.index(task) * args.episodes for task in args.tasks}
    return args


def plan(args):
    return [{'model': model, 'task': task, 'mode': mode,
             'cap': args.max_turns or CAPS[task], 'episodes': args.episodes,
             'interface': interface(args.study, model, mode, args.prompt_profile),
             'command': episode_command(sys.executable, args.study, model, task, mode,
                                        args.episodes, args.max_turns or CAPS[task],
                                        args.output / args.model_directories[model] / mode / task, catalog=args.model_catalog,
                                        seed_base=args.seed_bases[task], provider=args.provider, prompt_profile=args.prompt_profile)
                        + ['--grasp', args.grasp, '--graspgen-url', args.graspgen_url]
                        + (['--record-video', '--video-fps', str(args.video_fps)] if args.record_video else [])}

            for model in args.models for mode in args.modes for task in args.tasks]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare_bundled_cuda(cuda_home):
    """Provide linker aliases omitted by NVIDIA's pip wheels for FlashInfer JIT.

    Only add missing relative symlinks inside the selected serving environment;
    never overwrite existing files or modify a caller-supplied CUDA_HOME.
    """
    for link, target in ((cuda_home / 'lib64', 'lib'),
                         (cuda_home / 'lib' / 'libcudart.so', 'libcudart.so.13')):
        if not link.exists() and not link.is_symlink():
            destination = link.parent / target
            if not destination.exists():
                raise RuntimeError(f'Incomplete bundled CUDA installation: missing {destination}')
            link.symlink_to(target, target_is_directory=destination.is_dir())


def ready(url, proc=None):
    for _ in range(240):
        if proc is not None and proc.poll() is not None:
            raise RuntimeError('Server exited before readiness; inspect its log')
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError('Server readiness timeout')


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=60)


def server_command(args, model_id, revision):
    return [str(args.vllm), 'serve', model_id,
            *(['--revision', revision] if revision else []),
            '--served-model-name', model_id, '--host', '127.0.0.1', '--port', '8000',
            '--dtype', 'bfloat16', '--max-model-len', '32768', '--max-num-seqs', '1',
            '--gpu-memory-utilization', '0.80', '--limit-mm-per-prompt', '{"image":31}']


def main(argv=None):
    args = arguments(argv)
    jobs = plan(args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    if args.provider in ('openrouter', 'openai'):
        from guava.llm.provider_settings import DEFAULTS
        key_env = DEFAULTS[args.provider][1]
        if not os.environ.get(key_env):
            raise RuntimeError(f'Set {key_env} before starting evaluation')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a run; choose a new output directory')
    # Do not take over a service started by another experiment.
    if not args.external_servers:
        for port in (8000, 8114):
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Port {port} is occupied; stop your service or use --external-servers')
    subprocess.run([sys.executable, str(ROOT / 'scripts/setup_sam3_compat.py'), '--check'], check=True)
    args.output.mkdir(parents=True)
    sources = [p for base in ('guava', 'configs', 'scripts', 'patches')
               for p in (ROOT / base).rglob('*') if p.is_file() and p.suffix in ('.py', '.txt', '.yaml', '.patch')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = {'version': VERSION, 'study': args.study, 'jobs': jobs,
                'model_revisions': {m: args.model_catalog[m] for m in args.models},
                'model_directories': args.model_directories, 'prompt_profile': args.prompt_profile, 'source_hashes': hashes,
                'initialization': 'randomized-reset', 'seed': args.seed,
                'seed_bases': args.seed_bases, 'initial_settle_steps': 20,
                'motion_speed': 1, 'stop_on_reward': False, 'max_output_tokens': 8192,
                'temperature_local': 0, 'hosted_provider': args.provider, 'gpt_reasoning_effort': 'medium' if args.provider in ('openai', 'nvidia') else None,
                'grasp': args.grasp, 'rl_format_repairs': False,
                'record_video': args.record_video, 'video_fps': args.video_fps,
                'runtime': {'evaluation_python': sys.executable, 'serve_python': str(args.serve_python),
                            'sam_python': str(args.sam_python), 'vllm': str(args.vllm)},
                'started_unix': time.time()}
    write(args.output / 'manifest.json', manifest)
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'PYTHONPATH': str(ROOT)}
    # Compiler helpers such as ninja live beside vLLM in its isolated environment.
    env['PATH'] = str(args.vllm.absolute().parent) + os.pathsep + env.get('PATH', '')
    # Existing 5090 serving environments bundle CUDA 13; other installations
    # may provide CUDA_HOME/PATH themselves. Never assume a user's home path.
    import glob
    bundled = glob.glob(str(args.serve_python.resolve().parent.parent / 'lib/python*/site-packages/nvidia/cu13'))
    # venv Python can be a symlink to the system binary; also check the venv path.
    bundled += glob.glob(str(args.serve_python.absolute().parent.parent / 'lib/python*/site-packages/nvidia/cu13'))
    if bundled and 'CUDA_HOME' not in env:
        prepare_bundled_cuda(Path(bundled[0]))
        env['CUDA_HOME'] = bundled[0]
        env['PATH'] = str(Path(bundled[0]) / 'bin') + os.pathsep + env.get('PATH', '')
    model_proc = sam = None
    try:
        for model in args.models:
            folder = args.output / args.model_directories[model]
            folder.mkdir()
            model_id, revision = args.model_catalog[model]
            if not args.external_servers:
                if args.provider == 'local':
                    if revision:
                        subprocess.run([str(args.serve_python), '-c',
                        'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2])',
                        model_id, revision], check=True, env=env)
                    with (folder / 'model-server.log').open('x') as log:
                        model_proc = subprocess.Popen(server_command(args, model_id, revision),
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    ready('http://127.0.0.1:8000/v1/models', model_proc)
                with (folder / 'sam3.log').open('x') as log:
                    sam = subprocess.Popen([str(args.sam_python), '-u', '-m', 'guava.scripts.sam3_server',
                        '--host', '127.0.0.1', '--port', '8114'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            ready('http://127.0.0.1:8114/health', sam)
            for job in [j for j in jobs if j['model'] == model]:
                for p, digest in hashes.items():
                    if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest:
                        raise RuntimeError(f'Source changed during study: {p}')
                print('START', model, job['mode'], job['task'], flush=True)
                with (folder / f"{job['mode']}-{job['task']}.log").open('x') as log:
                    subprocess.run(job['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                dest = folder / job['mode'] / job['task']
                rows = json.loads((dest / 'summary.json').read_text())
                if len(rows) != args.episodes or any(r['end_reason'] == 'infrastructure_error' for r in rows):
                    raise RuntimeError('Incomplete task or infrastructure failure; all artifacts preserved')
                print('DONE', model, job['mode'], job['task'], sum(r['success'] for r in rows),
                      dict(Counter(r['end_reason'] for r in rows)), flush=True)
            stop(sam); sam = None
            stop(model_proc); model_proc = None
        write(args.output / 'complete.json', {'episodes': sum(j['episodes'] for j in jobs), 'time': time.time()})
    except Exception as exc:
        # No exception text: provider exceptions can contain credentials.
        write(args.output / 'failed.json', {'error_type': type(exc).__name__, 'time': time.time()})
        raise
    finally:
        try:
            stop(sam)
        finally:
            stop(model_proc)


if __name__ == '__main__':
    main()
