"""Idempotently apply the tracked SAM3 patch, without fetching or discarding edits."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '967fdd651f71ca14949122fed4c918a778ca9334'


def setup(check=False):
    repo = ROOT / 'third_party/sam3'
    patch = ROOT / 'patches/sam3-importlib-resources.patch'
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    head = git('rev-parse', 'HEAD')
    if head.returncode or head.stdout.strip() != PIN:
        raise RuntimeError('Initialize the pinned SAM3 submodule first; refusing to patch another revision.')
    if git('apply', '--reverse', '--check', str(patch)).returncode == 0:
        print('SAM3 compatibility patch already applied.')
        return
    if git('apply', '--check', str(patch)).returncode:
        raise RuntimeError('SAM3 has conflicting edits; no files were changed.')
    if check:
        raise RuntimeError('SAM3 patch is missing; run this script without --check.')
    result = git('apply', str(patch))
    if result.returncode:
        raise RuntimeError(result.stderr)
    print('Applied SAM3 importlib.resources compatibility patch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    setup(parser.parse_args().check)
