"""Deterministic JSON trajectory filter. Never edits reasoning, actions or labels.

Writes an acceptance manifest, not a reconstructed training dataset. Source files
and images remain untouched. Human visual review is still required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guava.task_catalog import TRAINING_TASKS, ALL_TASKS


def metadata(record):
    value = record.get('metadata', {})
    return json.loads(value) if isinstance(value, str) else value


def resolve_image(name, source, image_root=None):
    p = Path(name)
    candidates = [p] if p.is_absolute() else [source.parent / p, source.parent.parent / p, Path.cwd() / p]
    if image_root:
        candidates.insert(0, Path(image_root) / p)
    return next((p.resolve() for p in candidates if p.is_file()), None)


def inspect_record(record, source, max_actions=30, image_root=None):
    errors, warnings, actions = [], [], []
    md = metadata(record)
    task = md.get('runtime', {}).get('task_key')
    if not task:
        task = next((part for part in reversed(source.parts) if part in ALL_TASKS), None)
    if task and task not in TRAINING_TASKS:
        errors.append('not_a_training_task')
    elif not task:
        warnings.append('task_scope_unverified')
    label = re.search(r'_success_([01])\.json$', source.name)
    success = label[1] == '1' if label else md.get('success') is True
    if not success:
        errors.append('not_success_labeled')
    if label and md.get('success') is not None and md['success'] != success:
        warnings.append('filename_overrides_metadata_label')
    turns = record.get('conversations', [])
    if not turns or any(t.get('from') not in ('human', 'gpt') or not isinstance(t.get('value'), str) for t in turns):
        errors.append('invalid_conversation')
        return errors, warnings, None
    for turn in turns:
        if turn['from'] != 'gpt':
            continue
        text = turn['value']
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
        if text.count('<tool_call>') != len(calls) or len(calls) > 1:
            errors.append('malformed_or_multiple_calls')
        for body in calls:
            try:
                call = json.loads(body)
                if not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict):
                    raise ValueError('invalid schema')
                args = call['arguments']
                def finite(value):
                    if isinstance(value, float) and not math.isfinite(value):
                        return False
                    if isinstance(value, dict):
                        return all(finite(v) for v in value.values())
                    if isinstance(value, list):
                        return all(finite(v) for v in value)
                    return True
                if not finite(args):
                    errors.append('nonfinite_argument')
                if 'gripper_width' in args:
                    w = args['gripper_width']
                    if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0 <= w <= 100:
                        errors.append('gripper_width_outside_0_100')
                if sum(k in args for k in ('target_name', 'target', 'object_name')) > 1:
                    errors.append('conflicting_target_arguments')
                actions.append(call)
            except (ValueError, TypeError, AttributeError):
                errors.append('invalid_tool_json')
        if calls and turn.get('loss') is not False:
            rationale = re.search(r'<think>(.*?)</think>', text, re.S)
            if not rationale or not rationale[1].strip():
                errors.append('empty_supervised_reasoning')
    if not actions or len(actions) > max_actions:
        errors.append('action_count_outside_limit')
    terminal = re.sub(r'<think>.*?</think>', '', turns[-1]['value'], flags=re.S).strip()
    if turns[-1]['from'] != 'gpt' or not terminal.endswith('Task complete.') or '<tool_call>' in terminal:
        errors.append('missing_explicit_completion')
    images = record.get('images', [])
    if not isinstance(images, list):
        images = []
        errors.append('invalid_images')
    if sum(t['value'].count('<image>') for t in turns) != len(images) or not images:
        errors.append('image_placeholder_mismatch')
    hashes = []
    for name in images:
        image = resolve_image(name, source, image_root)
        if image is None:
            errors.append('missing_image')
        else:
            hashes.append(hashlib.sha256(image.read_bytes()).hexdigest())
    quality = md.get('quality', {})
    if quality.get('eligible') is False:
        errors.append('collector_quality_rejected')
    branch = md.get('branch', {}) or md.get('branch_metadata', {})
    if branch:
        if branch.get('failure_observed') is False:
            errors.append('perturbation_not_observed')
        parent = branch.get('parent_path')
        if parent and branch.get('parent_sha256'):
            p = Path(parent)
            if not p.is_file():
                warnings.append('parent_unavailable_for_recheck')
            elif hashlib.sha256(p.read_bytes()).hexdigest() != branch['parent_sha256']:
                errors.append('parent_hash_mismatch')
    # Compare complete supervision AND actual images, not filenames/action-only sequences.
    fingerprint = hashlib.sha256(json.dumps({'system': record.get('system'),
        'conversations': turns, 'image_hashes': hashes}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return sorted(set(errors)), sorted(set(warnings)), fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--max-actions', type=int, default=30)
    args = p.parse_args()
    if args.max_actions < 1:
        p.error('--max-actions must be positive')
    files = sorted({f.resolve() for entry in args.inputs for f in
                    ([entry] if entry.is_file() else entry.rglob('*.json'))
                    if re.search(r'_success_[01]\.json$', f.name)})
    if not files:
        p.error('No *_success_0.json or *_success_1.json trajectories found (JSONL is not used)')
    rows, seen = [], {}
    for source in files:
        blob = source.read_bytes()
        try:
            record = json.loads(blob)
            errors, warnings, fingerprint = inspect_record(record, source, args.max_actions, args.image_root)
            if not errors and fingerprint in seen:
                errors.append('duplicate_complete_supervision')
            duplicate = seen.get(fingerprint)
            if not errors:
                seen[fingerprint] = str(source)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors, warnings, fingerprint, duplicate = ['invalid_record:' + type(exc).__name__], [], None, None
        rows.append({'source': str(source), 'source_sha256': hashlib.sha256(blob).hexdigest(),
                     'accepted': not errors, 'reasons': errors, 'warnings': warnings,
                     'fingerprint': fingerprint, 'duplicate_of': duplicate})
    # Exclusive creation protects an earlier audit; all source data stays untouched.
    with args.output.open('x') as stream:
        json.dump({'rules_version': 1, 'max_actions': args.max_actions,
                   'accepted': sum(r['accepted'] for r in rows), 'total': len(rows),
                   'visual_review_required': True, 'episodes': rows}, stream, indent=2)
    print(f"Accepted {sum(r['accepted'] for r in rows)}/{len(rows)}; see {args.output}")


if __name__ == '__main__':
    main()
