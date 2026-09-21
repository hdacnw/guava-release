"""Opt-in benchmark first-call extraction with discarded-output auditing.

Run as ``python -m guava.inference.benchmark_compatible_tasks`` using the
randomized benchmark CLI. Pass --modes full and an explicit
--system-prompt (the benchmark uses guava/collection/system_prompt.txt).
This intentionally tolerates trailing JSON data and selects the first tool;
it does not change the default strict evaluator. Run wrappers in separate
processes.
"""
import json
import re
from guava.inference import benchmark_engine as tm
from guava.llm.client import _parse_tool_from_text
from guava.tools.schema import tools_to_prompt

strict_decision = tm.decision
original_episode = tm.run_episode
audit = []
tool_text = ''

def compatible(content, mode):
    assert mode == 'full'
    record = {'response_index': len(audit), 'raw_content': content}
    try:
        strict_decision(content, mode)
        record['strict_valid'] = True
    except ValueError as exc:
        record.update(strict_valid=False, strict_error=str(exc))
    name, args = _parse_tool_from_text(content)
    match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.S)
    record['tool_call_count'] = content.count('<tool_call>')
    if match:
        record['ignored_after_first_call'] = content[match.end():]
        try:
            body = match[1].strip()
            _, end = json.JSONDecoder().raw_decode(body)
            record['ignored_json_suffix'] = body[end:]
        except ValueError:
            record['json_prefix_decoded'] = False
    record['selected'] = {'name': name, 'arguments': args} if name else None
    audit.append(record)
    if name:
        return 'tool', {'name': name, 'arguments': args}
    # Retain explicit completion declarations; do not treat arbitrary prose as done.
    return strict_decision(content, mode)

def observation(mode, image_url, gripper, result='', initial=False,
                task_name='', object_names=()):
    text = f'Task: {task_name}\n\nAvailable tools:\n{tool_text}\n\n{gripper}' if initial else f'{result}\n{gripper}'
    return [{'type': 'image_url', 'image_url': {'url': image_url}},
            {'type': 'text', 'text': text}]

def episode(engine, client, mode, index, root, seed):
    global audit, tool_text
    audit = []
    if getattr(engine, 'table_frame', False):
        from guava.tools.coordinate_frame import model_registry
        # Schema only: invocation still uses the engine's physical registry and
        # sft_frame adapter. Do not execute this wrapped registry a second time.
        tool_text = tools_to_prompt(model_registry(engine.ctx, engine.registry))
    else:
        tool_text = tools_to_prompt(engine.registry)
    try:
        return original_episode(engine, client, mode, index, root, seed)
    finally:
        folder = root / mode / f'episode_{index:03d}'
        if folder.exists():
            tm.write(folder / 'parser_audit.json', audit)

if __name__ == '__main__':
    tm.decision = compatible
    tm.observation = observation
    tm.run_episode = episode
    tm.main()
