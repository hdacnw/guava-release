"""Provenance-aware handling of checkpoint rationale and supervision."""
from __future__ import annotations

import json
import re


def checkpoint_reasoning(record: dict | None, index: int, name: str, args: dict) -> tuple[str, bool, str]:
    """Reuse only an unchanged parent decision; never rationalize an intervention.

    The caller supplies the exact parent/checkpoint boundary used for replay.
    Preserve any existing context-only label in the parent as well.
    """
    turns = (record or {}).get('conversations', [])
    if index < 0 or index >= len(turns) or turns[index].get('from') != 'gpt':
        return '', False, 'checkpoint_parent_missing'
    parent = turns[index]
    text = parent.get('value', '')
    calls = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if len(calls) != 1:
        return '', False, 'checkpoint_parent_action_unverified'
    try:
        call = json.loads(calls[0])
    except (ValueError, TypeError):
        return '', False, 'checkpoint_parent_action_unverified'
    if call != {'name': name, 'arguments': args}:
        return '', False, 'checkpoint_action_modified'
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    rationale = match.group(1).strip() if match else ''
    if not rationale:
        return '', False, 'checkpoint_parent_reasoning_missing'
    if parent.get('loss') is False:
        return rationale, False, 'checkpoint_parent_context_only'
    return rationale, True, 'checkpoint_parent_restored'
