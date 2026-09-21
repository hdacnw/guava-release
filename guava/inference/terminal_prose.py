"""Terminal prose policy for future campaigns; never accepts action payloads."""
import re


def terminal_with_prose(answer):
    # Terminal-only output can contain ordinary prose, not XML/JSON/programs.
    if any(token in answer for token in ('<', '>', '{', '}', '```')):
        return None
    markers = re.findall(r'^\s*Task (complete|failed)\.?\s*$', answer, re.M | re.I)
    if len(markers) != 1:
        return None
    # A terminal declaration must be the final nonempty line, not a quoted
    # example followed by a qualification or a contradictory declaration.
    if not re.search(r'(?:^|\n)\s*Task (complete|failed)\.?\s*\Z', answer, re.I):
        return None
    return markers[0].lower()
