import inspect
import functools
from typing import get_type_hints


_PY_TO_JSON = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _unwrap(fn):
    """Return (underlying_function, bound_param_names) for plain functions and partials."""
    if isinstance(fn, functools.partial):
        bound = set(fn.keywords) | {
            k for k, _ in zip(inspect.signature(fn.func).parameters, fn.args)
        }
        return fn.func, bound
    return fn, set()


def build_schema(fn) -> dict:
    """Build an OpenAI function-calling schema for fn from its signature + docstring."""
    underlying, bound = _unwrap(fn)
    hints = get_type_hints(underlying)
    sig   = inspect.signature(underlying)
    doc   = (underlying.__doc__ or "").strip().split("\n")[0]

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if name in bound or name == "ctx":
            continue
        annotation = hints.get(name, str)
        if annotation is list or getattr(annotation, "__origin__", None) is list:
            prop = {"type": "array", "items": {"type": "number"}}
        else:
            prop = _PY_TO_JSON.get(annotation, {"type": "string"}).copy()
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": underlying.__name__,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_tools_schema(tool_fns: dict) -> list[dict]:
    return [build_schema(fn) for fn in tool_fns.values()]


def tools_to_prompt(tool_fns: dict) -> str:
    lines = ["Available tools:"]
    for fn in tool_fns.values():
        underlying, bound = _unwrap(fn)
        hints = get_type_hints(underlying)
        params = [
            f"{p}: {a.__name__ if hasattr(a, '__name__') else str(a)}"
            for p, a in hints.items()
            if p not in bound and p != "ctx" and p != "return"
        ]
        doc = (underlying.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  {underlying.__name__}({', '.join(params)}) — {doc}")
    return "\n".join(lines)
