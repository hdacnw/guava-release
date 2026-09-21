"""Canonical table-frame tool schema and validation, derived from actual tools.

Building the schema does not initialize a simulator or execute a tool. Use the
evaluation Python environment, which contains the tool modules' dependencies.
"""
from functools import lru_cache
import inspect
import math
from types import SimpleNamespace


@lru_cache(maxsize=1)
def registry():
    from guava.tools.tools import make_tool_registry
    from guava.tools.coordinate_frame import model_registry
    return model_registry(SimpleNamespace(model_z_offset=0.), make_tool_registry(None))


def tools_schema():
    from guava.tools.schema import build_tools_schema
    return build_tools_schema(registry())


def validate_call(name, arguments):
    if name not in registry() or not isinstance(arguments, dict):
        raise ValueError('Unknown tool or non-object arguments')
    inspect.signature(registry()[name]).bind(**arguments)
    from guava.tools.coordinate_frame import physical_arguments
    physical_arguments(name, arguments, 0.)  # Check move shape/finite values, no translation.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite tool argument')
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        if isinstance(value, list):
            for v in value: finite(v)
    finite(arguments)
    if 'target_name' in arguments and (not isinstance(arguments['target_name'], str) or not arguments['target_name'].strip()):
        raise ValueError('target_name must be a nonempty string')
