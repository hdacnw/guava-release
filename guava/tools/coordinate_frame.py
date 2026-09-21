"""Model-facing frame adapter; internal skills always use robot-base coordinates."""
import functools
import inspect
import json
import numpy as np

CONVENTION = ('Coordinate convention: all object and gripper positions and move(target_position) '
              'commands use a table-aligned frame in metres. The x/y axes match the robot-base '
              'frame; z=0 is the tabletop and positive z points upward. Dimensions, relative '
              'displacements, gripper widths, and orientations are unchanged. Account for '
              'TCP/finger and held-object geometry when selecting clearance.')
ALIASES = {'grasp': 'target', 'get_position': 'object_name', 'get_position_and_size': 'object_name'}


def measured_offset(env):
    """Simulator calibration, not a guessed or task-global constant."""
    base = env.base_wxyz_xyz()
    import viser.transforms as vtf
    R = vtf.SO3(wxyz=base[:4]).as_matrix()
    if not np.allclose(R[:, 2], [0, 0, 1], atol=1e-6):
        raise ValueError('A z-only table translation requires an upright base frame.')
    return float(base[6] - env.rob.table_offset[2])


def physical_arguments(name, args, offset):
    args = dict(args)
    alias = ALIASES.get(name)
    if alias and 'target_name' in args:
        if alias in args:
            raise ValueError(f'Cannot supply both target_name and {alias}')
        args[alias] = args.pop('target_name')
    if name == 'move':
        p = np.array(args['target_position'], dtype=float, copy=True)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('target_position must contain three finite coordinates')
        p[2] = round(float(p[2] - offset), 6)
        args['target_position'] = p.tolist()
    return args


def model_result(name, result, offset):
    encoded = isinstance(result, str)
    if name not in ('align', 'get_position', 'get_position_and_size'):
        return result
    value = json.loads(result) if encoded else result
    if name == 'get_position':
        value = list(value); value[2] = round(value[2] + offset, 6)
    else:
        value = dict(value); value['center'] = list(value['center'])
        value['center'][2] = round(value['center'][2] + offset, 6)
    return json.dumps(value) if encoded else value


def model_registry(ctx, registry):
    result = {}
    for name, fn in registry.items():
        def wrap(fn, name):
            underlying = fn.func if isinstance(fn, functools.partial) else fn
            @functools.wraps(underlying)
            def call(**kwargs):
                # The public schema is authoritative: reject legacy aliases.
                call.__signature__.bind(**kwargs)
                args = physical_arguments(name, kwargs, ctx.model_z_offset)
                return model_result(name, fn(**args), ctx.model_z_offset)
            alias = ALIASES.get(name)
            call.__signature__ = inspect.Signature([
                p.replace(name='target_name') if p.name == alias else p
                for p in inspect.signature(fn).parameters.values()])
            call.__annotations__ = {('target_name' if k == alias else k): v
                                     for k, v in underlying.__annotations__.items() if k != 'ctx'}
            call.__doc__ = (underlying.__doc__ or '').replace('robot base frame', 'model coordinate frame')
            return call
        result[name] = wrap(fn, name)
    return result
