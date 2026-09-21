"""Translate only the model interface; physical tools remain in robot-base frame."""
import json
import re
import numpy as np
import viser.transforms as vtf


def measured_offset(env):
    base = env.base_wxyz_xyz()
    if not np.allclose(vtf.SO3(wxyz=base[:4]).as_matrix()[:, 2], [0, 0, 1], atol=1e-6):
        raise ValueError('Table translation requires an upright robot base')
    return float(base[6] - env.rob.table_offset[2])


def translate(position, offset):
    p = np.array(position, dtype=float, copy=True)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Position must contain three finite coordinates')
    p[2] = round(float(p[2] + offset), 6)
    return p.tolist()


def physical_arguments(name, kwargs, offset):
    args = dict(kwargs)
    alias = {'grasp': 'target', 'get_position': 'object_name',
             'get_position_and_size': 'object_name'}.get(name)
    if alias and 'target_name' in args:
        if alias in args:
            raise ValueError('Duplicate target arguments')
        args[alias] = args.pop('target_name')
    if name == 'move':
        args['target_position'] = translate(args['target_position'], -offset)
    return args


def model_result(name, raw, offset):
    if name not in ('get_position', 'get_position_and_size', 'align'):
        return raw
    encoded = isinstance(raw, str)
    value = json.loads(raw) if encoded else raw
    if name == 'get_position':
        value = translate(value, offset)
    else:
        value = dict(value)
        value['center'] = translate(value['center'], offset)
    return json.dumps(value) if encoded else value


def gripper_text(ctx, text, offset):
    pos = translate(ctx.arm.position(), offset)
    return re.sub(r'position \[[^\]]+\]',
                  'position [' + ', '.join(f'{x:.3f}' for x in pos) + ']', text, count=1)
