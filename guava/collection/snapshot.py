"""Versioned branch snapshots for the native MuJoCo runtime.

Legacy qpos-only checkpoints remain readable for diagnostics, but cannot be used
as history-preserving collection checkpoints. Restore does not step physics.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import random
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def plain(value):
    if value is None or isinstance(value, (str, bool, int, float, np.ndarray, np.number)):
        return True
    if isinstance(value, (tuple, list)):
        return all(plain(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, (str, int)) and plain(v) for k, v in value.items())
    return False


def runtime_objects(env, arm, ctx):
    result = {'env': env, 'rob': env.rob, 'arm': arm, 'ctx': ctx}
    for i, robot in enumerate(env.rob.robots):
        result[f'robot/{i}'] = robot
        for name, gripper in robot.gripper.items():
            result[f'gripper/{i}/{name}'] = gripper
        for name, controller in robot.composite_controller.part_controllers.items():
            key = f'controller/{i}/{name}'
            result[key] = controller
            for attr in ('interpolator', 'interpolator_pos', 'interpolator_ori'):
                obj = getattr(controller, attr, None)
                if obj is not None:
                    result[f'{key}/{attr}'] = obj
    return result


def capture(env, arm, ctx):
    import mujoco
    from guava.tools.tools import _refresh
    # mj_step can leave derived geometry at the preceding integration state.
    # Save pixels and end-effector geometry from the exact state being captured.
    env.rob.sim.forward()
    _refresh(ctx)
    model, data = env.rob.sim.model._model, env.rob.sim.data._data
    signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, signature))
    mujoco.mj_getState(model, data, state, signature)
    # Model arrays include camera extrinsics, intrinsics, lighting, textures,
    # and static body positions not present in qpos.
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    objects = {}
    for key, obj in runtime_objects(env, arm, ctx).items():
        objects[key] = {k: deepcopy(v) for k, v in vars(obj).items()
                        if plain(v) and not k.startswith('last_')}
    return {'version': 2, 'xml': env.rob.sim.model.get_xml(),
            'model_binary': binary.tobytes(), 'signature': signature, 'state': state,
            'camera': asdict(env._cam_cfg), 'objects': objects,
            'np_random': np.random.get_state(), 'py_random': random.getstate(),
            'eef_pos': env.eef_pose_base()[0].copy(),
            'rgb': ctx.last_rgb.copy() if ctx.last_rgb is not None else None}


def restore_snapshot(snapshot, env, arm, ctx):
    import mujoco
    from guava.config import CameraConfig
    from guava.tools.tools import _refresh
    if snapshot.get('version') != 2:
        raise ValueError('A version-2 checkpoint is required for faithful history branches')
    if 'model_binary' in snapshot:
        from robosuite.utils.binding_utils import MjSim
        # Recompiling XML can change hidden collision structures / mesh hulls,
        # even when every exposed numeric model array is restored afterward.
        previous = vars(env.rob).get('_initialize_sim')
        def initialize_binary(xml_string=None):
            binary = snapshot['model_binary']
            compiled = mujoco.MjModel.from_binary_path('snapshot.mjb', assets={'snapshot.mjb': binary})
            env.rob.sim = MjSim(compiled)
            env.rob.sim.forward()
            env.rob.initialize_time(env.rob.control_freq)
        env.rob._initialize_sim = initialize_binary
        try:
            env.rob.reset_from_xml_string(snapshot['xml'])
        finally:
            if previous is None:
                del env.rob._initialize_sim
            else:
                env.rob._initialize_sim = previous
    else:
        env.rob.reset_from_xml_string(snapshot['xml'])
    sim = env.rob.sim
    model, data = sim.model._model, sim.data._data
    for name, saved in snapshot.get('model_arrays', {}).items():
        current = getattr(model, name)
        if current.shape != saved.shape:
            raise ValueError(f'Checkpoint model topology mismatch: {name}')
        if not np.array_equal(current, saved):
            if not current.flags.writeable:
                raise ValueError(f'Cannot restore model field: {name}')
            current[:] = saved
    # All compiled constants were saved too; recomputing them would change a
    # model whose camera/body arrays had been mutated after compilation.
    mujoco.mj_setState(model, data, snapshot['state'], snapshot['signature'])
    sim.forward()
    objects = runtime_objects(env, arm, ctx)
    if objects.keys() != snapshot['objects'].keys():
        raise ValueError('Controller layout differs from checkpoint')
    for key, fields in snapshot['objects'].items():
        for name, value in fields.items():
            setattr(objects[key], name, deepcopy(value))
    env.set_camera(CameraConfig(**snapshot['camera']))
    np.random.set_state(snapshot['np_random'])
    random.setstate(snapshot['py_random'])
    ctx.last_rgb = ctx.last_depth = ctx.last_K = ctx.last_pose_mat = None
    _refresh(ctx)
    if not np.allclose(env.eef_pose_base()[0], snapshot['eef_pos'], atol=1e-9, rtol=0):
        raise ValueError(f'Restored end-effector geometry disagrees with checkpoint: {env.eef_pose_base()[0]} versus {snapshot["eef_pos"]}')
    rgb = snapshot['rgb']
    if rgb is None or not np.array_equal(rgb, ctx.last_rgb):
        raise ValueError('Restored pre-branch image differs from checkpoint; branch held')
    return {'camera': snapshot['camera'], 'prebranch_pixels_exact': True,
            'model_restore': 'binary' if 'model_binary' in snapshot else 'xml_and_arrays',
            'eef_error_m': float(np.max(np.abs(env.eef_pose_base()[0] - snapshot['eef_pos'])))}
