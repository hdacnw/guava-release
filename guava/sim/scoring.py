"""Shared geometry/snapshot helpers; success criteria belong to individual tasks."""
import numpy as np

def geometry_vertices(sim, obj):
    """World-space collision mesh vertices, including compiled mesh offsets.

    Body origins need not lie in an object (the apple is offset by ~12 cm).
    The ordering task's two objects are meshes; fail closed for other shapes.
    """
    points = []
    for name in obj.contact_geoms:
        gid = sim.model.geom_name2id(name)
        if int(sim.model.geom_type[gid]) != 7:  # mjGEOM_MESH
            raise ValueError('Ordering geometry requires a mesh object')
        mid = int(sim.model.geom_dataid[gid])
        start, count = int(sim.model.mesh_vertadr[mid]), int(sim.model.mesh_vertnum[mid])
        vertices = sim.model.mesh_vert[start:start+count]
        rotation = sim.data.geom_xmat[gid].reshape(3, 3)
        points.append(vertices @ rotation.T + sim.data.geom_xpos[gid])
    if not points:
        raise ValueError('Ordering object has no collision geometry')
    return np.concatenate(points)


def geometry_center(vertices):
    return (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2


def _ids(rob, obj):
    return {rob.sim.model.geom_name2id(n) for n in obj.contact_geoms}


def snapshot(rob, names, *, geometry=False):
    sim = rob.sim
    groups = {n: _ids(rob, getattr(rob, n)) for n in names}
    table = {i for i in range(sim.model.ngeom)
             if (sim.model.geom_id2name(i) or '').startswith('table')
             and sim.model.geom_contype[i] != 0}
    groups['table'] = table
    pairs = set()
    for c in sim.data.contact[:sim.data.ncon]:
        if c.dist > .002:
            continue
        for a, aa in groups.items():
            if c.geom1 not in aa:
                continue
            for b, bb in groups.items():
                if a != b and c.geom2 in bb:
                    pairs.add(tuple(sorted((a, b))))
    objects = {}
    for n in names:
        obj = getattr(rob, n)
        bid = sim.model.body_name2id(obj.root_body)
        objects[n] = {
            'pos': sim.data.body_xpos[bid].tolist(),
            'rotation': sim.data.body_xmat[bid].reshape(3, 3).tolist(),
            'held': bool(rob._check_grasp(gripper=rob.robots[0].gripper, object_geoms=obj)),
        }
        if geometry:
            objects[n]['geometry_center'] = geometry_center(geometry_vertices(sim, obj)).tolist()
    camera = getattr(rob, "_scoring_camera", None) or rob.render_camera
    camera = camera[0] if isinstance(camera, (list, tuple)) else camera
    cam = sim.model.camera_name2id(camera)
    right = sim.data.cam_xmat[cam].reshape(3, 3)[:, 0]
    front = sim.data.cam_xmat[cam].reshape(3, 3)[:, 2]
    return {'objects': objects, 'contacts': [list(x) for x in sorted(pairs)],
            'table_xyz': np.asarray(rob.table_offset).tolist(),
            'table_half_xy': (np.array(rob.table_full_size[:2]) / 2).tolist(),
            'camera_right': right.tolist(),
            'camera_front': front.tolist(),
            'camera_position': sim.data.cam_xpos[cam].tolist(),
            'push_direction': getattr(rob, 'current_direction', None),
            'push_initial': (getattr(rob, '_basket_init_pos', None).tolist()
                             if getattr(rob, '_basket_init_pos', None) is not None else None)}


def predicates(state):
    o = state['objects']
    contacts = {tuple(sorted(x)) for x in state['contacts']}
    table = np.array(state['table_xyz'])
    def touch(a, b):
        return tuple(sorted((a, b))) in contacts
    def released(n):
        return not o[n]['held']
    def supported(n):
        p = np.array(o[n]['pos'])
        return bool(touch(n, 'table') and
                    np.all(np.abs(p[:2] - table[:2]) < np.array(state['table_half_xy'])) and
                    p[2] > table[2] - .025)
    def upright(n):
        return float(np.array(o[n]['rotation'])[2, 2]) > np.cos(np.deg2rad(30))
    def inside(n, container, half_xy, low_z, high_z):
        p = np.array(o[n]['pos']) - np.array(o[container]['pos'])
        local = np.array(o[container]['rotation']).T @ p
        return bool(np.all(np.abs(local[:2]) < half_xy) and low_z < local[2] < high_z)
    from types import SimpleNamespace
    return SimpleNamespace(touch=touch, released=released, supported=supported,
                           upright=upright, inside=inside)


def result(checks, metrics=None):
    return {'version': 'task-owned-v1', 'success': bool(all(checks.values())),
            'checks': {k: bool(v) for k, v in checks.items()}, 'metrics': metrics or {}}


def shaped_reward(task, action=None):
    """Dense guidance cannot count as success when the authoritative scorer fails."""
    if task.score()['success']:
        return 1.0
    return min(float(task._shaped_reward(action)), float(np.nextafter(1.0, 0.0)))
