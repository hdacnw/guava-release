

from __future__ import annotations
import base64, io, json
from dataclasses import dataclass
from typing import Any
import numpy as np
from PIL import Image

from guava.sim.env import RoboEnv
from guava.robot.arm import RobotArm


@dataclass
class ToolContext:
    env: RoboEnv
    arm: RobotArm
    perception: Any   # SAM3Solver | GTSolver
    grasp_planner: Any = None  # GraspGenClient | None
    bbox_debug_dir: Any = None  # Path | None
    graspgen_debug_dir: Any = None  # Path | None
    last_rgb: np.ndarray | None = None
    last_depth: np.ndarray | None = None
    last_K: np.ndarray | None = None
    last_pose_mat: np.ndarray | None = None
    grasp_displace_to: list[float] | None = None   # set by branch runner for grasp failure
    grasp_displace_delta: list[float] | None = None
    grasp_perturbation_applied: bool = False
    graspgen_horizontal: bool = False  # favour grasps whose closing axis is horizontal (knobs)
    graspgen_vertical: bool = False    # favour top-down grasps (approach Z points down in base)
    # True iff the most recent close_gripper / grasp(target) reported "grasped".
    # release() consults this to decide whether to auto-retract along -approach
    # afterwards (so the gripper isn't left at the drop pose for the next align).
    was_holding: bool = False
    # Diagnostic only; model-visible feedback merges blocked into grasped.
    last_grasp_feedback_raw: str | None = None
    model_z_offset: float = 0.0


def _encode(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _refresh(ctx: ToolContext) -> None:
    """Pull latest camera obs into ctx with a single render call for RGB + depth."""
    # mj_step integrates qpos after computing body/camera poses. Synchronize
    # derived poses before recording an observation or a branch checkpoint.
    ctx.env.rob.sim.forward()
    ctx.last_rgb, ctx.last_depth, ctx.last_K, ctx.last_pose_mat = _capture_with_depth(ctx.env)


def _capture_with_depth(env):
    # Capture RGB + metric depth using robosuite's observation and camera utilities.
    # Uses the robosuite obs pipeline (camera_depths=True) to avoid GL state issues.
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_extrinsic_matrix,
        get_real_depth_map,
    )

    sim     = env.rob.sim
    cam_cfg = env._cam_cfg or env.cfg.camera

    # ── RGB + depth from robosuite's obs pipeline ─────────────────────── #
    obs     = env.rob._get_observations(force_update=True)
    rgb_key = f"{cam_cfg.name}_image"
    dep_key = f"{cam_cfg.name}_depth"
    if rgb_key not in obs or dep_key not in obs:
        raise RuntimeError(
            f"Camera '{cam_cfg.name}' not found in observations. "
            f"Ensure it is in camera_names at environment creation."
        )

    # robosuite uses IMAGE_CONVENTION="opengl" (convention=1, no flip), so images
    # come out bottom-up (OpenGL row order). Flip both to image convention (top-down).
    rgb     = np.asarray(obs[rgb_key], dtype=np.uint8)[..., :3][::-1].copy()
    dep_raw = np.asarray(obs[dep_key], dtype=np.float64)
    if dep_raw.ndim == 3:
        dep_raw = dep_raw[..., 0]
    dep_raw = np.clip(np.nan_to_num(dep_raw[::-1].copy(), nan=1.0), 0.0, 1.0)  # flip, NaN→far, guard range

    depth_m = get_real_depth_map(sim, dep_raw).astype(np.float32)  # metric metres

    # ── Camera intrinsics ─────────────────────────────────────────────── #
    K = get_camera_intrinsic_matrix(sim, cam_cfg.name, cam_cfg.height, cam_cfg.width)

    # ── Camera extrinsics: cam-to-world, then world-to-base ───────────── #
    T_cam_to_world = get_camera_extrinsic_matrix(sim, cam_cfg.name)   # 4x4, OpenCV-cam→world

    base_id          = sim.model.body_name2id("fixed_mount0_base")
    T_base_to_world  = np.eye(4, dtype=np.float64)
    T_base_to_world[:3, :3] = sim.data.xmat[base_id].reshape(3, 3).copy()
    T_base_to_world[:3,  3] = sim.data.xpos[base_id].copy()
    T_world_to_base  = np.linalg.inv(T_base_to_world)

    pose_mat = (T_world_to_base @ T_cam_to_world).astype(np.float64)  # OpenCV-cam→base

    return rgb, depth_m, K, pose_mat


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _unproject_pointclouds(
    depth: np.ndarray, K: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pc_full N×3, pc_segment M×3) in camera frame."""
    d = depth[..., 0] if depth.ndim == 3 else depth
    h, w = d.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = d > 0
    z  = d[valid].astype(np.float32)
    x  = ((us[valid] - K[0, 2]) * z / K[0, 0]).astype(np.float32)
    y  = ((vs[valid] - K[1, 2]) * z / K[1, 1]).astype(np.float32)
    pc_full = np.stack([x, y, z], axis=1)

    seg = valid & mask
    zs = d[seg].astype(np.float32)
    xs = ((us[seg] - K[0, 2]) * zs / K[0, 0]).astype(np.float32)
    ys = ((vs[seg] - K[1, 2]) * zs / K[1, 1]).astype(np.float32)
    pc_segment = np.stack([xs, ys, zs], axis=1)

    return pc_full, pc_segment


def _voxel_downsample(pc: np.ndarray, voxel_size: float) -> np.ndarray:
    """Retain one centroid per voxel cell."""
    if len(pc) == 0:
        return pc
    keys = np.floor(pc / voxel_size).astype(np.int32)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((inv.max() + 1, pc.shape[1]), dtype=np.float64)
    counts = np.zeros(inv.max() + 1, dtype=np.int32)
    np.add.at(out, inv, pc)
    np.add.at(counts, inv, 1)
    return (out / counts[:, None]).astype(pc.dtype)


def _remove_statistical_outliers(pc: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Remove points whose mean k-NN distance exceeds mean + std_ratio * std."""
    if len(pc) <= k:
        return pc
    from scipy.spatial import cKDTree
    dists, _ = cKDTree(pc).query(pc, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return pc[mean_dists <= threshold]


# ------------------------------------------------------------------
# Align / gripper-state helpers
# ------------------------------------------------------------------

def _rot_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to wxyz quaternion (Shepperd's method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


_CLEARANCE_M = {"small": 0.05, "medium": 0.08, "large": 0.15}
_SIDE_POSITIONS  = {"left", "right", "front", "back"}
_ALL_POSITIONS   = _SIDE_POSITIONS | {"top"}

# Direction from gripper toward object in camera frame (X=right, Y=down, Z=depth)
# for each side position.  Used to orient the gripper's wide face toward the object.
_CAM_APPROACH_DIR = {
    "left":  np.array([1.0,  0.0, 0.0]),
    "right": np.array([-1.0, 0.0, 0.0]),
    "front": np.array([0.0,  0.0, 1.0]),
    "back":  np.array([0.0,  0.0, -1.0]),
}
# Measured offset: yaw angle when wide face points along robot-base +x axis.
_WIDE_FACE_YAW_OFFSET = -np.pi / 2   # -90°


def _compute_align_target(
    pc_segment: np.ndarray,
    position: str,
    clearance_m: float,
    pose_mat: np.ndarray,
    current_gripper_z: float,
) -> np.ndarray:
    """Return robot-base-frame target position for align().

    AABB is computed in camera frame (X=right, Y=down, Z=depth), so
    left/right/top/front/back map directly to image-space directions.
    For side positions the Z component is overridden with the current
    gripper height so only XY moves.
    """
    mins     = pc_segment.min(axis=0)   # [min_x, min_y, min_z]
    maxs     = pc_segment.max(axis=0)
    mean_cam = pc_segment.mean(axis=0)
    c        = clearance_m

    offsets = {
        "left":  np.array([mins[0] - c,   mean_cam[1], mean_cam[2], 1.0]),
        "right": np.array([maxs[0] + c,   mean_cam[1], mean_cam[2], 1.0]),
        # For top: estimate object radius from camera-frame X extent and offset from
        # the front surface.  AABB centre is biased ~R/2 toward camera;
        # mins[2] + R_est ≈ true centre depth for any convex round object.
        "top":   np.array([mean_cam[0],   mins[1] - c, mins[2] + (maxs[0] - mins[0]) / 2.0 * 0.75, 1.0]),
        "front": np.array([mean_cam[0],   mean_cam[1], mins[2] - c, 1.0]),  # min_z = closer to cam
        "back":  np.array([mean_cam[0],   mean_cam[1], maxs[2] + c, 1.0]),
    }
    p_base = pose_mat @ offsets[position]   # cam → robot base frame

    target = p_base[:3].copy()
    if position in _SIDE_POSITIONS:         # preserve current height for lateral moves
        target[2] = current_gripper_z
    return target


def _gripper_state_text(ctx: "ToolContext") -> str:
    """One-line summary of current gripper pose and width for LLM context."""
    pos = list(ctx.arm.position())
    pos[2] += ctx.model_z_offset
    w, x, y, z = ctx.arm.rotation()
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    width_pct = int(round(ctx.arm.gripper_width() / 0.08 * 100))
    return (
        f"Gripper is currently at position [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
        f"with a rotation of [roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°] "
        f"and {width_pct}% of maximum width."
    )


# ------------------------------------------------------------------
# Tool functions
# ------------------------------------------------------------------


def get_position(ctx: ToolContext, object_name: str) -> list[float]:
    """Get the 3-D position [x,y,z] of a named object in robot base frame."""
    if ctx.last_rgb is None:
        _refresh(ctx)
    mask = ctx.perception.get_mask(object_name, ctx.last_rgb)
    _, pc_segment = _unproject_pointclouds(ctx.last_depth, ctx.last_K, mask)
    if pc_segment.shape[0] < 10:
        raise RuntimeError(
            f"Too few points for '{object_name}' ({pc_segment.shape[0]}) — object may be occluded"
        )
    pc_segment = _remove_statistical_outliers(pc_segment, k=20, std_ratio=2.0)
    if pc_segment.shape[0] < 5:
        raise RuntimeError(
            f"Too few points for '{object_name}' after filtering — object may be occluded"
        )
    N = pc_segment.shape[0]
    h = np.hstack([pc_segment, np.ones((N, 1))])
    pc_base = (ctx.last_pose_mat @ h.T).T[:, :3]
    return [round(float(v), 3) for v in pc_base.mean(axis=0)]


def get_position_and_size(ctx: ToolContext, object_name: str) -> str:
    """Get position and bounding-box size of a named object in robot base frame.

    Returns JSON with 'center' [x,y,z] and 'extents' [dx,dy,dz].
    """
    if ctx.last_rgb is None:
        _refresh(ctx)
    mask = ctx.perception.get_mask(object_name, ctx.last_rgb)
    _, pc_segment = _unproject_pointclouds(ctx.last_depth, ctx.last_K, mask)
    if pc_segment.shape[0] < 10:
        raise RuntimeError(
            f"Too few points for '{object_name}' ({pc_segment.shape[0]}) — object may be occluded"
        )
    pc_segment = _remove_statistical_outliers(pc_segment, k=20, std_ratio=2.0)
    if pc_segment.shape[0] < 5:
        raise RuntimeError(
            f"Too few points for '{object_name}' after filtering — object may be occluded"
        )
    N = pc_segment.shape[0]
    h = np.hstack([pc_segment, np.ones((N, 1))])
    pc_base = (ctx.last_pose_mat @ h.T).T[:, :3]
    center  = [round(float(v), 3) for v in pc_base.mean(axis=0)]
    extents = [round(float(v), 3) for v in (pc_base.max(axis=0) - pc_base.min(axis=0))]
    return json.dumps({"center": center, "extents": extents})


def get_gripper_position(ctx: ToolContext) -> list[float]:
    """Return current gripper TCP position [x,y,z] in robot base frame."""
    return ctx.arm.position()


def get_gripper_rotation(ctx: ToolContext) -> list[float]:
    """Return gripper orientation as [roll, pitch, yaw] in degrees (XYZ intrinsic, base frame). For a straight-down top-down grasp: roll ≈ ±180°, pitch ≈ 0°. Yaw can be ignored."""
    import numpy as np
    w, x, y, z = ctx.arm.rotation()
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return [round(roll, 1), round(pitch, 1), round(yaw, 1)]


def get_gripper_width(ctx: ToolContext) -> float:
    """Return gripper opening width in metres (~0.08 = open, ~0 = closed)."""
    return ctx.arm.gripper_width()


def home_pose(ctx: ToolContext) -> str:
    """Move the gripper back to the robot's initial home configuration (position and orientation)."""
    return ctx.arm.home_pose()


def move(ctx: ToolContext, target_position: list[float]) -> str:
    """Move gripper TCP to [x,y,z] target (robot base frame), keeping orientation."""
    return ctx.arm.move(target_position)


def close_gripper(ctx: ToolContext) -> str:
    """Close gripper until contact or fully closed. Returns 'grasped' or 'closed'."""
    result = ctx.arm.grasp()
    return _grasp_feedback(ctx, result)


# ------------------------------------------------------------------
# Bounding-box debug visualisation
# ------------------------------------------------------------------

_bbox_call_count: dict[str, int] = {}


def _get_base_to_world(env) -> np.ndarray:
    """Return the 4×4 transform from robot base frame to world (sim) frame."""
    sim = env.rob.sim
    base_id = sim.model.body_name2id("fixed_mount0_base")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = sim.data.xmat[base_id].reshape(3, 3).copy()
    T[:3,  3] = sim.data.xpos[base_id].copy()
    return T


def _write_ply(path: Any, pts: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (px, py, pz), (r, g, b) in zip(pts, colors):
            f.write(f"{px:.6f} {py:.6f} {pz:.6f} {int(r)} {int(g)} {int(b)}\n")


def _save_bbox_debug(
    pc_segment: np.ndarray,
    pose_mat: np.ndarray,
    mins_cam: np.ndarray,
    maxs_cam: np.ndarray,
    target_base: np.ndarray,
    label: str,
    debug_dir: Any,
    *,
    pca_axes_base: tuple[np.ndarray, np.ndarray] | None = None,
    grasp_frame_base: tuple[np.ndarray, np.ndarray] | None = None,
    base_to_world: np.ndarray | None = None,
) -> None:
    """Save segmented pointcloud + AABB wireframe + target as a single coloured .ply.

    Optional overlays (all in base frame):
      pca_axes_base    — (seg_primary, seg_secondary), each shape (2, 3): endpoints
                         for the primary (cyan) and secondary (magenta) PCA axes.
      grasp_frame_base — (pos (3,), quat_wxyz (4,)): RGB triad at pos —
                         X=red (jaw closing for Panda), Y=green, Z=blue (approach/down).
      base_to_world    — 4×4 transform; if given, also writes a sibling
                         {key}_{idx:03d}_world.ply with the same geometry expressed
                         in world (sim) frame. Adds a small RGB triad at world origin.
    """
    import re
    from pathlib import Path as _Path

    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    idx = _bbox_call_count.get(key, 0)
    _bbox_call_count[key] = idx + 1
    out_dir = _Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pointcloud in base frame (grey)
    N  = pc_segment.shape[0]
    h  = np.hstack([pc_segment, np.ones((N, 1))])
    pc_base = (pose_mat @ h.T).T[:, :3]

    # AABB wireframe: 8 corners → 12 edges densely sampled (red)
    cx0, cy0, cz0 = mins_cam
    cx1, cy1, cz1 = maxs_cam
    corners = np.array([
        [cx0, cy0, cz0], [cx1, cy0, cz0], [cx1, cy1, cz0], [cx0, cy1, cz0],
        [cx0, cy0, cz1], [cx1, cy0, cz1], [cx1, cy1, cz1], [cx0, cy1, cz1],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    edge_cam = np.vstack([
        corners[a] + t * (corners[b] - corners[a])
        for a, b in edges
        for t in np.linspace(0, 1, 25)
    ])
    he = np.hstack([edge_cam, np.ones((len(edge_cam), 1))])
    edge_base = (pose_mat @ he.T).T[:, :3]

    pts_list = [pc_base, edge_base, target_base[None]]
    colors_list = [
        np.full((len(pc_base),   3), [180, 180, 180], dtype=np.uint8),
        np.full((len(edge_base), 3), [255,  50,  50], dtype=np.uint8),
        np.array([[0, 255, 0]], dtype=np.uint8),
    ]

    def _line(p0: np.ndarray, p1: np.ndarray, n: int = 50) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        return p0 + ts * (p1 - p0)

    if pca_axes_base is not None:
        seg0, seg1 = pca_axes_base
        line0 = _line(np.asarray(seg0[0], dtype=np.float64),
                      np.asarray(seg0[1], dtype=np.float64))
        line1 = _line(np.asarray(seg1[0], dtype=np.float64),
                      np.asarray(seg1[1], dtype=np.float64))
        pts_list   += [line0, line1]
        colors_list += [
            np.full((len(line0), 3), [  0, 200, 255], dtype=np.uint8),
            np.full((len(line1), 3), [255,   0, 200], dtype=np.uint8),
        ]

    if grasp_frame_base is not None:
        pos, quat = grasp_frame_base
        pos = np.asarray(pos, dtype=np.float64)
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        L = 0.04
        ax_x = _line(pos, pos + L * R[:, 0])
        ax_y = _line(pos, pos + L * R[:, 1])
        ax_z = _line(pos, pos + L * R[:, 2])
        pts_list   += [ax_x, ax_y, ax_z]
        colors_list += [
            np.full((len(ax_x), 3), [255,   0,   0], dtype=np.uint8),
            np.full((len(ax_y), 3), [  0, 255,   0], dtype=np.uint8),
            np.full((len(ax_z), 3), [  0,   0, 255], dtype=np.uint8),
        ]

    pts = np.vstack(pts_list)
    colors = np.vstack(colors_list)

    base_path = out_dir / f"{key}_{idx:03d}.ply"
    _write_ply(base_path, pts, colors)

    if base_to_world is not None:
        h_pts = np.hstack([pts, np.ones((len(pts), 1))])
        pts_world = (base_to_world @ h_pts.T).T[:, :3]

        # Small RGB triad at world origin so the world frame is visible in viewers.
        Lw = 0.10
        ox = _line(np.zeros(3), np.array([Lw, 0.0, 0.0]))
        oy = _line(np.zeros(3), np.array([0.0, Lw, 0.0]))
        oz = _line(np.zeros(3), np.array([0.0, 0.0, Lw]))
        world_extras_pts = np.vstack([ox, oy, oz])
        world_extras_colors = np.vstack([
            np.full((len(ox), 3), [255,   0,   0], dtype=np.uint8),
            np.full((len(oy), 3), [  0, 255,   0], dtype=np.uint8),
            np.full((len(oz), 3), [  0,   0, 255], dtype=np.uint8),
        ])
        pts_world = np.vstack([pts_world, world_extras_pts])
        colors_world = np.vstack([colors, world_extras_colors])

        world_path = out_dir / f"{key}_{idx:03d}_world.ply"
        _write_ply(world_path, pts_world, colors_world)


_graspgen_call_count: dict[str, int] = {}


def _save_graspgen_debug(
    pc_segment: np.ndarray,  # (M, 3) in camera frame — object points
    grasps: np.ndarray,       # (N, 4, 4) in camera frame
    scores: np.ndarray,       # (N,)
    best_idx: int,
    pose_mat: np.ndarray,    # cam → base (4×4)
    label: str,
    debug_dir: Any,
    pc_scene: np.ndarray | None = None,  # (S, 3) in camera frame — scene (non-object) points
) -> None:
    """Save GraspGen debug PLY: object PC + scene PC + all grasp approach axes + selected grasp triad."""
    import re
    from pathlib import Path as _Path

    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    idx = _graspgen_call_count.get(key, 0)
    _graspgen_call_count[key] = idx + 1
    out_dir = _Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _line(p0: np.ndarray, p1: np.ndarray, n: int = 30) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        return p0 + ts * (p1 - p0)

    pts_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []

    # Scene point cloud in base frame (dark grey), downsampled for file size
    if pc_scene is not None and len(pc_scene) > 0:
        if len(pc_scene) > 32768:
            idx_s = np.random.choice(len(pc_scene), 32768, replace=False)
            pc_scene_ds = pc_scene[idx_s]
        else:
            pc_scene_ds = pc_scene
        S = len(pc_scene_ds)
        h_s = np.hstack([pc_scene_ds, np.ones((S, 1))])
        pc_scene_base = (pose_mat @ h_s.T).T[:, :3]
        pts_list.append(pc_scene_base)
        colors_list.append(np.full((S, 3), [80, 80, 80], dtype=np.uint8))

    # Object point cloud in base frame (light grey)
    M = len(pc_segment)
    h = np.hstack([pc_segment, np.ones((M, 1))])
    pc_base = (pose_mat @ h.T).T[:, :3]
    pts_list.append(pc_base)
    colors_list.append(np.full((M, 3), [200, 200, 200], dtype=np.uint8))

    # All grasps: Z approach axis, colored by score (blue=low → red=high)
    R_pose = pose_mat[:3, :3]
    t_pose = pose_mat[:3, 3]
    s_min, s_max = scores.min(), scores.max()
    s_range = float(s_max - s_min) if s_max > s_min else 1.0
    L_all = 0.025
    for T_g, score in zip(grasps, scores):
        t_s = float((score - s_min) / s_range)
        col = np.array([[int(255 * t_s), int(80 * t_s), int(255 * (1 - t_s))]], dtype=np.uint8)
        pos_base = R_pose @ T_g[:3, 3] + t_pose
        z_base   = R_pose @ T_g[:3, 2]
        seg = _line(pos_base, pos_base + L_all * z_base)
        pts_list.append(seg)
        colors_list.append(np.tile(col, (len(seg), 1)))

    # Selected grasp: full RGB XYZ triad (larger), X=red Y=green Z=blue
    T_best = grasps[best_idx]
    pos_best = R_pose @ T_best[:3, 3] + t_pose
    L_best = 0.05
    for axis_i, col in enumerate([[255, 0, 0], [0, 255, 0], [0, 0, 255]]):
        ax_base = R_pose @ T_best[:3, axis_i]
        seg = _line(pos_best, pos_best + L_best * ax_base)
        pts_list.append(seg)
        colors_list.append(np.full((len(seg), 3), col, dtype=np.uint8))

    # XYZ reference axes at point cloud centroid (dim RGB)
    centroid = pc_base.mean(axis=0)
    L_ref = 0.04
    for axis_i, col in enumerate([[200, 80, 80], [80, 200, 80], [80, 80, 200]]):
        ax = np.zeros(3); ax[axis_i] = 1.0
        seg = _line(centroid, centroid + L_ref * ax)
        pts_list.append(seg)
        colors_list.append(np.full((len(seg), 3), col, dtype=np.uint8))

    pts    = np.vstack(pts_list)
    colors = np.vstack(colors_list)
    out_path = out_dir / f"{key}_{idx:03d}.ply"
    _write_ply(out_path, pts, colors)
    print(f"[graspgen debug] {out_path} ({len(grasps)} grasps, best={best_idx})")


_GRASP_APPROACH_CLEARANCE: float = 0.1 # metres above object top before descending


def _grasp_feedback(ctx, result):
    # A stalled gripper can be touching the table, not holding an object.
    # In simulator-GT mode bilateral object contact is directly observable.
    if ctx.env.cfg.perception == 'gt' and result == 'grasped':
        from guava.collection.checkpoint import _find_grasped_object
        ctx.env.rob.sim.forward()
        if _find_grasped_object(ctx.env, list(ctx.env.cfg.object_aliases.values())) is None:
            result = 'blocked'
    ctx.last_grasp_feedback_raw = result
    # User-selected API contract: obstruction is treated as a possible grasp.
    # Keep the raw contact verdict for diagnostics/branch-generation checks.
    if result == 'blocked':
        result = 'grasped'
    ctx.was_holding = result == 'grasped'
    return result


def _branch_grasp_target(ctx, intended):
    """Offset the descent, not a lateral sweep after touching the object."""
    goal = np.array(intended, dtype=float, copy=True)
    changed = ctx.grasp_displace_to is not None or ctx.grasp_displace_delta is not None
    if ctx.grasp_displace_to is not None:
        goal = np.asarray(ctx.grasp_displace_to, dtype=float)
    elif ctx.grasp_displace_delta is not None:
        goal += np.asarray(ctx.grasp_displace_delta, dtype=float)
    ctx.grasp_displace_to = ctx.grasp_displace_delta = None
    return goal, changed


def grasp(ctx: ToolContext, target: str) -> str:
    """Segment target with SAM3, plan a grasp, descend and close gripper.

    Uses GraspGen if available (6-DOF grasps for any orientation), otherwise
    falls back to PCA-based top-down grasp. Call align(target, 'top', 'medium')
    first when using PCA mode. Returns 'grasped' or 'closed'.
    """
    if not hasattr(ctx.perception, "get_mask"):
        raise RuntimeError("grasp(target) requires SAM3 perception backend (not GT)")

    _refresh(ctx)
    mask = ctx.perception.get_mask(target, ctx.last_rgb)

    # ── GraspGen path ───────────────────────────────────────────────────── #
    if ctx.grasp_planner is not None:
        segmap = mask.astype(np.int32)  # True → 1 (target), False → 0
        grasps, scores = ctx.grasp_planner.plan_grasps(
            ctx.last_depth, ctx.last_K, segmap, segmap_id=1
        )
        if len(grasps) == 0 or grasps.ndim < 3:
            raise RuntimeError(f"GraspGen found no valid grasps for '{target}'")

        if ctx.graspgen_horizontal:
            # Bias toward "flat grasps" while keeping GraspGen's confidence in
            # the mix: multiplier is |Yz| (∈ [0, 1]), 1 when both closing and
            # approach lie in the horizontal plane (gripper Y axis vertical).
            R_base_cam = ctx.last_pose_mat[:3, :3]
            y_world_z = (R_base_cam @ grasps[:, :3, 1].T)[2]  # (N,)
            scores = scores * np.abs(y_world_z)

        if ctx.graspgen_vertical:
            # Favour top-down grasps: approach axis (GraspGen Z col) should point downward
            # in base frame, so the back of the gripper has +z. Multiplier ∈ [0, 1].
            R_base_cam = ctx.last_pose_mat[:3, :3]
            approach_base_z = (R_base_cam @ grasps[:, :3, 2].T)[2]  # (N,)
            scores = scores * np.clip(-approach_base_z, 0.0, 1.0)

        best_idx = int(np.argmax(scores))
        T_cam = grasps[best_idx]  # 4×4 in camera frame

        if ctx.graspgen_horizontal or ctx.graspgen_vertical:
            R_base_cam = ctx.last_pose_mat[:3, :3]
            closing_world  = R_base_cam @ T_cam[:3, 0]
            up_world       = R_base_cam @ T_cam[:3, 1]
            approach_world = R_base_cam @ T_cam[:3, 2]
            tag = []
            if ctx.graspgen_horizontal: tag.append("horizontal")
            if ctx.graspgen_vertical:   tag.append("vertical")
            print(f"[graspgen_{'+'.join(tag)}] selected #{best_idx} of {len(grasps)}")
            print(f"  closing  (X): [{closing_world[0]:+.3f} {closing_world[1]:+.3f} {closing_world[2]:+.3f}]  |z|={abs(closing_world[2]):.3f}  "
                  f"(horizontal: want |z|≈0)")
            print(f"  up       (Y): [{up_world[0]:+.3f} {up_world[1]:+.3f} {up_world[2]:+.3f}]  |z|={abs(up_world[2]):.3f}  "
                  f"(horizontal: want |z|≈1)")
            print(f"  approach (Z): [{approach_world[0]:+.3f} {approach_world[1]:+.3f} {approach_world[2]:+.3f}]  z={approach_world[2]:+.3f}  "
                  f"(vertical: want z≈-1 for top-down)")

        # GraspGen origin = panda_hand/wrist; EE (panda_grasptarget) is depth further along Z.
        # Shift position to the contact point so IK targets the fingertip frame, not the wrist.
        _GRASPGEN_DEPTH = 0.10527314  # franka_panda.yaml "depth"
        T_offset = np.eye(4, dtype=np.float64)
        T_offset[2, 3] = _GRASPGEN_DEPTH
        T_tcp_base = ctx.last_pose_mat @ (T_cam @ T_offset)

        pos = T_tcp_base[:3, 3]  # EE (fingertip) position in robot base frame

        # GraspGen X (closing) and Panda eef X (closing) point the same way —
        # verified empirically: the finger pads' separation vector in world
        # frame coincides with the eef body's X axis. So no body-frame rotation
        # is needed; use the GraspGen orientation directly.
        quat = _rot_to_wxyz(T_tcp_base[:3, :3])
        quat /= np.linalg.norm(quat)

        # Approach: back off along -Z (Z points toward object)
        approach_dir = T_tcp_base[:3, 2]
        approach_pos = (pos - approach_dir * _GRASP_APPROACH_CLEARANCE).tolist()

        if ctx.graspgen_debug_dir is not None:
            _, pc_segment_dbg = _unproject_pointclouds(ctx.last_depth, ctx.last_K, mask)
            _, pc_scene_dbg   = _unproject_pointclouds(ctx.last_depth, ctx.last_K, ~mask)
            _save_graspgen_debug(
                pc_segment_dbg, grasps, scores, best_idx,
                ctx.last_pose_mat, target, ctx.graspgen_debug_dir,
                pc_scene=pc_scene_dbg,
            )

        goal, perturbed = _branch_grasp_target(ctx, pos)
        approach_pos = (goal - approach_dir * _GRASP_APPROACH_CLEARANCE).tolist()
        ctx.arm.move(approach_pos, quat)
        ctx.arm.move(goal.tolist(), quat)
        ctx.grasp_perturbation_applied = perturbed

        result = ctx.arm.grasp()
        return _grasp_feedback(ctx, result)

    # ── PCA path (default) ──────────────────────────────────────────────── #
    pc_full, pc_segment = _unproject_pointclouds(ctx.last_depth, ctx.last_K, mask)
    if pc_segment.shape[0] < 10:
        raise RuntimeError(
            f"Too few points for '{target}' ({pc_segment.shape[0]}) — object may be occluded"
        )
    pc_segment = _remove_statistical_outliers(pc_segment, k=20, std_ratio=2.0)
    if pc_segment.shape[0] < 5:
        raise RuntimeError(
            f"Too few points for '{target}' after filtering — object may be occluded"
        )

    # Transform segmented pointcloud to robot base frame (for heights)
    N  = pc_segment.shape[0]
    h  = np.hstack([pc_segment, np.ones((N, 1))])
    pc = (ctx.last_pose_mat @ h.T).T[:, :3]

    # XY: use camera-frame AABB with radius-estimated depth correction.
    # Only the camera-facing surface is visible; mean/AABB-centre is biased ~R/2
    # toward the camera.  Estimate radius from the X extent (image width ≈ diameter)
    # and offset from the front surface: mins[2] + R_est ≈ true centre depth.
    mins_cam = pc_segment.min(axis=0)
    maxs_cam = pc_segment.max(axis=0)
    r_est    = (maxs_cam[0] - mins_cam[0]) / 2.0
    ctr_cam  = np.array([pc_segment[:, 0].mean(),
                          pc_segment[:, 1].mean(),
                          mins_cam[2] + r_est, 1.0])
    ctr_base = (ctx.last_pose_mat @ ctr_cam)[:3]
    cx, cy   = float(ctr_base[0]), float(ctr_base[1])

    top_z = float(pc[:, 2].max())

    # Anchor the grasp on a real surface point near the top, then take PCA
    # over its local neighbourhood. Solid tops: the closest point to the
    # centroid IS essentially the centroid, so behaviour is unchanged. Annular
    # tops (bowl rim): the centroid sits in the empty interior, so the closest
    # real point lands on the rim, and local PCA's short axis = radial → jaws
    # straddle the rim wall.
    band     = pc[pc[:, 2] > top_z - 0.015]
    if len(band) < 10:
        band = pc
    band_xy   = band[:, :2]
    band_cent = band_xy.mean(axis=0)
    d_cent    = np.linalg.norm(band_xy - band_cent, axis=1)
    near_min  = band[d_cent < d_cent.min() + 0.005]
    cur_pos, _ = ctx.env.eef_pose_base()
    anchor    = near_min[int(np.argmin(np.linalg.norm(near_min[:, :2] - cur_pos[:2], axis=1)))]
    cx, cy    = float(anchor[0]), float(anchor[1])

    local = band[np.linalg.norm(band_xy - anchor[:2], axis=1) < 0.025]
    if len(local) < 3:
        local = band
    xy       = local[:, :2] - local[:, :2].mean(axis=0)
    _, _, Vt = np.linalg.svd(xy, full_matrices=False)
    # SVD's row signs are arbitrary. Canonicalize Vt[1] so identical scenes
    # don't randomly request a 180°-flipped wrist between runs.
    if Vt[1, 1] < 0 or (Vt[1, 1] == 0 and Vt[1, 0] < 0):
        Vt[1] = -Vt[1]

    yaw      = float(np.arctan2(Vt[1, 1], Vt[1, 0]))

    # Diagnostic: print the PCA axes in base-frame XY so we can verify the
    # closing axis ends up perpendicular to the brick's long axis.
    primary_xy   = (float(Vt[0, 0]), float(Vt[0, 1]))
    secondary_xy = (float(Vt[1, 0]), float(Vt[1, 1]))
    closing_xy   = (float(np.cos(yaw)), float(np.sin(yaw)))
    print(
        f"  [pca]   anchor xy=[{cx:.3f}, {cy:.3f}] top_z={top_z:.3f}  n_local={len(local)}"
    )
    print(
        f"  [pca]   primary (long)  axis (base xy)={[round(v, 3) for v in primary_xy]}"
    )
    print(
        f"  [pca]   secondary (short) axis (base xy)={[round(v, 3) for v in secondary_xy]}"
    )
    print(
        f"  [pca]   closing axis (base xy)={[round(v, 3) for v in closing_xy]}  yaw={np.degrees(yaw):+.1f}°"
    )

    # Contact height: just below the top so fingers wrap the rim/edge of the
    # object. (Avoids the previous mean-z heuristic which dipped into the
    # empty interior of bowl-shaped objects.)
    cz = top_z - 0.01

    # Jaw is symmetric mod π — pick the yaw closest to the current wrist so
    # the IK never requests a redundant 180° wrist rotation.
    _, cur_q = ctx.env.eef_pose_base()
    cw, cx_, cy_, cz_ = cur_q
    cur_yaw = float(np.arctan2(2*(cw*cz_ + cx_*cy_), 1 - 2*(cy_*cy_ + cz_*cz_)))
    def _wrap(d: float) -> float: return (d + np.pi) % (2*np.pi) - np.pi
    if abs(_wrap(yaw + np.pi - cur_yaw)) < abs(_wrap(yaw - cur_yaw)):
        yaw += np.pi

    # Straight-down quaternion with PCA yaw: roll=180°, pitch=0°, yaw=yaw
    quat = np.array([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])

    if ctx.bbox_debug_dir is not None:
        # PCA axis endpoints in base frame, sized to the object's actual extent
        # along each axis (so axis length = object span in that direction).
        xy_centroid = pc[:, :2].mean(axis=0)
        proj0 = xy @ Vt[0]
        proj1 = xy @ Vt[1]
        v0 = np.array([Vt[0, 0], Vt[0, 1], 0.0])
        v1 = np.array([Vt[1, 0], Vt[1, 1], 0.0])
        c3 = np.array([xy_centroid[0], xy_centroid[1], top_z])
        seg_primary   = np.stack([c3 + proj0.min() * v0, c3 + proj0.max() * v0])
        seg_secondary = np.stack([c3 + proj1.min() * v1, c3 + proj1.max() * v1])

        _save_bbox_debug(
            pc_segment, ctx.last_pose_mat,
            mins_cam, maxs_cam, ctr_base,
            target, ctx.bbox_debug_dir,
            pca_axes_base=(seg_primary, seg_secondary),
            grasp_frame_base=(np.array([cx, cy, cz]), quat),
            base_to_world=_get_base_to_world(ctx.env),
        )

    goal, perturbed = _branch_grasp_target(ctx, np.asarray([cx, cy, cz]))
    approach = goal + np.array([0, 0, top_z + _GRASP_APPROACH_CLEARANCE - cz])
    ctx.arm.move(approach.tolist(), quat)
    ctx.arm.move(goal.tolist(), quat)
    ctx.grasp_perturbation_applied = perturbed

    # ### Pure debug code ###
    # import viser.transforms as vtf
    # _, actual_q = ctx.env.eef_pose_base()
    # R_actual = vtf.SO3(wxyz=actual_q).as_matrix()
    # R_req    = vtf.SO3(wxyz=quat).as_matrix()
    # print(f"  [grasp-debug] requested eef-X (jaw) in base : {R_req[:, 0].round(3)}")
    # print(f"  [grasp-debug] actual    eef-X (jaw) in base : {R_actual[:, 0].round(3)}")
    # print(f"  [grasp-debug] Vt[1] target jaw dir          : {np.array([Vt[1,0], Vt[1,1], 0]).round(3)}")
    # ### Pure debug code ###
    result = ctx.arm.grasp()
    return _grasp_feedback(ctx, result)


def align(ctx: ToolContext, target_name: str, position: str, clearance: str) -> str:
    """Move gripper to a named side of target_name. position: top/left/right/front/back. clearance: small/medium/large. Z unchanged for side positions."""
    position  = position.strip().lower()
    clearance = clearance.strip().lower()
    if position not in _ALL_POSITIONS:
        raise ValueError(f"position must be one of {sorted(_ALL_POSITIONS)}, got '{position}'")
    if clearance not in _CLEARANCE_M:
        raise ValueError(f"clearance must be one of {sorted(_CLEARANCE_M)}, got '{clearance}'")
    if not hasattr(ctx.perception, "get_mask"):
        raise RuntimeError("align() requires SAM3 perception backend (not GT)")

    _refresh(ctx)
    mask = ctx.perception.get_mask(target_name, ctx.last_rgb)
    _, pc_segment = _unproject_pointclouds(ctx.last_depth, ctx.last_K, mask)
    if pc_segment.shape[0] < 10:
        raise RuntimeError(
            f"Too few points for '{target_name}' ({pc_segment.shape[0]}) — object may be occluded"
        )

    pc_segment = _remove_statistical_outliers(pc_segment, k=20, std_ratio=2.0)
    if pc_segment.shape[0] < 5:
        raise RuntimeError(
            f"Too few points for '{target_name}' after filtering — object may be occluded"
        )

    # Transform to base frame for position/size reporting
    N_pts  = pc_segment.shape[0]
    h_pts  = np.hstack([pc_segment, np.ones((N_pts, 1))])
    pc_base = (ctx.last_pose_mat @ h_pts.T).T[:, :3]
    obj_center  = [round(float(v), 3) for v in pc_base.mean(axis=0)]
    obj_extents = [round(float(v), 3) for v in (pc_base.max(axis=0) - pc_base.min(axis=0))]

    current_z  = ctx.arm.position()[2]
    target_pos = _compute_align_target(
        pc_segment, position, _CLEARANCE_M[clearance], ctx.last_pose_mat, current_z
    )

    if ctx.bbox_debug_dir is not None:
        mins_cam = pc_segment.min(axis=0)
        maxs_cam = pc_segment.max(axis=0)
        _save_bbox_debug(pc_segment, ctx.last_pose_mat,
                         mins_cam, maxs_cam, target_pos,
                         f"{target_name}_{position}", ctx.bbox_debug_dir,
                         base_to_world=_get_base_to_world(ctx.env))

    if position in _SIDE_POSITIONS:
        # Reset roll/pitch, then rotate yaw so wide face points toward object.
        ctx.arm.align_down()
        base_dir = ctx.last_pose_mat[:3, :3] @ _CAM_APPROACH_DIR[position]
        approach_angle = np.arctan2(base_dir[1], base_dir[0])
        desired_yaw = approach_angle - _WIDE_FACE_YAW_OFFSET
        _, cur_quat = ctx.env.eef_pose_base()
        w, x, y, z = cur_quat
        cur_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        delta_deg = np.degrees((desired_yaw - cur_yaw + np.pi) % (2 * np.pi) - np.pi)
        # print(f"[align push-yaw] approach_angle={np.degrees(approach_angle):.1f}° desired_yaw={np.degrees(desired_yaw):.1f}° cur_yaw={np.degrees(cur_yaw):.1f}° delta={delta_deg:.1f}°")
        # Gripper is symmetric under 180° rotation for pushing; prefer anticlockwise
        if delta_deg < -89.9:   # clockwise ≥90° → use anticlockwise equivalent
            delta_deg += 180
        # print(f"[align push-yaw] clamped delta={delta_deg:.1f}°")
        if abs(delta_deg) > 0.5:
            ctx.arm.rotate(delta_deg, 'z')

    ctx.arm.move(target_pos.tolist())
    return json.dumps({"center": obj_center, "extents": obj_extents})


_RETRACT_AFTER_RELEASE_M = 0.10
# Width below which the gripper is treated as "closed on nothing" — if we
# reach release() in that state we skip the auto-retract, even if was_holding
# is set, because the object slipped out during a motion (gripper kept closing
# once nothing was resisting) and the original grasp pose is no longer the
# meaningful reference to retract from.
_MIN_HOLDING_WIDTH_M = 0.002


def _body_z_in_world(quat_wxyz: np.ndarray) -> np.ndarray:
    """Third column of the rotation matrix — the gripper's approach axis."""
    w, x, y, z = quat_wxyz
    return np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])


def release(ctx: ToolContext) -> str:
    """Open gripper; retract 10 cm opposite approach only if a successful grasp still holds an object.
    Returns 'released'. A release after a missed grasp or silent mid-motion slip
    opens the fingers without an automatic retreat. Observe the resulting pose;
    retraction may also fail if the retreat target is unreachable."""
    actually_holding = ctx.was_holding and ctx.arm.gripper_width() > _MIN_HOLDING_WIDTH_M
    ctx.was_holding = False
    result = ctx.arm.release()
    if actually_holding:
        pos = np.asarray(ctx.arm.position(), dtype=np.float64)
        quat = np.asarray(ctx.arm.rotation(), dtype=np.float64)
        approach = _body_z_in_world(quat)
        retract = (pos - _RETRACT_AFTER_RELEASE_M * approach).tolist()
        try:
            ctx.arm.move(retract)
        except Exception as e:
            # Don't fail the release just because the retract path is unreachable;
            # the gripper is already open, the LLM can recover on its own.
            print(f"[release] auto-retract failed: {type(e).__name__}: {e}")
    return result


def rotate(ctx: ToolContext, angle_deg: float, axis: str) -> str:
    """Rotate gripper in place by angle_deg around body-frame axis (x/y/z)."""
    return ctx.arm.rotate(angle_deg, axis)


def stop(ctx: ToolContext) -> str:
    """Signal end of episode."""
    return "stop"


def align_gripper_down(ctx: ToolContext) -> str:
    """Orient gripper to face straight down (roll=180°, pitch=0°)."""
    return ctx.arm.align_down()


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

def make_tool_registry(ctx: ToolContext) -> dict[str, Any]:
    """Return {name: callable(**kwargs)} — curried with ctx.

    The registry exposes exactly the nine tools listed in the system prompt
    (`configs/prompts/sft_v13b_short.txt`). The model-facing coordinate adapter
    exposes canonical target_name arguments. Keep this physical tool set
    in lockstep with that prompt: if you add a tool, add it both here AND
    there; if you drop one from the prompt, drop it here too. Older legacy
    aliases (`get_object_position`, `face_down`, `align_gripper_down`,
    gripper-state getters) are no longer registered — the v5 model never
    calls them.
    """
    import functools
    fns = [
        move, grasp, align, get_position, get_position_and_size,
        close_gripper, release, rotate, home_pose,
    ]
    return {fn.__name__: functools.partial(fn, ctx) for fn in fns}
