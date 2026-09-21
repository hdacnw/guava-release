from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _unproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return (M, 3) point cloud in camera frame for pixels where mask is True."""
    d = depth[..., 0] if depth.ndim == 3 else depth
    h, w = d.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sel = (d > 0) & mask
    z = d[sel].astype(np.float32)
    x = ((us[sel] - K[0, 2]) * z / K[0, 0]).astype(np.float32)
    y = ((vs[sel] - K[1, 2]) * z / K[1, 1]).astype(np.float32)
    return np.stack([x, y, z], axis=1)


class GraspGenClient:
    """Thin wrapper around the GraspGen ZMQ client.

    Presents the same plan_grasps(depth, cam_K, segmap) interface used by
    tools.py so that it can be used as a drop-in replacement for GraspNetClient.
    """

    def __init__(self, host: str = "localhost", port: int = 5556, timeout_ms: int = 60_000):
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._client = None  # lazy-init on first call

    def _ensure_client(self):
        if self._client is not None:
            return
        # GraspGen ships its ZMQ client in third_party/GraspGen/grasp_gen/serving/
        graspgen_root = Path(__file__).resolve().parents[2] / "third_party" / "GraspGen"
        if str(graspgen_root) not in sys.path:
            sys.path.insert(0, str(graspgen_root))
        from grasp_gen.serving.zmq_client import GraspGenClient as _ZMQClient
        self._client = _ZMQClient(
            host=self._host,
            port=self._port,
            timeout_ms=self._timeout_ms,
            wait_for_server=False,
        )

    def is_available(self) -> bool:
        try:
            self._ensure_client()
            return self._client.health_check()
        except Exception:
            return False

    def plan_grasps(
        self,
        depth: np.ndarray,
        cam_K: np.ndarray,
        segmap: np.ndarray,
        segmap_id: int = 1,
        num_grasps: int = 200,
        topk_num_grasps: int = 50,
        collision_threshold: float = 0.002,
        max_scene_points: int = 32768,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (grasps Nx4x4, scores N) in camera frame, collision-free w.r.t. scene.

        Args:
            depth:              (H, W) float32 metric depth in metres.
            cam_K:              (3, 3) camera intrinsic matrix.
            segmap:             (H, W) int32 mask; target pixels == segmap_id.
            segmap_id:          which segmap value identifies the target object.
            num_grasps:         diffusion samples requested from the server.
            topk_num_grasps:    maximum grasps to return.
            collision_threshold: min distance (m) to scene for a grasp to be kept.
            max_scene_points:   scene PC is downsampled to this size for speed.
        """
        self._ensure_client()

        obj_mask   = (segmap == segmap_id)
        pc_obj_cam = _unproject(depth, cam_K, obj_mask)
        if len(pc_obj_cam) < 10:
            return np.empty((0, 4, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

        # Scene PC: all valid depth pixels that are NOT the target object
        pc_scene_cam = _unproject(depth, cam_K, ~obj_mask)

        # Centre both PCs on the object centroid (GraspGen requirement)
        centroid        = pc_obj_cam.mean(axis=0)
        pc_obj_centred  = (pc_obj_cam   - centroid).astype(np.float32)
        pc_scene_centred = (pc_scene_cam - centroid).astype(np.float32)

        response = self._client._request({
            "action":              "infer_collision_free",
            "point_cloud":         pc_obj_centred,
            "scene_pc":            pc_scene_centred,
            "num_grasps":          num_grasps,
            "topk_num_grasps":     topk_num_grasps,
            "collision_threshold": collision_threshold,
            "max_scene_points":    max_scene_points,
        })

        grasps_centred = np.asarray(response["grasps"],      dtype=np.float32)
        scores         = np.asarray(response["confidences"], dtype=np.float32)

        if len(grasps_centred) == 0:
            return np.empty((0, 4, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

        # Translate grasp positions back to camera frame
        grasps = grasps_centred.copy()
        grasps[:, :3, 3] += centroid
        return grasps, scores

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
