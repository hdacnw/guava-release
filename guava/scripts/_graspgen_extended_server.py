"""Extended GraspGen ZMQ server with collision-free grasp filtering.

This script runs under GraspGen's Python venv (not guava's). It is invoked by
guava/scripts/graspgen_server.py via os.execv().

Adds the `infer_collision_free` action which accepts an object point cloud and a
scene point cloud, runs grasp inference on the object, then filters out grasps
that collide with the scene using the gripper collision mesh.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from grasp_gen.serving.zmq_server import GraspGenZMQServer
from grasp_gen.grasp_server import GraspGenSampler
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
from grasp_gen.robot import get_gripper_info

logger = logging.getLogger(__name__)


class CollisionAwareGraspGenZMQServer(GraspGenZMQServer):
    """Extends GraspGenZMQServer with scene-aware collision filtering."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        gripper_info = get_gripper_info(self._gripper_name)
        self._gripper_collision_mesh = gripper_info.collision_mesh
        logger.info(
            "Loaded gripper collision mesh for '%s' (%d vertices)",
            self._gripper_name,
            len(self._gripper_collision_mesh.vertices),
        )

    def _handle(self, request: dict) -> dict:
        if request.get("action") == "infer_collision_free":
            return self._handle_infer_collision_free(request)
        return super()._handle(request)

    def _handle_infer_collision_free(self, request: dict) -> dict:
        """
        Request fields:
          point_cloud (N,3) float32   — object point cloud, centroid-centred
          scene_pc    (M,3) float32   — scene point cloud, same frame (object removed)
          collision_threshold float   — min distance to flag collision (default 0.002 m)
          max_scene_points    int     — downsample scene to this many pts (default 8192)
          + all standard infer params (num_grasps, topk_num_grasps, …)

        Response fields: same as infer + collision_ms timing.
        """
        point_cloud = request.get("point_cloud")
        scene_pc    = request.get("scene_pc")
        if point_cloud is None:
            return {"error": "Missing required field 'point_cloud'"}
        if scene_pc is None:
            return {"error": "Missing required field 'scene_pc'"}

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        scene_pc    = np.asarray(scene_pc,    dtype=np.float32)

        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            return {"error": f"point_cloud must be (N,3), got {point_cloud.shape}"}
        if scene_pc.ndim != 2 or scene_pc.shape[1] != 3:
            return {"error": f"scene_pc must be (M,3), got {scene_pc.shape}"}

        collision_threshold = float(request.get("collision_threshold", 0.002))
        max_scene_points    = int(request.get("max_scene_points", 32768))

        infer_params = {
            "grasp_threshold": float(request.get("grasp_threshold", -1.0)),
            "num_grasps":      int(request.get("num_grasps", 200)),
            "topk_num_grasps": int(request.get("topk_num_grasps", -1)),
            "min_grasps":      int(request.get("min_grasps", 40)),
            "max_tries":       int(request.get("max_tries", 6)),
            "remove_outliers": bool(request.get("remove_outliers", True)),
        }

        t0 = time.monotonic()
        grasps, grasp_conf = GraspGenSampler.run_inference(
            point_cloud, self._sampler, **infer_params
        )
        infer_ms = (time.monotonic() - t0) * 1000

        if len(grasps) == 0:
            return {
                "grasps": np.empty((0, 4, 4), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
                "num_grasps": 0,
                "timing": {"infer_ms": infer_ms, "collision_ms": 0.0},
            }

        grasps_np = grasps.cpu().numpy().astype(np.float32)
        conf_np   = grasp_conf.cpu().numpy().astype(np.float32)

        # Downsample scene PC for faster collision checking
        if len(scene_pc) > max_scene_points:
            idx      = np.random.choice(len(scene_pc), max_scene_points, replace=False)
            scene_pc = scene_pc[idx]

        t1 = time.monotonic()
        collision_mask = filter_colliding_grasps(
            scene_pc=scene_pc,
            grasp_poses=grasps_np,
            gripper_collision_mesh=self._gripper_collision_mesh,
            collision_threshold=collision_threshold,
        )
        collision_ms = (time.monotonic() - t1) * 1000

        cf_grasps = grasps_np[collision_mask]
        cf_conf   = conf_np[collision_mask]

        logger.info(
            "infer_collision_free: %d/%d collision-free grasps "
            "(infer=%.0fms collision=%.0fms)",
            len(cf_grasps), len(grasps_np), infer_ms, collision_ms,
        )

        return {
            "grasps":      cf_grasps,
            "confidences": cf_conf,
            "num_grasps":  len(cf_grasps),
            "timing":      {"infer_ms": infer_ms, "collision_ms": collision_ms},
        }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="GraspGen ZMQ server with collision-free grasp filtering"
    )
    parser.add_argument("--gripper_config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()

    server = CollisionAwareGraspGenZMQServer(
        gripper_config=args.gripper_config,
        host=args.host,
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
