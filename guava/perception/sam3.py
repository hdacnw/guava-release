from __future__ import annotations
import numpy as np
import requests
from pathlib import Path


class SAM3Solver:
    def __init__(self, url: str, debug_dir: str | None = None):
        self.url = url.rstrip("/")
        self._debug_dir = Path(debug_dir) if debug_dir else None

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.url}/health", timeout=1.0)
            return True
        except Exception:
            return False

    def get_mask(self, prompt: str, rgb: np.ndarray) -> np.ndarray:
        """Return the best boolean mask (H×W) for the given text prompt."""
        import base64, io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = requests.post(
            f"{self.url}/segment",
            json={"image_b64": b64, "text_prompt": prompt},
            timeout=30.0,
        )
        resp.raise_for_status()
        masks = resp.json().get("masks", [])
        if not masks:
            raise RuntimeError(f"SAM3 found no objects matching '{prompt}'")
        best = max(masks, key=lambda m: m.get("score", 0.0))
        mask = np.array(best["mask"], dtype=bool)
        if not mask.any():
            raise RuntimeError(f"SAM3 mask for '{prompt}' is empty")

        if self._debug_dir is not None:
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(mask)
            v, u = np.unravel_index(np.argmax(dist), dist.shape)
            _save_debug_overlay(rgb, mask, int(u), int(v), prompt, self._debug_dir)

        return mask

    def get_position(
        self,
        prompt: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        pose_mat: np.ndarray,
    ) -> list[float]:
        """Segment object by text prompt, unproject centroid to 3-D."""
        import base64, io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = requests.post(
            f"{self.url}/segment",
            json={"image_b64": b64, "text_prompt": prompt},
            timeout=30.0,
        )
        resp.raise_for_status()
        masks = resp.json().get("masks", [])
        if not masks:
            raise RuntimeError(f"SAM3 found no objects matching '{prompt}'")

        best = max(masks, key=lambda m: m.get("score", 0.0))
        mask = np.array(best["mask"], dtype=bool)
        if not mask.any():
            raise RuntimeError(f"SAM3 mask for '{prompt}' is empty")

        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask)
        v, u = np.unravel_index(np.argmax(dist), dist.shape)
        u, v = int(u), int(v)

        depth2d = depth[..., 0] if depth.ndim == 3 else depth
        valid = depth2d[mask]
        valid = valid[valid > 0]
        if valid.size > 0:
            d_front = float(np.percentile(valid, 5))
            R_px = float(np.sqrt(mask.sum() / np.pi))
            r_world = R_px * d_front / float(K[0, 0])
            z_center = d_front + r_world
        else:
            z_center = None

        if self._debug_dir is not None:
            _save_debug_overlay(rgb, mask, u, v, prompt, self._debug_dir)
        return _unproject(u, v, depth, K, pose_mat, z_override=z_center)


_debug_call_count: dict[str, int] = {}


def _save_debug_overlay(
    rgb: np.ndarray, mask: np.ndarray, u: int, v: int, prompt: str, out_dir: Path,
) -> None:
    from PIL import Image, ImageDraw
    import re
    key = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    idx = _debug_call_count.get(key, 0)
    _debug_call_count[key] = idx + 1

    overlay = rgb.copy().astype(np.float32)
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 180, 0], dtype=np.float32) * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(img)
    r = 6
    draw.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0), width=2)
    draw.line([u - 12, v, u + 12, v], fill=(255, 0, 0), width=2)
    draw.line([u, v - 12, u, v + 12], fill=(255, 0, 0), width=2)
    draw.text((u + 8, v - 8), f"({u},{v})", fill=(255, 255, 0))

    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"{key}_{idx:03d}.png")


def _unproject(u, v, depth, K, pose_mat, z_override=None) -> list[float]:
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    z = z_override if z_override is not None else float(depth[v, u])
    if z <= 0:
        raise ValueError(f"Invalid depth {z} at ({u},{v})")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam_pt = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
    p = pose_mat @ np.append(cam_pt, 1.0)
    return [float(p[0]), float(p[1]), float(p[2])]
