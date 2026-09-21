"""Shared bounding-box support for imported object assets."""
import numpy as np
from robosuite.models.objects import MujocoXMLObject

class _BBoxXMLObject(MujocoXMLObject):
    def __init__(self, fname, name, half_size, **kwargs):
        super().__init__(fname=fname, name=name, **kwargs)
        # Apply the active scale to the bbox so placement math matches the mesh.
        s = self._scale if self._scale is not None else 1.0
        if np.isscalar(s):
            scale_vec = np.array([s, s, s], dtype=float)
        else:
            scale_vec = np.asarray(s, dtype=float)
        self._half_size = np.asarray(half_size, dtype=float) * scale_vec

    @property
    def bottom_offset(self):
        return np.array([0.0, 0.0, -self._half_size[2]])

    @property
    def top_offset(self):
        return np.array([0.0, 0.0, self._half_size[2]])

    @property
    def horizontal_radius(self):
        return float(np.linalg.norm(self._half_size[:2]))

    def get_bounding_box_half_size(self):
        return self._half_size.copy()
