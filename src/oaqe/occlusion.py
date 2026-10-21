"""Geometric occlusion reasoning on the lane centre line."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from oaqe.geometry import segments_blocked_by_boxes
from oaqe.types import LaneSpec


@dataclass(frozen=True)
class OccluderBox:
    """Vehicle volume expressed in the lane frame of the lane being analysed.

    ``visible`` is False for ghost vehicles (stopped vehicles remembered but not
    currently detected). Ghosts occlude like any physical vehicle, but the road
    surface they cover is *not* considered observed.
    """

    s_min: float
    s_max: float
    l_min: float
    l_max: float
    z_max: float
    visible: bool = True


class OcclusionModel:
    """Decides which parts of a lane are hidden from the camera by known vehicles.

    The camera is a point at ``camera_position_m`` (ground x, y and height z). A probe
    point on the lane centre line at ``probe_height_m`` is *hidden* if the straight
    segment from the camera to the probe intersects another vehicle's bounding volume.
    """

    def __init__(
        self, camera_position_m: tuple[float, float, float], probe_height_m: float = 0.9
    ) -> None:
        self.camera = np.asarray(camera_position_m, dtype=float)
        self.probe_height_m = float(probe_height_m)

    def camera_in_lane_frame(self, lane: LaneSpec) -> np.ndarray:
        """Camera position as ``(s, l, z)`` in the frame of ``lane``."""
        s, l_off = lane.to_lane(self.camera[:2])
        return np.array([s, l_off, self.camera[2]])

    def hidden_mask(
        self,
        lane: LaneSpec,
        s_grid: np.ndarray,
        boxes: Sequence[OccluderBox],
        lateral_m: float = 0.0,
    ) -> np.ndarray:
        """Boolean mask over ``s_grid`` marking probe points hidden from the camera."""
        s_grid = np.asarray(s_grid, dtype=float)
        if len(boxes) == 0:
            return np.zeros(len(s_grid), dtype=bool)
        targets = np.column_stack(
            [s_grid, np.full_like(s_grid, lateral_m), np.full_like(s_grid, self.probe_height_m)]
        )
        bmin = np.array([[b.s_min, b.l_min, 0.0] for b in boxes])
        bmax = np.array([[b.s_max, b.l_max, b.z_max] for b in boxes])
        blocked = segments_blocked_by_boxes(self.camera_in_lane_frame(lane), targets, bmin, bmax)
        hidden = blocked.any(axis=1)
        inside = (
            (s_grid[:, None] >= bmin[None, :, 0])
            & (s_grid[:, None] <= bmax[None, :, 0])
            & (lateral_m >= bmin[None, :, 1])
            & (lateral_m <= bmax[None, :, 1])
        )
        visible_flags = np.array([b.visible for b in boxes])
        in_visible = (inside & visible_flags[None, :]).any(axis=1)
        in_ghost = (inside & ~visible_flags[None, :]).any(axis=1)
        return (hidden & ~in_visible) | in_ghost
