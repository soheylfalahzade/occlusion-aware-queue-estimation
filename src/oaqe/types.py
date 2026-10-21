"""Core data containers and vehicle-class priors shared across the package."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class VehiclePrior:
    """Nominal physical dimensions of a vehicle class, in metres."""

    length_m: float
    width_m: float
    height_m: float


VEHICLE_PRIORS: dict[str, VehiclePrior] = {
    "car": VehiclePrior(4.5, 1.8, 1.5),
    "motorcycle": VehiclePrior(2.1, 0.8, 1.4),
    "bus": VehiclePrior(11.0, 2.5, 3.2),
    "truck": VehiclePrior(7.5, 2.5, 3.0),
}
DEFAULT_CLASS = "car"


def prior_for(cls_name: str) -> VehiclePrior:
    """Return the dimension prior for ``cls_name`` (falls back to a passenger car)."""
    return VEHICLE_PRIORS.get(cls_name, VEHICLE_PRIORS[DEFAULT_CLASS])


@dataclass(frozen=True)
class Detection:
    """One tracked detection in image space (output of any detector + tracker)."""

    track_id: int
    cls: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float = 1.0

    @property
    def ground_anchor_px(self) -> tuple[float, float]:
        """Bottom-centre of the bounding box, the pixel assumed to lie on the road plane."""
        x1, _, x2, y2 = self.bbox_xyxy
        return 0.5 * (x1 + x2), y2


@dataclass(frozen=True)
class LaneSpec:
    """A lane expressed in the metric ground plane.

    Lane coordinates are ``(s, l)``: ``s`` is the distance from the stop line measured
    upstream (against the direction of travel) and ``l`` is the signed lateral offset
    from the lane centre line, positive to the left of ``direction``.
    """

    lane_id: str
    stop_point_m: tuple[float, float]
    direction: tuple[float, float]
    width_m: float = 3.5
    length_m: float = 100.0
    approach_id: str = "A"

    @property
    def unit(self) -> np.ndarray:
        """Unit vector pointing from the stop line upstream."""
        d = np.asarray(self.direction, dtype=float)
        return d / np.linalg.norm(d)

    @property
    def normal(self) -> np.ndarray:
        """Unit vector pointing to the left of :attr:`unit`."""
        u = self.unit
        return np.array([-u[1], u[0]])

    def to_lane(self, xy: ArrayLike) -> np.ndarray:
        """Convert ground-plane points ``(..., 2)`` to lane coordinates ``(..., 2)``."""
        p = np.asarray(xy, dtype=float) - np.asarray(self.stop_point_m, dtype=float)
        return np.stack([p @ self.unit, p @ self.normal], axis=-1)

    def to_world(self, sl: ArrayLike) -> np.ndarray:
        """Convert lane coordinates ``(..., 2)`` back to ground-plane points."""
        sl = np.asarray(sl, dtype=float)
        return (
            np.asarray(self.stop_point_m, dtype=float)
            + sl[..., 0:1] * self.unit
            + sl[..., 1:2] * self.normal
        )


@dataclass(frozen=True)
class LaneEstimate:
    """Per-lane output of the estimator at one output tick."""

    lane_id: str
    timestamp_s: float
    queue_length_m: float
    queue_std_m: float
    queue_ci95_m: tuple[float, float]
    vehicle_count: float
    count_std: float
    occupancy: float
    visible_fraction: float
    confidence: float
    n_visible_stopped: int
    n_ghost: int
    n_hidden_inferred: float
    has_measurement: bool

    def as_dict(self) -> dict[str, Any]:
        """Flat dictionary representation (CSV / JSON friendly)."""
        d = asdict(self)
        lo, hi = d.pop("queue_ci95_m")
        d["queue_ci95_lo_m"], d["queue_ci95_hi_m"] = lo, hi
        return d
