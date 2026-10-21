"""Max-pressure quantities and fuzzy-controller inputs derived from lane estimates."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from oaqe.types import LaneEstimate, LaneSpec


def spillback_risk(occupancy: float, low: float = 0.5, high: float = 0.9) -> float:
    """Piecewise-linear risk in [0, 1] that rises from ``low`` to ``high`` link occupancy."""
    return float(np.clip((occupancy - low) / (high - low), 0.0, 1.0))


@dataclass(frozen=True)
class MovementSpec:
    """A signalised movement: one upstream lane feeding downstream lanes with turn ratios."""

    movement_id: str
    upstream_lane: str
    downstream_lanes: tuple[str, ...] = ()
    turn_ratios: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.downstream_lanes) != len(self.turn_ratios):
            raise ValueError("downstream_lanes and turn_ratios must have equal length")


@dataclass(frozen=True)
class PressureEstimate:
    """Max-pressure weight of a movement with propagated uncertainty."""

    movement_id: str
    timestamp_s: float
    pressure: float
    std: float
    normalized_pressure: float
    spillback_risk: float
    confidence: float


class PressureEstimator:
    """Computes ``w(m) = n_up - sum_k r_k n_down_k`` from lane estimates.

    Standard deviations are propagated assuming independent lane errors. The
    normalised pressure divides each count by the lane's nominal storage capacity.
    """

    def __init__(
        self,
        movements: Sequence[MovementSpec],
        lanes: Sequence[LaneSpec],
        pitch_m: float = 6.8,
    ) -> None:
        self.movements = list(movements)
        self.capacity = {ln.lane_id: max(ln.length_m / pitch_m, 1.0) for ln in lanes}

    def compute(self, estimates: Mapping[str, LaneEstimate]) -> list[PressureEstimate]:
        """Pressure for every movement whose lanes are all present in ``estimates``."""
        out: list[PressureEstimate] = []
        for mv in self.movements:
            ids = (mv.upstream_lane, *mv.downstream_lanes)
            if any(i not in estimates for i in ids):
                continue
            up = estimates[mv.upstream_lane]
            down = [estimates[i] for i in mv.downstream_lanes]
            ratios = np.asarray(mv.turn_ratios, dtype=float)
            pairs = list(zip(ratios, down, strict=True))
            p = up.vehicle_count - sum(r * d.vehicle_count for r, d in pairs)
            var = up.count_std**2 + sum((r * d.count_std) ** 2 for r, d in pairs)
            norm = up.vehicle_count / self.capacity[mv.upstream_lane] - sum(
                r * d.vehicle_count / self.capacity[d.lane_id] for r, d in pairs
            )
            risk = max((spillback_risk(d.occupancy) for d in down), default=0.0)
            conf = min(e.confidence for e in (up, *down))
            out.append(
                PressureEstimate(
                    mv.movement_id, up.timestamp_s, float(p), float(np.sqrt(var)),
                    float(norm), risk, float(conf),
                )
            )
        return out


def to_fuzzy_inputs(
    estimate: LaneEstimate, lane: LaneSpec, pitch_m: float = 6.8
) -> dict[str, float]:
    """Normalised crisp inputs for a fuzzy signal controller.

    Returns:
        ``queue_norm`` in [0, 1] (count over storage capacity), ``queue_m``, ``confidence``
        in [0, 1], ``std_norm`` (queue std over lane length) and ``spillback_risk``.
    """
    cap = max(lane.length_m / pitch_m, 1.0)
    return {
        "queue_norm": float(np.clip(estimate.vehicle_count / cap, 0.0, 1.0)),
        "queue_m": estimate.queue_length_m,
        "confidence": estimate.confidence,
        "std_norm": float(np.clip(estimate.queue_std_m / lane.length_m, 0.0, 1.0)),
        "spillback_risk": spillback_risk(estimate.occupancy),
    }
