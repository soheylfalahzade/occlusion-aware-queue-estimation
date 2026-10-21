"""Single-frame queue observation: visible chain, hidden-gap inference and tail uncertainty."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from oaqe.config import EstimatorConfig
from oaqe.occlusion import OccluderBox, OcclusionModel
from oaqe.types import LaneSpec


@dataclass(frozen=True)
class StoppedVehicle:
    """A stopped vehicle in lane coordinates (``s_front_m`` = bumper nearest the stop line)."""

    s_front_m: float
    length_m: float
    cls: str = "car"
    ghost: bool = False


@dataclass(frozen=True)
class QueueObservation:
    """Result of one observation step (before temporal filtering)."""

    length_m: float
    variance_m2: float
    count: float
    count_var: float
    visible_fraction: float
    has_measurement: bool
    n_visible: int
    n_ghost: int
    n_hidden_inferred: float


def ground_truth_queue(
    s_front: np.ndarray,
    lengths: np.ndarray,
    speeds: np.ndarray,
    gap_max_m: float,
    head_tol_m: float,
    v_stop_mps: float = 1.0,
    max_length_m: float | None = None,
) -> tuple[float, int, list[int]]:
    """Reference queue definition used for evaluation.

    The queue is the chain of stopped vehicles starting at the stop line in which
    consecutive bumper-to-bumper gaps do not exceed ``gap_max_m`` (``head_tol_m`` for the
    first vehicle). The queue length is the position of the last member's rear bumper.

    Returns:
        ``(length_m, count, member_indices)`` with indices into the input arrays.
    """
    s_front, lengths, speeds = (np.asarray(a, dtype=float) for a in (s_front, lengths, speeds))
    order = [i for i in np.argsort(s_front) if s_front[i] >= -2.0 and speeds[i] < v_stop_mps]
    rear_prev, members = 0.0, []
    for k, i in enumerate(order):
        gap = s_front[i] - rear_prev
        if gap > (head_tol_m if k == 0 else gap_max_m):
            break
        members.append(int(i))
        rear_prev = max(rear_prev, s_front[i] + lengths[i])
    length = rear_prev if members else 0.0
    if max_length_m is not None:
        length = min(length, max_length_m)
    return float(length), len(members), members


def _hidden_run(grid: np.ndarray, hidden: np.ndarray, start: float, limit: float) -> float:
    """Length of the contiguous hidden stretch beginning at ``start`` (capped at ``limit``)."""
    idx = int(np.searchsorted(grid, start - 1e-9))
    step = float(grid[1] - grid[0])
    run = 0
    while idx + run < len(grid) and hidden[idx + run] and grid[idx + run] <= start + limit:
        run += 1
    return run * step


class QueueObserver:
    """Turns stopped-vehicle detections plus an occlusion model into a queue observation.

    Algorithm (per lane, per tick):

    1. Walk the stopped vehicles from the stop line upstream, linking neighbours whose gap
       is at most ``gap_max_m``.
    2. If a larger gap contains no visible empty stretch longer than ``gap_max_m`` but does
       contain hidden road that can hold vehicles, bridge it and infer
       ``floor(hidden_run / pitch + 0.4)`` vehicles per hidden run instead of ending the queue.
    3. Behind the last chain member, measure the contiguous hidden stretch ``U``. A queue
       that continues into it with probability ``p`` and is uniformly extended over
       ``[0, U]`` contributes mean ``p U / 2`` and variance ``U^2 (p/3 - p^2/4)``.
    4. If nothing is stopped and the stop-line area is hidden, no measurement is produced.
    """

    def __init__(self, cfg: EstimatorConfig, occlusion: OcclusionModel) -> None:
        self.cfg = cfg
        self.occlusion = occlusion

    def observe(
        self,
        lane: LaneSpec,
        stopped: Sequence[StoppedVehicle],
        occluders: Sequence[OccluderBox],
    ) -> QueueObservation:
        """Compute the queue observation for ``lane``."""
        cfg = self.cfg
        pitch = cfg.mean_vehicle_length_m + cfg.standstill_gap_m
        n_ghost = sum(v.ghost for v in stopped)
        if cfg.naive_count:
            n = float(len(stopped))
            return QueueObservation(
                n * pitch, (0.5 * pitch) ** 2, n, 1.0, 1.0, True, len(stopped), n_ghost, 0.0
            )

        grid = np.arange(0.0, lane.length_m + 0.5 * cfg.s_grid_step_m, cfg.s_grid_step_m)
        hidden = self.occlusion.hidden_mask(lane, grid, occluders)

        step = float(grid[1] - grid[0])

        def gap_vehicles(a: float, b: float) -> int | None:
            """Vehicles explained by hidden road in ``[a, b]``; ``None`` if the queue breaks.

            The chain may only bridge a gap if every *visible* stretch inside it is shorter
            than ``gap_max_m`` (no visible empty road) and the hidden stretches can hold at
            least one vehicle.
            """
            sel = (grid >= a) & (grid <= b)
            g, h = grid[sel], hidden[sel]
            n_inf, i = 0, 0
            while i < len(g):
                j = i
                while j < len(g) and h[j] == h[i]:
                    j += 1
                length = (j - i) * step
                if h[i]:
                    n_inf += int(np.floor(length / pitch + 0.4))
                elif length > cfg.gap_max_m:
                    return None
                i = j
            return n_inf if n_inf > 0 else None

        chain: list[StoppedVehicle] = []
        inferred, rear_prev = 0, 0.0
        for k, veh in enumerate(sorted(stopped, key=lambda v: v.s_front_m)):
            gap = veh.s_front_m - rear_prev
            tol = cfg.head_tol_m if k == 0 else cfg.gap_max_m
            if gap > tol:
                n_gap = gap_vehicles(rear_prev, veh.s_front_m) if cfg.use_gap_inference else None
                if n_gap is None:
                    break
                inferred += n_gap
            chain.append(veh)
            rear_prev = max(rear_prev, veh.s_front_m + veh.length_m)

        rear = rear_prev if chain else 0.0
        run = 0.0
        if cfg.use_tail_inference:
            run = _hidden_run(grid, hidden, rear, cfg.tail_lookahead_m)
        p = cfg.tail_continue_prob
        ext_mean = p * run / 2.0
        ext_var = run * run * (p / 3.0 - p * p / 4.0)

        span_end = max(rear, cfg.head_tol_m)
        vis_frac = 1.0 - float(hidden[grid <= span_end].mean())
        n_chain = len(chain)
        has_meas = not (n_chain == 0 and run >= cfg.head_tol_m)

        base_var = (cfg.meas_sigma_m if n_chain else cfg.empty_sigma_m) ** 2
        if chain and chain[-1].ghost:
            base_var += cfg.ghost_sigma_m**2
        return QueueObservation(
            length_m=rear + ext_mean,
            variance_m2=base_var + ext_var,
            count=n_chain + inferred + ext_mean / pitch,
            count_var=0.25 + 0.3 * inferred + ext_var / pitch**2,
            visible_fraction=vis_frac,
            has_measurement=has_meas,
            n_visible=n_chain - sum(v.ghost for v in chain),
            n_ghost=n_ghost,
            n_hidden_inferred=float(inferred) + ext_mean / pitch,
        )
