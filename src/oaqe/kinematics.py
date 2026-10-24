"""Per-track ground-plane kinematics with hysteretic stop detection."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import ArrayLike


class TrackKinematics:
    """Sliding-window least-squares velocity and a stopped/moving state machine.

    A track becomes *stopped* when its speed drops below ``v_stop`` and returns to
    *moving* only when it exceeds ``v_go`` (hysteresis suppresses chatter caused by
    localisation noise). The state persists while the track is not observed, which
    is what allows stopped vehicles to survive short occlusions.
    """

    def __init__(
        self,
        track_id: int,
        cls: str,
        lane_id: str | None,
        window_s: float = 2.0,
        min_history_s: float = 1.0,
        v_stop: float = 1.0,
        v_go: float = 2.0,
    ) -> None:
        self.track_id = track_id
        self.cls = cls
        self.lane_id = lane_id
        self.window_s = window_s
        self.min_history_s = min_history_s
        self.v_stop = v_stop
        self.v_go = v_go
        self.stopped = False
        self.last_seen = -math.inf
        self._hist: deque[tuple[float, float, float]] = deque()

    def update(self, t: float, xy: ArrayLike, cls: str, lane_id: str | None) -> None:
        """Add an observation at time ``t`` (seconds) and refresh the motion state."""
        p = np.asarray(xy, dtype=float)
        self._hist.append((t, float(p[0]), float(p[1])))
        while self._hist and t - self._hist[0][0] > self.window_s:
            self._hist.popleft()
        self.cls, self.lane_id, self.last_seen = cls, lane_id, t
        self._refresh_state()

    def velocity(self) -> np.ndarray | None:
        """Least-squares ground velocity (m/s), or ``None`` without enough history."""
        if len(self._hist) < 3:
            return None
        arr = np.asarray(self._hist)
        if arr[-1, 0] - arr[0, 0] < self.min_history_s:
            return None
        tt = arr[:, 0] - arr[:, 0].mean()
        denom = float((tt**2).sum())
        if denom < 1e-9:
            return None
        vx = float((tt * (arr[:, 1] - arr[:, 1].mean())).sum() / denom)
        vy = float((tt * (arr[:, 2] - arr[:, 2].mean())).sum() / denom)
        return np.array([vx, vy])

    def speed(self) -> float:
        """Scalar speed in m/s (NaN if undetermined)."""
        v = self.velocity()
        return float("nan") if v is None else float(np.hypot(*v))

    def _refresh_state(self) -> None:
        v = self.velocity()
        if v is None:
            return
        speed = float(np.hypot(*v))
        if self.stopped and speed > self.v_go:
            self.stopped = False
        elif not self.stopped and speed < self.v_stop:
            self.stopped = True

    def position(self) -> np.ndarray:
        """Best ground-plane position: window mean if stopped, latest fix otherwise."""
        arr = np.asarray(self._hist)
        if self.stopped and len(arr) > 1:
            return arr[:, 1:3].mean(axis=0)
        return arr[-1, 1:3].copy()
