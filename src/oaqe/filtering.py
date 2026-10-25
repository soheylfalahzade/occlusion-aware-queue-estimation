"""Scalar constant-velocity Kalman filter with innovation gating."""
from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """Tracks a bounded scalar (queue length or count) and its rate of change.

    The state is ``[value, rate]`` with a discrete white-noise-acceleration process
    model. Measurements whose normalised innovation exceeds ``gate_chi2`` have their
    noise variance inflated in proportion, which bounds the influence of outliers
    without discarding genuine regime changes (e.g. the start of a discharge wave).
    """

    def __init__(
        self,
        accel_sigma: float,
        rate_sigma0: float = 2.0,
        gate_chi2: float = 9.0,
        lower: float = 0.0,
        upper: float | None = None,
    ) -> None:
        self._q = accel_sigma**2
        self._rate_var0 = rate_sigma0**2
        self._gate = gate_chi2
        self._lower, self._upper = lower, upper
        self.x: np.ndarray = np.zeros(2)
        self.p: np.ndarray = np.diag([1e2, self._rate_var0])

    def reset(self, value: float, var: float) -> None:
        """(Re-)initialise the state at ``value`` with variance ``var``."""
        self.x = np.array([value, 0.0])
        self.p = np.diag([max(var, 1e-6), self._rate_var0])
        self._clip()

    def predict(self, dt: float) -> None:
        """Propagate the state ``dt`` seconds forward."""
        if dt <= 0.0:
            return
        f = np.array([[1.0, dt], [0.0, 1.0]])
        g = np.array([0.5 * dt * dt, dt])
        self.p = f @ self.p @ f.T + self._q * np.outer(g, g)
        self.x = f @ self.x
        self._clip()

    def update(self, z: float, r: float) -> None:
        """Fuse a measurement ``z`` with variance ``r``."""
        r = max(r, 1e-6)
        y = z - self.x[0]
        s = self.p[0, 0] + r
        d2 = y * y / s
        if d2 > self._gate:
            r *= d2 / self._gate
            s = self.p[0, 0] + r
        k = self.p[:, 0] / s
        self.x = self.x + k * y
        i_kh = np.eye(2) - np.outer(k, [1.0, 0.0])
        self.p = i_kh @ self.p @ i_kh.T + r * np.outer(k, k)
        self._clip()

    @property
    def mean(self) -> float:
        """Current value estimate."""
        return float(self.x[0])

    @property
    def std(self) -> float:
        """Standard deviation of the value estimate."""
        return float(np.sqrt(max(self.p[0, 0], 0.0)))

    def _clip(self) -> None:
        if self.x[0] < self._lower:
            self.x[0], self.x[1] = self._lower, max(self.x[1], 0.0)
        if self._upper is not None and self.x[0] > self._upper:
            self.x[0], self.x[1] = self._upper, min(self.x[1], 0.0)
