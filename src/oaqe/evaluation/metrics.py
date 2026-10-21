"""Evaluation metrics: error statistics, interval calibration, bootstrap CIs, decisions."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

OCCLUSION_BINS = (0.0, 0.1, 0.25, 0.5, 1.0001)
OCCLUSION_LABELS = ("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1]")


def error_summary(df: pd.DataFrame) -> dict[str, float]:
    """Point and interval metrics for per-tick rows.

    Expected columns: ``q_true, q_est, q_lo, q_hi, n_true, n_est``.
    """
    err = df["q_est"] - df["q_true"]
    covered = (df["q_true"] >= df["q_lo"]) & (df["q_true"] <= df["q_hi"])
    return {
        "n_ticks": float(len(df)),
        "mae_m": float(err.abs().mean()),
        "rmse_m": float(np.sqrt((err**2).mean())),
        "bias_m": float(err.mean()),
        "count_mae": float((df["n_est"] - df["n_true"]).abs().mean()),
        "coverage95": float(covered.mean()),
        "ci_width_m": float((df["q_hi"] - df["q_lo"]).mean()),
    }


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``stat`` over i.i.d. units (scenarios)."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean paired difference ``a - b`` with a 95% bootstrap CI (aligned per-scenario values)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d, np.mean, n_boot, 0.05, seed)
    return float(d.mean()), lo, hi


def per_scenario_mae(df: pd.DataFrame) -> pd.DataFrame:
    """MAE of queue length per (variant, scenario) pair."""
    g = df.assign(abs_err=(df["q_est"] - df["q_true"]).abs()).groupby(["variant", "scenario"])
    return g["abs_err"].mean().rename("mae_m").reset_index()


def occlusion_binned(df: pd.DataFrame) -> pd.DataFrame:
    """MAE as a function of the fraction of queued vehicles that are currently undetected."""
    d = df.dropna(subset=["occlusion_rate"]).copy()
    d["occ_bin"] = pd.cut(
        d["occlusion_rate"], list(OCCLUSION_BINS), labels=list(OCCLUSION_LABELS), right=False
    )
    d["abs_err"] = (d["q_est"] - d["q_true"]).abs()
    out = d.groupby(["variant", "occ_bin"], observed=True)["abs_err"].agg(["mean", "sem", "size"])
    return out.rename(columns={"mean": "mae_m", "sem": "sem_m", "size": "n"}).reset_index()


def decision_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Max-pressure-style lane ranking quality from per-tick, per-lane counts.

    For each tick with at least one queued vehicle the lane with the largest estimated
    count is compared with the true ranking. ``top1`` is the fraction of ticks where the
    chosen lane is (one of) the true maximum(s); ``regret`` is the mean relative shortfall
    of the chosen lane's true count against the maximum true count.
    """
    hits, regrets = [], []
    for _, tick in df.groupby(["scenario", "t"]):
        if tick["n_true"].max() <= 0 or len(tick) < 2:
            continue
        chosen = tick.loc[tick["n_est"].idxmax()]
        best = tick["n_true"].max()
        hits.append(float(chosen["n_true"] >= best))
        regrets.append(float((best - chosen["n_true"]) / best))
    if not hits:
        return {"top1": float("nan"), "regret": float("nan")}
    return {"top1": float(np.mean(hits)), "regret": float(np.mean(regrets))}
