"""Reproducible synthetic benchmark: baselines, ablations and occlusion-stratified metrics.

Run ``oaqe-benchmark --quick`` for a smoke run or ``oaqe-benchmark --seeds 5 --jobs 4`` for the
full grid. Every scenario is generated from an explicit seed, and all estimator variants are
evaluated on the *same* detection stream, so comparisons are paired.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import oaqe
from oaqe.config import EstimatorConfig
from oaqe.evaluation import metrics as mt
from oaqe.pipeline import QueueEstimationPipeline
from oaqe.sim.synthetic import SceneConfig, SyntheticScene

LOG = logging.getLogger("oaqe.benchmark")
WARMUP_S = 60.0

_OFF = {"use_gap_inference": False, "use_tail_inference": False, "use_ghosts": False}
VARIANTS: dict[str, dict[str, object]] = {
    "naive_count": {"naive_count": True, "use_filter": False, **_OFF},
    "visible_chain": {"use_filter": False, **_OFF},
    "visible_chain+kf": dict(_OFF),
    "full-no_filter": {"use_filter": False},
    "full-no_ghosts": {"use_ghosts": False},
    "full-no_gap": {"use_gap_inference": False},
    "full-no_tail": {"use_tail_inference": False},
    "full": {},
}


@dataclass(frozen=True)
class ScenarioSpec:
    """One benchmark scenario."""

    name: str
    seed: int
    camera_height_m: float
    truck_fraction: float
    demand_vph: float
    duration_s: float


def run_scenario(spec: ScenarioSpec) -> list[dict[str, float | str | int]]:
    """Simulate one scenario and return per-tick, per-lane, per-variant rows."""
    scene = SyntheticScene(
        SceneConfig(
            seed=spec.seed, camera_height_m=spec.camera_height_m,
            truck_fraction=spec.truck_fraction, demand_vph=spec.demand_vph,
        )
    )
    hom, cam_pos = scene.estimator_inputs()
    base = EstimatorConfig()
    pipes = {
        name: QueueEstimationPipeline(scene.lanes, hom, cam_pos, replace(base, **over))
        for name, over in VARIANTS.items()
    }
    recent: deque[set[int]] = deque(maxlen=max(1, int(round(base.track_ttl_s * scene.cfg.fps))))
    rows: list[dict[str, float | str | int]] = []
    for _ in range(int(spec.duration_s * scene.cfg.fps)):
        frame = scene.step()
        recent.append(frame.detected_true_ids)
        seen = set().union(*recent)
        for name, pipe in pipes.items():
            for est in pipe.process_frame(frame.t, frame.detections):
                if frame.t < WARMUP_S:
                    continue
                idx = int(est.lane_id[1:])
                q_true, n_true, members = scene.truth_queue(idx, base.gap_max_m, base.head_tol_m)
                occ = (1.0 - len(members & seen) / len(members)) if members else float("nan")
                rows.append({
                    "scenario": spec.name, "seed": spec.seed, "variant": name, "lane": est.lane_id,
                    "t": round(frame.t, 3), "cam_h": spec.camera_height_m,
                    "truck_frac": spec.truck_fraction, "q_true": q_true,
                    "q_est": est.queue_length_m,
                    "q_lo": est.queue_ci95_m[0], "q_hi": est.queue_ci95_m[1],
                    "n_true": n_true, "n_est": est.vehicle_count,
                    "confidence": est.confidence, "occlusion_rate": occ,
                })
    return rows


def build_specs(seeds: int, duration_s: float, quick: bool, seed0: int = 0) -> list[ScenarioSpec]:
    """Scenario grid over camera height x truck share (lower camera = heavier occlusion)."""
    heights = [6.0, 4.0] if quick else [10.0, 6.0, 4.0]
    trucks = [0.30] if quick else [0.05, 0.30]
    return [
        ScenarioSpec(f"h{h:g}_t{int(tf * 100)}_s{seed0 + s}", seed0 + s, h, tf, 500.0, duration_s)
        for h in heights for tf in trucks for s in range(seeds)
    ]


def _fmt_ci(mean: float, ci: tuple[float, float]) -> str:
    return f"{mean:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-variant summary table with scenario-level bootstrap CIs on MAE."""
    per = mt.per_scenario_mae(df)
    rows = []
    for name in VARIANTS:
        d = df[df["variant"] == name]
        s = mt.error_summary(d)
        vals = per[per["variant"] == name]["mae_m"].to_numpy()
        dec = mt.decision_metrics(d)
        rows.append({
            "variant": name, "mae_m": s["mae_m"],
            "mae_ci": _fmt_ci(s["mae_m"], mt.bootstrap_ci(vals)),
            "rmse_m": s["rmse_m"], "bias_m": s["bias_m"], "count_mae": s["count_mae"],
            "coverage95": s["coverage95"], "ci_width_m": s["ci_width_m"],
            "top1": dec["top1"], "regret": dec["regret"],
        })
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame, reference: str = "full") -> pd.DataFrame:
    """Paired scenario-level MAE differences ``variant - reference`` with 95% bootstrap CIs."""
    per = mt.per_scenario_mae(df).pivot(index="variant", columns="scenario", values="mae_m")
    ref = per.loc[reference].to_numpy()
    rows = []
    for name in VARIANTS:
        if name == reference:
            continue
        diff, lo, hi = mt.paired_bootstrap_diff(per.loc[name].to_numpy(), ref)
        rows.append({"variant": name, "delta_mae_vs_full_m": diff, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    body = ""
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def make_figure(df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Two-panel figure: MAE vs occlusion rate, and 95% interval coverage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ["naive_count", "visible_chain", "visible_chain+kf", "full"]
    occ = mt.occlusion_binned(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in shown:
        d = occ[occ["variant"] == name]
        axes[0].errorbar(d["occ_bin"].astype(str), d["mae_m"], yerr=d["sem_m"], marker="o",
                         capsize=3, label=name)
    axes[0].set_xlabel("occlusion rate (fraction of queued vehicles undetected)")
    axes[0].set_ylabel("queue-length MAE [m]")
    axes[0].set_title("Error vs occlusion")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    s = summary[summary["variant"].isin(shown)]
    axes[1].bar(s["variant"], s["coverage95"])
    axes[1].axhline(0.95, color="k", ls="--", lw=1, label="nominal 0.95")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("empirical coverage of 95% interval")
    axes[1].set_title("Interval calibration")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``oaqe-benchmark``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds, duration = (1, 240.0) if args.quick else (args.seeds, args.duration)
    specs = build_specs(seeds, duration, args.quick, args.seed0)
    LOG.info("running %d scenarios x %d variants", len(specs), len(VARIANTS))
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_scenario, specs))
    else:
        results = [run_scenario(s) for s in specs]
    df = pd.DataFrame([r for rows in results for r in rows])
    df.to_csv(out / "ticks.csv", index=False)
    summary = summarise(df)
    paired = paired_table(df)
    occ = mt.occlusion_binned(df)
    summary.to_csv(out / "summary.csv", index=False)
    occ.to_csv(out / "occlusion_binned.csv", index=False)
    make_figure(df, summary, out / "error_vs_occlusion.png")
    shown = summary.drop(columns=["mae_m"]).rename(columns={"mae_ci": "MAE [95% CI] (m)"})
    occ_wide = occ.pivot(index="variant", columns="occ_bin", values="mae_m").reset_index()
    md = (
        "# Benchmark summary\n\n"
        f"Scenarios: {len(specs)}; variants: {len(VARIANTS)}; evaluation ticks per variant: "
        f"{int((df['variant'] == 'full').sum())}.\n\n## Overall\n\n{_md_table(shown)}\n"
        f"## Paired MAE difference vs `full` (positive = variant worse)\n\n{_md_table(paired)}\n"
        f"## MAE (m) by occlusion rate\n\n{_md_table(occ_wide)}"
    )
    (out / "summary.md").write_text(md, encoding="utf-8")
    manifest = {
        "package_version": oaqe.__version__, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "elapsed_s": round(time.time() - t0, 1),
        "specs": [asdict(s) for s in specs], "estimator_defaults": asdict(EstimatorConfig()),
        "scene_defaults": asdict(SceneConfig()), "variants": VARIANTS,
    }
    manifest_json = json.dumps(manifest, indent=2, default=str)
    (out / "manifest.json").write_text(manifest_json, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
