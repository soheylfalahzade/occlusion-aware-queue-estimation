"""Data-driven animation: true queue, camera-visible vehicles and the estimator output."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from oaqe.config import EstimatorConfig
from oaqe.pipeline import QueueEstimationPipeline
from oaqe.sim.synthetic import SceneConfig, SyntheticScene


def render_queue_gif(out: str | Path, seed: int = 4, camera_height_m: float = 4.0,
                     truck_fraction: float = 0.3, t_start: float = 150.0, n_frames: int = 60,
                     fps: int = 5) -> dict[str, Any]:
    """Simulate a heavily occluded approach and animate one lane at 1 Hz.

    Top: road plane (stop line at x = 0). Vehicles detected within the last 0.6 s are green,
    undetected ones are red; the black tick is the true queue end, the blue bar the full
    estimator and the orange tick the visible-chain baseline. Bottom: queue length over time.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    from PIL import Image

    scene = SyntheticScene(SceneConfig(seed=seed, camera_height_m=camera_height_m,
                                       truck_fraction=truck_fraction))
    hom, cam = scene.estimator_inputs()
    base = EstimatorConfig()
    full = QueueEstimationPipeline(scene.lanes, hom, cam, base)
    naive = QueueEstimationPipeline(scene.lanes, hom, cam, replace(
        base, use_gap_inference=False, use_tail_inference=False, use_ghosts=False,
        use_filter=False))
    recent: list[set[int]] = []
    keep = int(round(base.track_ttl_s * scene.cfg.fps))
    ticks: list[dict[str, Any]] = []
    while scene.traffic.t < t_start + n_frames:
        fr = scene.step()
        recent = (recent + [fr.detected_true_ids])[-keep:]
        ef, en = full.process_frame(fr.t, fr.detections), naive.process_frame(fr.t, fr.detections)
        if ef and fr.t >= t_start:
            seen = set().union(*recent)
            snap = [(v.lane_idx, v.s_front_m, v.length_m, v.vid in seen) for v in fr.vehicles]
            truth = [scene.truth_queue(i)[0] for i in range(scene.cfg.n_lanes)]
            ticks.append({"t": fr.t, "veh": snap, "truth": truth,
                          "full": {e.lane_id: e for e in ef}, "naive": {e.lane_id: e for e in en}})
    lane = scene.cfg.n_lanes - 1
    lid = f"L{lane}"
    t0 = ticks[0]["t"]
    images = []
    for k, tk in enumerate(ticks):
        fig, (a, b) = plt.subplots(2, 1, figsize=(8.0, 4.8), dpi=72,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
        for li in range(scene.cfg.n_lanes):
            y = li * 3.5
            a.add_patch(Rectangle((0, y - 1.75), 100, 3.5, fc="#f2f2f2", ec="#bbbbbb", lw=0.6))
            for (vl, s0, ln, seen_flag) in tk["veh"]:
                if vl == li:
                    a.add_patch(Rectangle((s0, y - 0.8), ln, 1.6,
                                          fc="#7bc77b" if seen_flag else "#e57373", ec="k", lw=0.4))
            a.plot([tk["truth"][li]] * 2, [y - 1.6, y + 1.6], color="k", lw=2.2)
            en = tk["full"][f"L{li}"]
            a.plot([0, en.queue_length_m], [y + 1.25, y + 1.25], color="#1f77b4", lw=3)
            a.plot([en.queue_ci95_m[0], en.queue_ci95_m[1]], [y + 1.5, y + 1.5],
                   color="#1f77b4", lw=1)
            nv = tk["naive"][f"L{li}"].queue_length_m
            a.plot([nv] * 2, [y - 1.6, y - 0.2], color="#ff7f0e", lw=2.2)
        a.axvline(0, color="k", lw=1)
        a.set_xlim(-3, 100)
        a.set_ylim(-2.2, 3.5 * scene.cfg.n_lanes - 1.0)
        a.set_yticks([])
        a.set_xlabel("distance from stop line [m]   green = detected, red = undetected; "
                     "black = true queue end; blue = estimate (95% CI); orange = visible-chain",
                     fontsize=6.5)
        a.set_title(f"t = {tk['t'] - t0:4.0f} s   camera height {camera_height_m:g} m, "
                    f"{int(truck_fraction * 100)}% trucks", fontsize=9)
        ts = [x["t"] - t0 for x in ticks[: k + 1]]
        b.plot(ts, [x["truth"][lane] for x in ticks[: k + 1]], color="k", lw=1.8, label="true")
        b.plot(ts, [x["naive"][lid].queue_length_m for x in ticks[: k + 1]], color="#ff7f0e",
               lw=1.2, label="visible chain")
        qs = np.array([x["full"][lid].queue_length_m for x in ticks[: k + 1]])
        lo = np.array([x["full"][lid].queue_ci95_m[0] for x in ticks[: k + 1]])
        hi = np.array([x["full"][lid].queue_ci95_m[1] for x in ticks[: k + 1]])
        b.fill_between(ts, lo, hi, color="#1f77b4", alpha=0.18)
        b.plot(ts, qs, color="#1f77b4", lw=1.8, label="occlusion-aware (95% CI)")
        b.set_xlim(0, ticks[-1]["t"] - t0)
        b.set_ylim(0, 100)
        b.set_ylabel(f"queue length, lane {lid} [m]", fontsize=8)
        b.set_xlabel("time [s]", fontsize=8)
        b.tick_params(labelsize=7)
        b.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        canvas = cast(FigureCanvasAgg, fig.canvas)
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
                      .quantize(colors=96))
        plt.close(fig)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return {"frames": len(images)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``oaqe-gif``)."""
    import argparse

    ap = argparse.ArgumentParser(description=render_queue_gif.__doc__)
    ap.add_argument("--out", default="docs/assets/queue.gif")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(render_queue_gif(args.out, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
