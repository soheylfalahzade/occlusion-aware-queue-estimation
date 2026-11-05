"""Helpers to label queue lengths on real footage (blind to the estimator).

``oaqe-label-frames`` writes frames of a video with the lane centre lines and distance ticks
measured from the stop line (from the site calibration), plus an empty ``labels_template.csv``.
The annotator reads the queue length of each lane from the ticks and fills the CSV; rows left
empty are dropped by ``--finalize`` and the result goes to ``oaqe-score``. No estimate is drawn,
so the labels do not depend on the estimator.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from oaqe.config import SiteConfig, load_site_config
from oaqe.geometry import Homography


def lane_ticks(site: SiteConfig,
               spacing_m: float = 5.0) -> dict[str, list[tuple[float, np.ndarray]]]:
    """Ground points of the centre line of each lane every ``spacing_m`` from the stop line."""
    out: dict[str, list[tuple[float, np.ndarray]]] = {}
    for lane in site.lanes:
        d = np.asarray(lane.direction, dtype=float)
        d = d / np.linalg.norm(d)
        p0 = np.asarray(lane.stop_point_m, dtype=float)
        s_values = np.arange(0.0, lane.length_m + 1e-9, spacing_m)
        out[lane.lane_id] = [(float(s), p0 + s * d) for s in s_values]
    return out


def template_rows(lane_ids: list[str], timestamps: list[float]) -> list[dict[str, str | float]]:
    """One empty label row per labelled instant and lane."""
    return [{"timestamp_s": round(t, 2), "lane_id": lane, "queue_length_m": ""}
            for t in timestamps for lane in lane_ids]


def finalize_labels(template_csv: str, out_csv: str) -> int:
    """Keep the rows of the template with a queue length; return how many were kept."""
    import pandas as pd

    df = pd.read_csv(template_csv)
    df = df[pd.to_numeric(df["queue_length_m"], errors="coerce").notna()].copy()
    df["queue_length_m"] = df["queue_length_m"].astype(float)
    if (df["queue_length_m"] < 0).any():
        raise SystemExit("negative queue length in the labels")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return len(df)


def draw_overlay(frame: np.ndarray, site: SiteConfig, hom: Homography,
                 spacing_m: float) -> np.ndarray:
    """Draw centre lines, ticks and distance labels on a copy of ``frame``."""
    import cv2

    img = frame.copy()
    for lane_id, ticks in lane_ticks(site, spacing_m).items():
        px = hom.ground_to_image(np.array([g for _, g in ticks]))
        pts = px.round().astype(int)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)
        for (s, _), (x, y) in zip(ticks, pts, strict=True):
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
            if s > 0 and round(s) % 10 == 0:
                cv2.putText(img, f"{s:g}", (int(x) + 5, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f"{s:g}", (int(x) + 5, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 1, cv2.LINE_AA)
        x0, y0 = pts[0]
        cv2.putText(img, lane_id, (int(x0) - 10, int(y0) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def main_label_frames(argv: list[str] | None = None) -> int:
    """Write frames with distance ticks and an empty label template, or finalize the labels."""
    ap = argparse.ArgumentParser(description=main_label_frames.__doc__)
    ap.add_argument("--config")
    ap.add_argument("--video")
    ap.add_argument("--every-s", type=float, default=20.0, help="spacing of the labelled instants")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=0.0, help="0 = end of the video")
    ap.add_argument("--tick-m", type=float, default=5.0)
    ap.add_argument("--out-dir", default="label_frames")
    ap.add_argument("--finalize", metavar="TEMPLATE_CSV",
                    help="drop empty rows of a filled template and write --labels-out")
    ap.add_argument("--labels-out", default="labels.csv")
    args = ap.parse_args(argv)
    if args.finalize:
        n = finalize_labels(args.finalize, args.labels_out)
        print(f"{n} labelled rows -> {args.labels_out}")
        return 0
    if not (args.config and args.video):
        raise SystemExit("--config and --video are required")
    import cv2

    site = load_site_config(args.config)
    hom = Homography.from_correspondences(site.image_points_px, site.ground_points_m)
    rmse = hom.ground_rmse_m(site.image_points_px, site.ground_points_m)
    print(f"calibration ground RMSE: {rmse:.2f} m")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = args.end_s if args.end_s > 0 else duration
    stamps = [float(t) for t in np.arange(args.start_s, end, args.every_s)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept: list[float] = []
    for t in stamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out / f"t{t:08.2f}.png"), draw_overlay(frame, site, hom, args.tick_m))
        kept.append(t)
    rows = template_rows([lane.lane_id for lane in site.lanes], kept)
    tpl = out / "labels_template.csv"
    with open(tpl, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp_s", "lane_id", "queue_length_m"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(kept)} frames and {tpl} written to {out}")
    return 0
