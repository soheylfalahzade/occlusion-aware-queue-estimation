"""Command-line entry points."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from oaqe.config import load_site_config
from oaqe.pipeline import QueueEstimationPipeline

LOG = logging.getLogger("oaqe")


def main_video(argv: list[str] | None = None) -> int:
    """Run the estimator on a video file: detector + ByteTrack -> lane estimates (CSV)."""
    ap = argparse.ArgumentParser(description=main_video.__doc__)
    ap.add_argument("--config", required=True, help="site YAML (see configs/site_template.yaml)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--out", default="results/video_estimates.csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import cv2  # local import: optional dependency

    from oaqe.detection import YoloByteTrackDetector

    pipe = QueueEstimationPipeline.from_site_config(load_site_config(args.config))
    detector = YoloByteTrackDetector(args.weights, args.conf, args.imgsz, args.device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer: csv.DictWriter | None = None
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            for est in pipe.process_frame(n / fps, detector(frame)):
                row = est.as_dict()
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            n += 1
    LOG.info("processed %d frames at %.2f fps -> %s", n, fps, args.out)
    return 0


def score_against_labels(estimates_csv: str, labels_csv: str, tol_s: float = 1.0,
                         n_boot: int = 2000, seed: int = 0) -> dict[str, float | int]:
    """Compare video estimates with manual labels (the only way to score real footage).

    ``labels_csv`` has columns ``timestamp_s, lane_id, queue_length_m`` (and optionally
    ``vehicle_count``). Each label is matched to the nearest estimate of the same lane within
    ``tol_s`` seconds. Returns MAE, bias and 95% interval coverage with bootstrap intervals over
    labelled instants.
    """
    import numpy as np
    import pandas as pd

    est = pd.read_csv(estimates_csv)
    lab = pd.read_csv(labels_csv)
    rows = []
    for _, r in lab.iterrows():
        e = est[est["lane_id"].astype(str) == str(r["lane_id"])]
        if e.empty:
            continue
        k = (e["timestamp_s"] - r["timestamp_s"]).abs().idxmin()
        if abs(e.loc[k, "timestamp_s"] - r["timestamp_s"]) > tol_s:
            continue
        rows.append((float(e.loc[k, "queue_length_m"]) - float(r["queue_length_m"]),
                     float(e.loc[k, "queue_ci95_lo_m"] <= r["queue_length_m"]
                           <= e.loc[k, "queue_ci95_hi_m"])))
    if not rows:
        raise SystemExit("no label could be matched to an estimate (check lane_id / timestamps)")
    err = np.array([x[0] for x in rows])
    cov = np.array([x[1] for x in rows])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(err), size=(n_boot, len(err)))
    mae = np.abs(err[idx]).mean(axis=1)
    return {"n_matched": len(err), "n_labels": int(len(lab)), "mae_m": float(np.abs(err).mean()),
            "mae_ci95_lo": float(np.quantile(mae, 0.025)),
            "mae_ci95_hi": float(np.quantile(mae, 0.975)), "bias_m": float(err.mean()),
            "coverage95": float(cov.mean())}


def main_score(argv: list[str] | None = None) -> int:
    """Score ``oaqe-video`` output against manually labelled queue lengths."""
    import json

    ap = argparse.ArgumentParser(description=main_score.__doc__)
    ap.add_argument("--estimates", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tol-s", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = score_against_labels(args.estimates, args.labels, args.tol_s)
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0
