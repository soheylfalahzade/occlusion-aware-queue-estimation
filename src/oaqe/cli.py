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
