"""Detector adapters. The estimator itself is detector-agnostic (see ``Detection``)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from oaqe.types import Detection

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class Detector(Protocol):
    """Anything that maps a BGR frame to tracked detections."""

    def __call__(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - protocol
        ...


class YoloByteTrackDetector:
    """Ultralytics YOLO (v8/v10/11) with the built-in ByteTrack tracker.

    No training is performed: pretrained COCO weights are used and only the four
    vehicle classes are kept. Requires the ``video`` extra (``pip install .[video]``).
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        imgsz: int = 1280,
        device: str | None = None,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install the 'video' extra: pip install -e '.[video]'") from exc
        self._model = YOLO(weights)
        self._kw = {
            "persist": True, "tracker": tracker, "conf": conf, "imgsz": imgsz,
            "classes": list(COCO_VEHICLES), "verbose": False,
        }
        if device is not None:
            self._kw["device"] = device

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        result = self._model.track(frame, **self._kw)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        return [
            Detection(int(i), COCO_VEHICLES.get(int(c), "car"),
                      (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(s))
            for i, c, b, s in zip(ids, cls, xyxy, conf, strict=True)
        ]
