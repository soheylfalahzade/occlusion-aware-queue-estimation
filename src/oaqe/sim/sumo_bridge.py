"""Optional SUMO/TraCI bridge: render SUMO vehicles through the synthetic camera and estimate.

For every simulation step the vehicles on the monitored lanes are read through TraCI,
rendered as tracker-like detections by :class:`~oaqe.sim.synthetic.CameraRenderer`, and fed
to the estimator. Estimates are logged next to SUMO's own halting statistics so that the
estimator can later replace ground-truth queue inputs of a signal controller.

Requires SUMO and the ``sumo`` extra (``pip install -e '.[sumo]'``). All lanes must belong to
one straight edge (parallel lanes, ordered from the lane nearest the camera).
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from oaqe.config import EstimatorConfig
from oaqe.pipeline import QueueEstimationPipeline
from oaqe.queue_model import ground_truth_queue
from oaqe.sim.synthetic import CameraRenderer, SceneConfig, SimVehicle, SyntheticScene

LOG = logging.getLogger("oaqe.sumo")
_CLASS_MAP = {"passenger": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle"}


def _read_vehicles(traci, lane_ids: list[str], cfg: SceneConfig) -> list[SimVehicle]:
    vehicles: list[SimVehicle] = []
    for idx, lane_id in enumerate(lane_ids):
        lane_len = traci.lane.getLength(lane_id)
        for vid in traci.lane.getLastStepVehicleIDs(lane_id):
            length = traci.vehicle.getLength(vid)
            cls = _CLASS_MAP.get(traci.vehicle.getVehicleClass(vid), "car")
            height = {"car": 1.5, "truck": 3.0, "bus": 3.2, "motorcycle": 1.4}[cls]
            width = {"car": 1.8, "truck": 2.5, "bus": 2.5, "motorcycle": 0.8}[cls]
            vehicles.append(
                SimVehicle(
                    vid=hash(vid) & 0x7FFFFFFF, lane_idx=idx, y_m=idx * cfg.lane_width_m,
                    s_front_m=lane_len - traci.vehicle.getLanePosition(vid),
                    speed_mps=traci.vehicle.getSpeed(vid), cls=cls, length_m=length,
                    width_m=width, height_m=height,
                )
            )
    return vehicles


def run(sumo_cfg: str, lane_ids: list[str], steps: int, out_csv: str, camera_height_m: float,
        seed: int = 0, sumo_binary: str = "sumo") -> None:
    """Run SUMO headless and log camera-based estimates against SUMO ground truth."""
    import traci  # optional dependency

    scene_cfg = SceneConfig(n_lanes=len(lane_ids), camera_height_m=camera_height_m, seed=seed)
    scene = SyntheticScene(scene_cfg)  # used only for camera, calibration and lane geometry
    hom, cam_pos = scene.estimator_inputs()
    renderer = CameraRenderer(scene.camera, scene_cfg, np.random.default_rng(seed + 1))
    est_cfg = EstimatorConfig()
    pipe = QueueEstimationPipeline(scene.lanes, hom, cam_pos, est_cfg)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    traci.start([sumo_binary, "-c", sumo_cfg, "--seed", str(seed), "--no-step-log", "true"])
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t", "lane", "q_est_m", "q_std_m", "count_est", "q_true_m",
                             "count_true", "sumo_halting"])
            for _ in range(steps):
                traci.simulationStep()
                t = traci.simulation.getTime()
                vehicles = _read_vehicles(traci, lane_ids, scene_cfg)
                dets, _ = renderer.render(t, vehicles)
                for est in pipe.process_frame(t, dets):
                    idx = int(est.lane_id[1:])
                    mine = [v for v in vehicles if v.lane_idx == idx]
                    s = np.array([v.s_front_m for v in mine])
                    q, n, _ = ground_truth_queue(
                        s, np.array([v.length_m for v in mine]),
                        np.array([v.speed_mps for v in mine]), est_cfg.gap_max_m,
                        est_cfg.head_tol_m, 1.0, scene_cfg.lane_length_m,
                    ) if mine else (0.0, 0, [])
                    writer.writerow([
                        f"{t:.2f}", lane_ids[idx], f"{est.queue_length_m:.2f}",
                        f"{est.queue_std_m:.2f}", f"{est.vehicle_count:.2f}", f"{q:.2f}", n,
                        traci.lane.getLastStepHaltingNumber(lane_ids[idx]),
                    ])
    finally:
        traci.close()
    LOG.info("wrote %s", out_csv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``oaqe-sumo``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sumocfg", required=True)
    ap.add_argument("--lanes", nargs="+", required=True, help="lane IDs, camera-near first")
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--camera-height", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sumo_estimates.csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.sumocfg, args.lanes, args.steps, args.out, args.camera_height, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
