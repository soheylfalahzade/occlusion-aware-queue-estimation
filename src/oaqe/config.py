"""Configuration dataclasses and YAML loading."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from oaqe.types import LaneSpec


@dataclass
class EstimatorConfig:
    """Hyper-parameters of the queue estimator (all lengths in metres, times in seconds)."""

    output_period_s: float = 1.0
    # --- track management -------------------------------------------------------------
    track_ttl_s: float = 0.6
    ghost_ttl_s: float = 12.0
    dedup_radius_m: float = 3.0
    lane_tol_m: float = 0.6
    anchor: str = "front"  # which footprint point the bbox bottom-centre maps to
    # --- kinematics ---------------------------------------------------------------------
    kin_window_s: float = 2.0
    kin_min_history_s: float = 1.0
    v_stop_mps: float = 1.0
    v_go_mps: float = 2.0
    # --- queue model ----------------------------------------------------------------------
    gap_max_m: float = 8.0
    head_tol_m: float = 8.0
    mean_vehicle_length_m: float = 4.8
    standstill_gap_m: float = 2.0
    tail_lookahead_m: float = 30.0
    tail_continue_prob: float = 0.1
    probe_height_m: float = 0.9
    s_grid_step_m: float = 0.5
    # --- uncertainty and filtering ----------------------------------------------------------
    meas_sigma_m: float = 1.0
    empty_sigma_m: float = 0.6
    ghost_sigma_m: float = 1.5
    process_accel_sigma: float = 4.0
    count_process_sigma: float = 0.6
    gate_chi2: float = 9.0
    confidence_ref_std_m: float = 3.0
    # --- ablation switches ------------------------------------------------------------------
    naive_count: bool = False
    use_gap_inference: bool = True
    use_tail_inference: bool = True
    use_ghosts: bool = True
    use_filter: bool = True

    def __post_init__(self) -> None:
        if self.anchor not in {"front", "center", "rear"}:
            raise ValueError(f"anchor must be front|center|rear, got {self.anchor!r}")
        if self.v_go_mps <= self.v_stop_mps:
            raise ValueError("v_go_mps must exceed v_stop_mps (hysteresis)")
        if not 0.0 <= self.tail_continue_prob <= 1.0:
            raise ValueError("tail_continue_prob must lie in [0, 1]")


@dataclass
class SiteConfig:
    """Camera/site description needed to instantiate the pipeline."""

    camera_position_m: tuple[float, float, float]
    image_points_px: list[list[float]]
    ground_points_m: list[list[float]]
    lanes: list[LaneSpec]
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    image_size_px: tuple[int, int] = (1920, 1080)


def _build_estimator(raw: dict[str, Any] | None) -> EstimatorConfig:
    raw = raw or {}
    allowed = {f.name for f in fields(EstimatorConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown estimator keys: {sorted(unknown)}")
    return EstimatorConfig(**raw)


def load_site_config(path: str | Path) -> SiteConfig:
    """Load a :class:`SiteConfig` from a YAML file (see ``configs/site_template.yaml``)."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    lanes = [
        LaneSpec(
            lane_id=str(item["lane_id"]),
            stop_point_m=tuple(item["stop_point_m"]),
            direction=tuple(item["direction"]),
            width_m=float(item.get("width_m", 3.5)),
            length_m=float(item.get("length_m", 100.0)),
            approach_id=str(item.get("approach_id", "A")),
        )
        for item in raw["lanes"]
    ]
    return SiteConfig(
        camera_position_m=tuple(raw["camera"]["position_m"]),
        image_size_px=tuple(raw["camera"].get("image_size_px", (1920, 1080))),
        image_points_px=raw["calibration"]["image_points_px"],
        ground_points_m=raw["calibration"]["ground_points_m"],
        lanes=lanes,
        estimator=_build_estimator(raw.get("estimator")),
    )
