"""End-to-end lane-level queue estimation from tracked detections."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from oaqe.config import EstimatorConfig, SiteConfig
from oaqe.filtering import ConstantVelocityKalman
from oaqe.geometry import Homography
from oaqe.kinematics import TrackKinematics
from oaqe.occlusion import OccluderBox, OcclusionModel
from oaqe.queue_model import QueueObservation, QueueObserver, StoppedVehicle
from oaqe.types import Detection, LaneEstimate, LaneSpec, prior_for


@dataclass
class _Vehicle:
    lane_id: str
    approach_id: str
    cls: str
    xy: np.ndarray
    stopped: bool
    ghost: bool
    length_m: float
    width_m: float
    height_m: float


@dataclass
class _LaneFilter:
    kf_len: ConstantVelocityKalman
    kf_cnt: ConstantVelocityKalman
    t_last: float


class QueueEstimationPipeline:
    """Consumes per-frame tracked detections and emits lane estimates at a fixed rate.

    Frame-rate work is limited to localisation and kinematics. Occlusion reasoning,
    ghost bookkeeping, observation and filtering run once per output period.
    """

    def __init__(
        self,
        lanes: Sequence[LaneSpec],
        homography: Homography,
        camera_position_m: tuple[float, float, float],
        config: EstimatorConfig | None = None,
    ) -> None:
        self.cfg = config or EstimatorConfig()
        self.lanes = {ln.lane_id: ln for ln in lanes}
        self.homography = homography
        self.occlusion = OcclusionModel(camera_position_m, self.cfg.probe_height_m)
        self.observer = QueueObserver(self.cfg, self.occlusion)
        self._tracks: dict[int, TrackKinematics] = {}
        self._filters: dict[str, _LaneFilter] = {}
        self._raw: dict[str, tuple[float, float, float, float]] = {}
        self._next_tick: float | None = None

    @classmethod
    def from_site_config(cls, site: SiteConfig) -> QueueEstimationPipeline:
        """Build a pipeline from a :class:`~oaqe.config.SiteConfig`."""
        hom = Homography.from_correspondences(site.image_points_px, site.ground_points_m)
        return cls(site.lanes, hom, site.camera_position_m, site.estimator)

    def reset(self) -> None:
        """Forget all tracks and filter state."""
        self._tracks.clear()
        self._filters.clear()
        self._raw.clear()
        self._next_tick = None

    # ------------------------------------------------------------------ frame ingestion
    def process_frame(self, t: float, detections: Sequence[Detection]) -> list[LaneEstimate]:
        """Ingest one frame; returns lane estimates when an output tick is due, else ``[]``."""
        self._ingest(t, detections)
        period = self.cfg.output_period_s
        if self._next_tick is None:
            self._next_tick = t + period
            return []
        if t + 1e-9 < self._next_tick:
            return []
        while self._next_tick <= t + 1e-9:
            self._next_tick += period
        return self._tick(t)

    def _assign_lane(self, xy: np.ndarray) -> str | None:
        best, best_l = None, math.inf
        for lane in self.lanes.values():
            s, off = lane.to_lane(xy)
            inside = -3.0 <= s <= lane.length_m + 3.0
            if inside and abs(off) <= lane.width_m / 2 + self.cfg.lane_tol_m and abs(off) < best_l:
                best, best_l = lane.lane_id, abs(off)
        return best

    def _ingest(self, t: float, detections: Sequence[Detection]) -> None:
        if not detections:
            return
        px = np.array([d.ground_anchor_px for d in detections], dtype=float)
        ground = self.homography.image_to_ground(px)
        for det, xy in zip(detections, ground, strict=True):
            if not np.all(np.isfinite(xy)):
                continue
            lane_id = self._assign_lane(xy)
            if lane_id is None:
                continue
            kin = self._tracks.get(det.track_id)
            if kin is None:
                kin = self._relink(t, xy, lane_id) if self.cfg.use_ghosts else None
                if kin is None:
                    c = self.cfg
                    kin = TrackKinematics(
                        det.track_id, det.cls, lane_id, c.kin_window_s, c.kin_min_history_s,
                        c.v_stop_mps, c.v_go_mps,
                    )
                kin.track_id = det.track_id
                self._tracks[det.track_id] = kin
            kin.update(t, xy, det.cls, lane_id)

    def _relink(self, t: float, xy: np.ndarray, lane_id: str) -> TrackKinematics | None:
        """Inherit the state of a nearby stopped ghost when the tracker issues a new ID."""
        lane = self.lanes[lane_id]
        s_new, l_new = lane.to_lane(xy)
        found: int | None = None
        for tid, kin in self._tracks.items():
            age = t - kin.last_seen
            if not (self.cfg.track_ttl_s < age <= self.cfg.ghost_ttl_s):
                continue
            if not kin.stopped or kin.lane_id != lane_id:
                continue
            s_old, l_old = lane.to_lane(kin.position())
            if abs(s_new - s_old) <= self.cfg.dedup_radius_m and abs(l_new - l_old) <= 1.5:
                found = tid
                break
        return self._tracks.pop(found) if found is not None else None

    # ------------------------------------------------------------------ output tick
    def _vehicle(self, kin: TrackKinematics, ghost: bool) -> _Vehicle:
        prior = prior_for(kin.cls)
        lane = self.lanes[kin.lane_id]  # type: ignore[index]
        return _Vehicle(
            kin.lane_id, lane.approach_id, kin.cls, kin.position(),  # type: ignore[arg-type]
            kin.stopped, ghost, prior.length_m, prior.width_m, prior.height_m,
        )

    def _front_offset(self, length_m: float) -> float:
        return {"front": 0.0, "center": -0.5 * length_m, "rear": -length_m}[self.cfg.anchor]

    def _box(self, lane: LaneSpec, veh: _Vehicle) -> tuple[OccluderBox, float]:
        s_anchor, off = lane.to_lane(veh.xy)
        s_front = float(s_anchor) + self._front_offset(veh.length_m)
        box = OccluderBox(
            s_front, s_front + veh.length_m, off - veh.width_m / 2, off + veh.width_m / 2,
            veh.height_m, not veh.ghost,
        )
        return box, s_front

    def _tick(self, t: float) -> list[LaneEstimate]:
        cfg = self.cfg
        visible: list[_Vehicle] = []
        ghost_candidates: list[tuple[int, TrackKinematics]] = []
        for tid, kin in list(self._tracks.items()):
            age = t - kin.last_seen
            if age <= cfg.track_ttl_s:
                visible.append(self._vehicle(kin, ghost=False))
            elif cfg.use_ghosts and kin.stopped and age <= cfg.ghost_ttl_s:
                ghost_candidates.append((tid, kin))
            else:
                del self._tracks[tid]

        by_approach: dict[str, list[_Vehicle]] = defaultdict(list)
        for veh in visible:
            by_approach[veh.approach_id].append(veh)

        ghosts: list[_Vehicle] = []
        for tid, kin in ghost_candidates:
            lane = self.lanes[kin.lane_id]  # type: ignore[index]
            gveh = self._vehicle(kin, ghost=True)
            _, g_front = self._box(lane, gveh)
            duplicate = any(
                v.lane_id == gveh.lane_id and v.stopped
                and abs(self._box(lane, v)[1] - g_front) < cfg.dedup_radius_m
                for v in visible
            )
            occluders = [self._box(lane, v)[0] for v in by_approach[lane.approach_id]]
            centre = np.array([g_front + gveh.length_m / 2])
            lateral = float(lane.to_lane(gveh.xy)[1])
            hidden = bool(self.occlusion.hidden_mask(lane, centre, occluders, lateral)[0])
            if duplicate or not hidden:
                del self._tracks[tid]
            else:
                ghosts.append(gveh)

        for gveh in ghosts:
            by_approach[gveh.approach_id].append(gveh)

        return [self._estimate_lane(t, lane, by_approach[lane.approach_id])
                for lane in self.lanes.values()]

    def _estimate_lane(self, t: float, lane: LaneSpec, vehicles: list[_Vehicle]) -> LaneEstimate:
        cfg = self.cfg
        occluders, stopped = [], []
        for veh in vehicles:
            box, s_front = self._box(lane, veh)
            occluders.append(box)
            if veh.lane_id == lane.lane_id and veh.stopped and s_front >= -3.0:
                stopped.append(StoppedVehicle(s_front, veh.length_m, veh.cls, veh.ghost))
        obs = self.observer.observe(lane, stopped, occluders)
        q, sd, cnt, cnt_sd = self._fuse(t, lane, obs)
        ci = (max(0.0, q - 1.96 * sd), min(lane.length_m, q + 1.96 * sd))
        prec = 1.0 / (1.0 + (sd / cfg.confidence_ref_std_m) ** 2)
        conf = prec * (0.5 + 0.5 * obs.visible_fraction) * (1.0 if obs.has_measurement else 0.8)
        return LaneEstimate(
            lane.lane_id, t, q, sd, ci, cnt, cnt_sd, q / lane.length_m, obs.visible_fraction,
            float(conf), obs.n_visible, obs.n_ghost, obs.n_hidden_inferred, obs.has_measurement,
        )

    def _fuse(
        self, t: float, lane: LaneSpec, obs: QueueObservation
    ) -> tuple[float, float, float, float]:
        cfg = self.cfg
        lid = lane.lane_id
        if not cfg.use_filter:
            if obs.has_measurement:
                self._raw[lid] = (
                    obs.length_m, math.sqrt(obs.variance_m2), obs.count, math.sqrt(obs.count_var)
                )
            else:
                q, sd, c, csd = self._raw.get(lid, (0.0, 15.0, 0.0, 3.0))
                self._raw[lid] = (q, sd * 1.1, c, csd * 1.1)
            return self._raw[lid]

        st = self._filters.get(lid)
        if st is None:
            st = _LaneFilter(
                ConstantVelocityKalman(cfg.process_accel_sigma, 2.0, cfg.gate_chi2, 0.0,
                                       lane.length_m),
                ConstantVelocityKalman(cfg.count_process_sigma, 0.5, cfg.gate_chi2, 0.0, None),
                t,
            )
            self._filters[lid] = st
            if obs.has_measurement:
                st.kf_len.reset(obs.length_m, obs.variance_m2)
                st.kf_cnt.reset(obs.count, obs.count_var)
            else:
                st.kf_len.reset(0.0, 15.0**2)
                st.kf_cnt.reset(0.0, 3.0**2)
        else:
            dt = t - st.t_last
            st.t_last = t
            st.kf_len.predict(dt)
            st.kf_cnt.predict(dt)
            if obs.has_measurement:
                st.kf_len.update(obs.length_m, obs.variance_m2)
                st.kf_cnt.update(obs.count, obs.count_var)
        return st.kf_len.mean, st.kf_len.std, st.kf_cnt.mean, st.kf_cnt.std
