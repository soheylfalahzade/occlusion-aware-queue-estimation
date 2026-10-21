"""Synthetic traffic + camera renderer producing detector/tracker-like output with ground truth.

The traffic layer is an IDM car-following model behind a fixed-time signal. The camera
layer projects 3-D vehicle volumes through a pinhole model, computes per-vehicle visibility
by ray casting against all other vehicles, and emits bounding boxes of the *visible* parts
with a visibility-dependent, temporally correlated miss process and tracker ID switches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from oaqe.geometry import Homography, segments_blocked_by_boxes
from oaqe.queue_model import ground_truth_queue
from oaqe.types import Detection, LaneSpec


@dataclass
class SceneConfig:
    """Parameters of the synthetic scene."""

    n_lanes: int = 3
    lane_width_m: float = 3.5
    lane_length_m: float = 100.0
    spawn_s_m: float = 130.0
    demand_vph: float = 500.0
    lane_demand_factors: tuple[float, ...] = (1.0, 0.8, 1.2)
    truck_fraction: float = 0.15
    bus_fraction: float = 0.03
    cycle_s: float = 90.0
    green_s: float = 40.0
    fps: float = 10.0
    camera_height_m: float = 7.0
    camera_x_m: float = -14.0
    camera_y_m: float = -8.0
    aim_xy_m: tuple[float, float] = (45.0, 3.5)
    image_size_px: tuple[int, int] = (1920, 1080)
    hfov_deg: float = 70.0
    pixel_noise_px: float = 1.5
    calibration_noise_px: float = 1.0
    camera_position_noise_m: float = 0.3
    detector_recall: float = 0.97
    detect_f0: float = 0.30
    detect_k: float = 10.0
    detect_corr: float = 0.9
    min_bbox_height_px: float = 22.0
    id_switch_gap_s: float = 1.0
    id_switch_prob: float = 0.5
    seed: int = 0


@dataclass
class SimVehicle:
    """Ground-truth vehicle state. ``s_front_m`` is the bumper nearest the stop line."""

    vid: int
    lane_idx: int
    y_m: float
    s_front_m: float
    speed_mps: float
    cls: str
    length_m: float
    width_m: float
    height_m: float
    desired_speed_mps: float = 13.9
    max_accel_mps2: float = 1.6


@dataclass
class FrameData:
    """One rendered frame: tracker-like detections plus ground truth."""

    t: float
    detections: list[Detection]
    detected_true_ids: set[int]
    vehicles: list[SimVehicle] = field(default_factory=list)


class PinholeCamera:
    """Ideal pinhole camera looking at ``target`` with world z up."""

    def __init__(self, position, target, image_size, hfov_deg: float) -> None:
        self.position = np.asarray(position, dtype=float)
        fwd = np.asarray(target, dtype=float) - self.position
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.rot = np.stack([right, down, fwd])
        w, h = image_size
        self.focal = (w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        self.centre = (w / 2.0, h / 2.0)
        self.image_size = image_size

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixels. Returns ``(uv, depth)``; invalid depth -> NaN uv."""
        pc = (np.asarray(pts, dtype=float) - self.position) @ self.rot.T
        z = pc[:, 2]
        safe = np.where(z > 0.1, z, np.nan)
        u = self.focal * pc[:, 0] / safe + self.centre[0]
        v = self.focal * pc[:, 1] / safe + self.centre[1]
        return np.stack([u, v], axis=1), z


_LATTICE = np.array(
    [(a, b, c) for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)
     if not (a == 0.5 and b == 0.5 and c == 0.5)]
)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class CameraRenderer:
    """Renders tracker-like detections from 3-D vehicle states (traffic-source agnostic)."""

    def __init__(self, camera: PinholeCamera, cfg: SceneConfig, rng: np.random.Generator) -> None:
        self.camera, self.cfg, self.rng = camera, cfg, rng
        self._latent: dict[int, float] = {}
        self._track: dict[int, tuple[float, int]] = {}
        self._fresh_id = 1_000_000

    def render(self, t: float, vehicles: list[SimVehicle]) -> tuple[list[Detection], set[int]]:
        """Return detections and the set of *true* vehicle IDs that were detected."""
        cfg, rng = self.cfg, self.rng
        if not vehicles:
            return [], set()
        n_v, n_p = len(vehicles), len(_LATTICE)
        bmin = np.array([[v.s_front_m, v.y_m - v.width_m / 2, 0.0] for v in vehicles])
        bmax = np.array(
            [[v.s_front_m + v.length_m, v.y_m + v.width_m / 2, v.height_m] for v in vehicles]
        )
        pts = (bmin[:, None, :] + _LATTICE[None] * (bmax - bmin)[:, None, :]).reshape(-1, 3)
        owner = np.repeat(np.arange(n_v), n_p)
        hits = segments_blocked_by_boxes(self.camera.position, pts, bmin, bmax)
        own_blocked = hits[np.arange(len(pts)), owner]
        hits[np.arange(len(pts)), owner] = False
        other_blocked = hits.any(axis=1)
        uv, _ = self.camera.project(pts)
        w, h = cfg.image_size_px
        in_img = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        in_img &= (uv[:, 1] >= 0) & (uv[:, 1] < h)
        self_vis = (~own_blocked) & in_img
        vis = self_vis & ~other_blocked

        detections: list[Detection] = []
        detected: set[int] = set()
        for i, veh in enumerate(vehicles):
            sl = slice(i * n_p, (i + 1) * n_p)
            n_self, n_vis = int(self_vis[sl].sum()), int(vis[sl].sum())
            z_prev = self._latent.get(veh.vid, rng.normal())
            z = cfg.detect_corr * z_prev + math.sqrt(1 - cfg.detect_corr**2) * rng.normal()
            self._latent[veh.vid] = z
            if n_vis == 0 or n_self == 0:
                continue
            f_rel = n_vis / n_self
            p_det = cfg.detector_recall / (1.0 + math.exp(-cfg.detect_k * (f_rel - cfg.detect_f0)))
            pts_v = uv[sl][vis[sl]]
            (u0, v0), (u1, v1) = pts_v.min(axis=0), pts_v.max(axis=0)
            p_det *= min(1.0, (v1 - v0) / cfg.min_bbox_height_px)
            if _phi(z) >= p_det:
                continue
            jitter = rng.normal(0.0, cfg.pixel_noise_px, 4)
            box = (
                float(np.clip(u0 + jitter[0], 0, w)), float(np.clip(v0 + jitter[1], 0, h)),
                float(np.clip(u1 + jitter[2], 0, w)), float(np.clip(v1 + jitter[3], 0, h)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detected.add(veh.vid)
            detections.append(
                Detection(self._tracker_id(t, veh.vid), veh.cls, box, 0.5 + 0.5 * f_rel)
            )
        return detections, detected

    def _tracker_id(self, t: float, vid: int) -> int:
        last = self._track.get(vid)
        if last is None:
            tid = vid
        else:
            tid = last[1]
            lost = t - last[0] > self.cfg.id_switch_gap_s
            if lost and self.rng.random() < self.cfg.id_switch_prob:
                self._fresh_id += 1
                tid = self._fresh_id
        self._track[vid] = (t, tid)
        return tid


def _idm_accel(v: float, v0: float, a_max: float, gap: float, dv: float) -> float:
    s_star = 2.0 + max(0.0, v * 1.4 + v * dv / (2.0 * math.sqrt(a_max * 2.0)))
    acc = a_max * (1.0 - (v / v0) ** 4 - (s_star / max(gap, 0.1)) ** 2)
    return float(np.clip(acc, -9.0, a_max))


class TrafficSimulator:
    """Multi-lane IDM traffic with Poisson arrivals behind a fixed-time signal."""

    def __init__(self, cfg: SceneConfig, rng: np.random.Generator) -> None:
        self.cfg, self.rng = cfg, rng
        self.lanes: list[list[SimVehicle]] = [[] for _ in range(cfg.n_lanes)]
        self.t = 0.0
        self._next_vid = 1
        self._offset = float(rng.uniform(0.0, cfg.cycle_s))

    def lane_y(self, idx: int) -> float:
        return idx * self.cfg.lane_width_m

    def is_red(self, t: float) -> bool:
        return ((t + self._offset) % self.cfg.cycle_s) >= self.cfg.green_s

    def _spawn(self, lane_idx: int) -> None:
        cfg, rng = self.cfg, self.rng
        r = rng.random()
        if r < cfg.truck_fraction:
            cls, length = "truck", float(np.clip(rng.normal(7.5, 1.2), 5.5, 12.0))
            h, w, v0, a = float(rng.normal(3.0, 0.3)), 2.5, float(rng.normal(12.5, 1.0)), 1.0
        elif r < cfg.truck_fraction + cfg.bus_fraction:
            cls, length = "bus", float(rng.normal(11.0, 0.5))
            h, w, v0, a = 3.2, 2.5, float(rng.normal(12.0, 1.0)), 1.0
        else:
            cls, length = "car", float(rng.normal(4.5, 0.25))
            h, w, v0, a = float(rng.normal(1.5, 0.08)), 1.8, float(rng.normal(13.9, 1.2)), 1.6
        lane = self.lanes[lane_idx]
        speed = min(v0, lane[-1].speed_mps + 3.0) if lane else v0
        lane.append(
            SimVehicle(self._next_vid, lane_idx, self.lane_y(lane_idx), cfg.spawn_s_m, speed, cls,
                       length, w, h, max(v0, 5.0), a)
        )
        self._next_vid += 1

    def step(self, dt: float) -> None:
        cfg = self.cfg
        red = self.is_red(self.t)
        for idx, lane in enumerate(self.lanes):
            lane.sort(key=lambda v: v.s_front_m)
            accs = []
            for k, veh in enumerate(lane):
                gap, dv = math.inf, 0.0
                if k > 0:
                    lead = lane[k - 1]
                    gap = veh.s_front_m - (lead.s_front_m + lead.length_m)
                    dv = veh.speed_mps - lead.speed_mps
                if red and veh.s_front_m > 0.0 and veh.s_front_m < gap:
                    gap, dv = veh.s_front_m, veh.speed_mps
                if math.isinf(gap):
                    gap, dv = 1e3, 0.0
                accs.append(
                    _idm_accel(veh.speed_mps, veh.desired_speed_mps, veh.max_accel_mps2, gap, dv)
                )
            for veh, acc in zip(lane, accs, strict=True):
                veh.speed_mps = max(0.0, veh.speed_mps + acc * dt)
                veh.s_front_m -= veh.speed_mps * dt
            lane[:] = [v for v in lane if v.s_front_m > -15.0]
            factor = cfg.lane_demand_factors[idx % len(cfg.lane_demand_factors)]
            rate = cfg.demand_vph * factor / 3600.0
            clear = not lane or cfg.spawn_s_m - (lane[-1].s_front_m + lane[-1].length_m) >= 10.0
            if clear and self.rng.random() < rate * dt:
                self._spawn(idx)
        self.t += dt

    def vehicles(self) -> list[SimVehicle]:
        return [v for lane in self.lanes for v in lane]


class SyntheticScene:
    """Traffic + camera + calibration bundle with a frame-by-frame interface."""

    def __init__(self, cfg: SceneConfig | None = None) -> None:
        self.cfg = cfg or SceneConfig()
        ss = np.random.SeedSequence(self.cfg.seed).spawn(4)
        rng_traffic, rng_render, rng_calib, rng_pose = (np.random.default_rng(s) for s in ss)
        c = self.cfg
        self.traffic = TrafficSimulator(c, rng_traffic)
        self.camera = PinholeCamera(
            (c.camera_x_m, c.camera_y_m, c.camera_height_m), self._aim_point(c),
            c.image_size_px, c.hfov_deg,
        )
        self.renderer = CameraRenderer(self.camera, c, rng_render)
        self._rng_calib, self._rng_pose = rng_calib, rng_pose
        self.dt = 1.0 / c.fps

    @staticmethod
    def _aim_point(c: SceneConfig) -> tuple[float, float, float]:
        """Aim the optical axis so that the stop line and the far end of the ROI are centred."""
        cam_xy = np.array([c.camera_x_m, c.camera_y_m])
        mid_y = c.lane_width_m * (c.n_lanes - 1) / 2.0
        dep = [
            math.atan2(c.camera_height_m, float(np.linalg.norm(np.array([x, mid_y]) - cam_xy)))
            for x in (0.0, c.lane_length_m)
        ]
        pitch = 0.5 * (dep[0] + dep[1])
        horiz = c.camera_height_m / math.tan(pitch)
        direction = np.array([c.aim_xy_m[0], mid_y]) - cam_xy
        direction /= np.linalg.norm(direction)
        xy = cam_xy + horiz * direction
        return float(xy[0]), float(xy[1]), 0.0

    @property
    def lanes(self) -> list[LaneSpec]:
        """Lane specifications in the synthetic world frame (x = upstream, y = lateral)."""
        c = self.cfg
        return [
            LaneSpec(f"L{i}", (0.0, self.traffic.lane_y(i)), (1.0, 0.0), c.lane_width_m,
                     c.lane_length_m, "A")
            for i in range(c.n_lanes)
        ]

    def estimator_inputs(self) -> tuple[Homography, tuple[float, float, float]]:
        """Noisy calibration homography and noisy camera position, as a deployment would have."""
        c = self.cfg
        xs = np.array([5.0, 25.0, 45.0, 65.0, 85.0])
        ys = np.array([-c.lane_width_m / 2, c.lane_width_m / 2 + c.lane_width_m * (c.n_lanes - 1)])
        ys = np.concatenate([ys, [c.lane_width_m * (c.n_lanes - 1) / 2]])
        ground = np.array([(x, y) for x in xs for y in ys])
        pix, _ = self.camera.project(np.column_stack([ground, np.zeros(len(ground))]))
        pix = pix + self._rng_calib.normal(0.0, c.calibration_noise_px, pix.shape)
        hom = Homography.from_correspondences(pix, ground)
        pos = self.camera.position + self._rng_pose.normal(0.0, c.camera_position_noise_m, 3)
        return hom, (float(pos[0]), float(pos[1]), float(pos[2]))

    def step(self) -> FrameData:
        """Advance one frame and render it."""
        self.traffic.step(self.dt)
        vehicles = self.traffic.vehicles()
        dets, ids = self.renderer.render(self.traffic.t, vehicles)
        return FrameData(self.traffic.t, dets, ids, vehicles)

    def truth_queue(self, lane_idx: int, gap_max_m: float = 8.0, head_tol_m: float = 8.0,
                    v_stop_mps: float = 1.0) -> tuple[float, int, set[int]]:
        """Ground-truth queue ``(length_m, count, member vehicle IDs)`` for a lane."""
        lane = self.traffic.lanes[lane_idx]
        if not lane:
            return 0.0, 0, set()
        s = np.array([v.s_front_m for v in lane])
        ln = np.array([v.length_m for v in lane])
        sp = np.array([v.speed_mps for v in lane])
        length, count, members = ground_truth_queue(
            s, ln, sp, gap_max_m, head_tol_m, v_stop_mps, self.cfg.lane_length_m
        )
        return length, count, {lane[i].vid for i in members}
