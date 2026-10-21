import numpy as np

from oaqe.config import EstimatorConfig
from oaqe.filtering import ConstantVelocityKalman
from oaqe.occlusion import OccluderBox, OcclusionModel
from oaqe.queue_model import QueueObserver, StoppedVehicle, ground_truth_queue
from oaqe.types import LaneSpec

LANE = LaneSpec("L", (0.0, 0.0), (1.0, 0.0), 3.5, 100.0)


def test_ground_truth_chain_rules():
    s = np.array([1.0, 7.0, 13.0, 40.0])
    ln = np.array([4.5, 4.5, 4.5, 4.5])
    v = np.array([0.0, 0.0, 0.0, 0.0])
    length, count, members = ground_truth_queue(s, ln, v, gap_max_m=8.0, head_tol_m=8.0)
    assert count == 3 and sorted(members) == [0, 1, 2]
    assert abs(length - 17.5) < 1e-9
    # a moving vehicle is not part of the queue and breaks the chain
    s2 = np.array([1.0, 7.0, 16.0, 40.0])
    v = np.array([0.0, 5.0, 0.0, 0.0])
    assert ground_truth_queue(s2, ln, v, 8.0, 8.0)[1] == 1


def test_ground_truth_empty_and_clipping():
    far = ground_truth_queue(np.array([30.0]), np.array([4.5]), np.array([0.0]), 8, 8)
    assert far[:2] == (0.0, 0)
    s = np.arange(0.0, 200.0, 6.5)
    out = ground_truth_queue(s, np.full_like(s, 4.5), np.zeros_like(s), 8, 8, max_length_m=50.0)
    assert out[0] == 50.0


def test_observer_infers_vehicle_hidden_behind_truck():
    cfg = EstimatorConfig()
    obs = QueueObserver(cfg, OcclusionModel((-10.0, -3.0, 4.0)))
    car_a = StoppedVehicle(1.0, 4.5)
    truck = StoppedVehicle(7.5, 7.5, "truck")
    car_d = StoppedVehicle(25.0, 4.5)  # a hidden car sits between the truck and this one
    boxes = [
        OccluderBox(1.0, 5.5, -0.9, 0.9, 1.5),
        OccluderBox(7.5, 15.0, -1.25, 1.25, 3.0),
        OccluderBox(25.0, 29.5, -0.9, 0.9, 1.5),
    ]
    res = obs.observe(LANE, [car_a, truck, car_d], boxes)
    assert res.n_hidden_inferred >= 1.0
    assert res.count >= 4.0
    assert abs(res.length_m - 29.5) < 3.0


def test_observer_breaks_chain_when_gap_is_visible():
    cfg = EstimatorConfig()
    obs = QueueObserver(cfg, OcclusionModel((-10.0, -6.0, 6.0)))
    stopped = [StoppedVehicle(1.0, 4.5), StoppedVehicle(30.0, 4.5)]
    res = obs.observe(LANE, stopped, [OccluderBox(1.0, 5.5, -0.9, 0.9, 1.5)])
    assert res.count < 2.0 + 1e-9 and res.length_m < 12.0


def test_observer_no_measurement_when_stop_line_is_hidden():
    cfg = EstimatorConfig()
    obs = QueueObserver(cfg, OcclusionModel((-10.0, -1.0, 3.0)))
    wall = OccluderBox(-8.0, -0.5, -1.25, 1.25, 3.2)  # large vehicle right at the stop line
    res = obs.observe(LANE, [], [wall])
    assert not res.has_measurement


def test_kalman_converges_and_gates_outliers():
    kf = ConstantVelocityKalman(accel_sigma=0.5, gate_chi2=9.0, lower=0.0, upper=100.0)
    kf.reset(0.0, 100.0)
    for _ in range(50):
        kf.predict(1.0)
        kf.update(20.0, 1.0)
    assert abs(kf.mean - 20.0) < 0.5
    before = kf.mean
    kf.predict(1.0)
    kf.update(90.0, 1.0)  # gross outlier is down-weighted
    assert abs(kf.mean - before) < 20.0
    assert 0.0 <= kf.mean <= 100.0
