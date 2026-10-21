import numpy as np

from oaqe.config import EstimatorConfig, load_site_config
from oaqe.pipeline import QueueEstimationPipeline
from oaqe.pressure import MovementSpec, PressureEstimator, spillback_risk, to_fuzzy_inputs
from oaqe.sim.synthetic import SceneConfig, SyntheticScene


def _run(cfg: SceneConfig, est_cfg: EstimatorConfig, frames: int):
    scene = SyntheticScene(cfg)
    hom, cam = scene.estimator_inputs()
    pipe = QueueEstimationPipeline(scene.lanes, hom, cam, est_cfg)
    rows, last = [], []
    for _ in range(frames):
        fr = scene.step()
        ests = pipe.process_frame(fr.t, fr.detections)
        for e in ests:
            q, n, _ = scene.truth_queue(int(e.lane_id[1:]))
            rows.append((fr.t, q, e.queue_length_m, n, e.vehicle_count))
        if ests:
            last = ests
    return np.array(rows), last, scene


def test_end_to_end_unoccluded_scene_is_accurate():
    cfg = SceneConfig(seed=7, camera_height_m=10.0, truck_fraction=0.0, bus_fraction=0.0)
    rows, last, _ = _run(cfg, EstimatorConfig(), frames=2500)
    rows = rows[rows[:, 0] > 60.0]
    mae = np.abs(rows[:, 1] - rows[:, 2]).mean()
    assert mae < 4.0
    assert all(np.isfinite(e.queue_length_m) and e.queue_length_m >= 0.0 for e in last)
    assert all(e.queue_ci95_m[0] <= e.queue_length_m <= e.queue_ci95_m[1] for e in last)
    assert all(0.0 <= e.confidence <= 1.0 for e in last)


def test_pipeline_is_deterministic_given_seed():
    cfg = SceneConfig(seed=3, camera_height_m=6.0)
    a, _, _ = _run(cfg, EstimatorConfig(), frames=900)
    b, _, _ = _run(cfg, EstimatorConfig(), frames=900)
    np.testing.assert_array_equal(a, b)


def test_output_rate_matches_period():
    cfg = SceneConfig(seed=1)
    rows, _, scene = _run(cfg, EstimatorConfig(output_period_s=2.0), frames=400)
    ticks = np.unique(rows[:, 0])
    assert np.allclose(np.diff(ticks), 2.0, atol=scene.dt + 1e-6)


def test_pressure_and_fuzzy_interfaces():
    cfg = SceneConfig(seed=5, camera_height_m=8.0)
    _, last, scene = _run(cfg, EstimatorConfig(), frames=1500)
    est = {e.lane_id: e for e in last}
    pe = PressureEstimator([MovementSpec("m0", "L0", ("L1",), (0.5,))], scene.lanes)
    out = pe.compute(est)
    assert len(out) == 1 and out[0].std >= 0.0
    fz = to_fuzzy_inputs(est["L0"], scene.lanes[0])
    assert set(fz) == {"queue_norm", "queue_m", "confidence", "std_norm", "spillback_risk"}
    assert all(0.0 <= v <= 1.0 for k, v in fz.items() if k != "queue_m")
    assert spillback_risk(0.2) == 0.0 and spillback_risk(1.0) == 1.0


def test_template_config_loads():
    from pathlib import Path

    site = load_site_config(Path(__file__).parents[1] / "configs" / "site_template.yaml")
    pipe = QueueEstimationPipeline.from_site_config(site)
    assert len(pipe.lanes) == len(site.lanes) >= 1
