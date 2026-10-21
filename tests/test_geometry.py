import numpy as np
import pytest

from oaqe.geometry import Homography, segments_blocked_by_boxes


def test_homography_roundtrip_and_recovery():
    true_h = np.array([[0.02, 0.001, -5.0], [0.0005, 0.05, -20.0], [1e-5, 8e-4, 1.0]])
    h_true = Homography(true_h)
    rng = np.random.default_rng(0)
    px = rng.uniform([100, 300], [1800, 1000], size=(12, 2))
    ground = h_true.image_to_ground(px)
    est = Homography.from_correspondences(px, ground)
    np.testing.assert_allclose(est.image_to_ground(px), ground, atol=1e-6)
    np.testing.assert_allclose(est.ground_to_image(ground), px, atol=1e-4)
    assert est.ground_rmse_m(px, ground) < 1e-6


def test_homography_requires_four_points():
    with pytest.raises(ValueError):
        Homography.from_correspondences([[0, 0], [1, 1], [2, 0]], [[0, 0], [1, 1], [2, 0]])


def test_segment_box_blocking():
    origin = np.array([0.0, 0.0, 5.0])
    bmin, bmax = np.array([[10.0, -1.0, 0.0]]), np.array([[14.0, 1.0, 3.0]])
    targets = np.array(
        [
            [20.0, 0.0, 0.5],  # behind the box, low: blocked
            [20.0, 0.0, 4.8],  # passes above the box: clear
            [12.0, 0.0, 3.0],  # on the top face: not blocked (surface tolerance)
            [20.0, 6.0, 0.5],  # far to the side: clear
        ]
    )
    blocked = segments_blocked_by_boxes(origin, targets, bmin, bmax)[:, 0]
    assert blocked.tolist() == [True, False, False, False]
