import numpy as np

from oaqe.occlusion import OccluderBox, OcclusionModel
from oaqe.types import LaneSpec

LANE = LaneSpec("L", (0.0, 0.0), (1.0, 0.0), 3.5, 100.0)


def test_lane_frame_roundtrip():
    lane = LaneSpec("x", (3.0, -2.0), (0.0, 1.0))
    pts = np.array([[3.0, 10.0], [4.0, 5.0]])
    np.testing.assert_allclose(lane.to_world(lane.to_lane(pts)), pts, atol=1e-12)


def test_tall_vehicle_hides_road_behind_it_but_low_camera_view_is_clear():
    model = OcclusionModel((-10.0, -6.0, 4.0), probe_height_m=0.9)
    truck = OccluderBox(10.0, 17.0, -1.25, 1.25, 3.0, visible=True)
    grid = np.arange(0.0, 60.0, 0.5)
    hidden = model.hidden_mask(LANE, grid, [truck])
    assert hidden[(grid > 17.5) & (grid < 22.0)].all()  # shadow behind the truck
    assert not hidden[grid < 9.0].any()  # in front of the truck
    assert not hidden[(grid >= 10.0) & (grid <= 17.0)].any()  # the truck itself is visible
    assert not hidden[grid > 50.0].any()  # far beyond the shadow


def test_higher_camera_shortens_shadow():
    truck = OccluderBox(10.0, 17.0, -1.25, 1.25, 3.0)
    grid = np.arange(0.0, 80.0, 0.5)
    low = OcclusionModel((-10.0, -6.0, 4.0)).hidden_mask(LANE, grid, [truck]).sum()
    high = OcclusionModel((-10.0, -6.0, 12.0)).hidden_mask(LANE, grid, [truck]).sum()
    assert high < low


def test_ghost_box_is_hidden_not_observed():
    model = OcclusionModel((-10.0, -6.0, 6.0))
    ghost = OccluderBox(10.0, 15.0, -0.9, 0.9, 1.5, visible=False)
    hidden = model.hidden_mask(LANE, np.array([12.0]), [ghost])
    assert hidden[0]
