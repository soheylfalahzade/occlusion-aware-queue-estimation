import numpy as np
import pandas as pd
import pytest

from oaqe.config import load_site_config
from oaqe.geometry import Homography
from oaqe.labeling import finalize_labels, lane_ticks, template_rows


def test_lane_ticks_start_at_stop_line_and_step_along_direction():
    site = load_site_config("configs/site_template.yaml")
    ticks = lane_ticks(site, 5.0)
    first = ticks["L0"]
    assert first[0][0] == 0.0 and np.allclose(first[0][1], [0.0, 0.0])
    assert np.allclose(first[2][1], [10.0, 0.0])
    assert first[-1][0] == 100.0


def test_template_and_finalize_drop_empty_rows(tmp_path):
    rows = template_rows(["L0", "L1"], [0.0, 20.0])
    assert len(rows) == 4 and rows[0]["queue_length_m"] == ""
    tpl = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(tpl, index=False)
    df = pd.read_csv(tpl)
    df.loc[0, "queue_length_m"] = 12.5
    df.to_csv(tpl, index=False)
    assert finalize_labels(str(tpl), str(tmp_path / "labels.csv")) == 1
    assert pd.read_csv(tmp_path / "labels.csv").queue_length_m.iloc[0] == 12.5


def test_draw_overlay_changes_pixels():
    pytest.importorskip("cv2")
    from oaqe.labeling import draw_overlay
    site = load_site_config("configs/site_template.yaml")
    hom = Homography.from_correspondences(site.image_points_px, site.ground_points_m)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert draw_overlay(frame, site, hom, 5.0).sum() > 0
