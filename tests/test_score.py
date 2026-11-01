import pandas as pd

from oaqe.cli import score_against_labels


def test_score_matches_nearest_estimate_within_tolerance(tmp_path):
    est = pd.DataFrame({"lane_id": ["a"] * 4, "timestamp_s": [0.0, 1.0, 2.0, 3.0],
                        "queue_length_m": [10.0, 12.0, 14.0, 16.0],
                        "queue_ci95_lo_m": [8.0, 10.0, 12.0, 14.0],
                        "queue_ci95_hi_m": [12.0, 14.0, 16.0, 18.0]})
    lab = pd.DataFrame({"lane_id": ["a", "a", "b", "a"], "timestamp_s": [0.1, 2.0, 1.0, 50.0],
                        "queue_length_m": [11.0, 13.0, 5.0, 9.0]})
    est.to_csv(tmp_path / "e.csv", index=False)
    lab.to_csv(tmp_path / "l.csv", index=False)
    r = score_against_labels(str(tmp_path / "e.csv"), str(tmp_path / "l.csv"), n_boot=200)
    assert r["n_matched"] == 2 and r["n_labels"] == 4          # lane b and t=50 s do not match
    assert abs(r["mae_m"] - 1.0) < 1e-9 and abs(r["bias_m"] - 0.0) < 1e-9
    assert r["coverage95"] == 1.0
