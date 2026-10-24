from oaqe.viz import render_queue_gif


def test_queue_gif_smoke(tmp_path):
    out = tmp_path / "q.gif"
    info = render_queue_gif(out, seed=4, t_start=70.0, n_frames=3)
    assert out.exists() and out.stat().st_size > 1000 and info["frames"] >= 2
