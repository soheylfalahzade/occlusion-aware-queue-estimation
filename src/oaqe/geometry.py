"""Planar geometry: image-to-ground homography and segment/box visibility tests."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _normalisation_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: zero centroid, mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    mean_dist = np.linalg.norm(pts - centroid, axis=1).mean()
    scale = np.sqrt(2.0) / max(float(mean_dist), 1e-12)
    return np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
    )


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    hom = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1)
    out = hom @ h.T
    return out[..., :2] / out[..., 2:3]


class Homography:
    """Projective map between the image plane and the metric road plane (z = 0)."""

    def __init__(self, matrix: ArrayLike) -> None:
        h = np.asarray(matrix, dtype=float).reshape(3, 3)
        if abs(h[2, 2]) > 1e-12:
            h = h / h[2, 2]
        self._h = h
        self._h_inv = np.linalg.inv(h)

    @property
    def matrix(self) -> np.ndarray:
        """3x3 matrix mapping homogeneous image pixels to ground coordinates."""
        return self._h.copy()

    @classmethod
    def from_correspondences(cls, image_pts: ArrayLike, ground_pts: ArrayLike) -> Homography:
        """Estimate the homography with the normalised direct linear transform (DLT).

        Args:
            image_pts: ``(N, 2)`` pixel coordinates, ``N >= 4``.
            ground_pts: ``(N, 2)`` metric ground-plane coordinates of the same points.
        """
        src = np.asarray(image_pts, dtype=float)
        dst = np.asarray(ground_pts, dtype=float)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
            raise ValueError("need >= 4 matching (N, 2) point pairs")
        t_src, t_dst = _normalisation_matrix(src), _normalisation_matrix(dst)
        s, d = _apply(t_src, src), _apply(t_dst, dst)
        rows = []
        for (x, y), (u, v) in zip(s, d, strict=True):
            rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
            rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
        _, _, vt = np.linalg.svd(np.asarray(rows))
        h_norm = vt[-1].reshape(3, 3)
        return cls(np.linalg.inv(t_dst) @ h_norm @ t_src)

    def image_to_ground(self, pts_px: ArrayLike) -> np.ndarray:
        """Map pixels ``(..., 2)`` to metric ground coordinates ``(..., 2)``."""
        return _apply(self._h, np.asarray(pts_px, dtype=float))

    def ground_to_image(self, pts_m: ArrayLike) -> np.ndarray:
        """Map ground coordinates ``(..., 2)`` to pixels ``(..., 2)``."""
        return _apply(self._h_inv, np.asarray(pts_m, dtype=float))

    def ground_rmse_m(self, image_pts: ArrayLike, ground_pts: ArrayLike) -> float:
        """Root-mean-square ground-plane error (metres) over a set of correspondences."""
        err = self.image_to_ground(image_pts) - np.asarray(ground_pts, dtype=float)
        return float(np.sqrt((err**2).sum(axis=1).mean()))


def segments_blocked_by_boxes(
    origin: ArrayLike,
    targets: ArrayLike,
    box_min: ArrayLike,
    box_max: ArrayLike,
    eps: float = 1e-3,
) -> np.ndarray:
    """Slab test between segments ``origin -> target`` and axis-aligned boxes.

    A segment is blocked by a box if it enters the box before reaching the target
    (a relative tolerance ``eps`` lets targets lying *on* a box surface pass).

    Returns:
        Boolean matrix of shape ``(N_targets, N_boxes)``.
    """
    o = np.asarray(origin, dtype=float)
    t = np.atleast_2d(np.asarray(targets, dtype=float))
    bmin = np.atleast_2d(np.asarray(box_min, dtype=float))
    bmax = np.atleast_2d(np.asarray(box_max, dtype=float))
    if bmin.size == 0:
        return np.zeros((len(t), 0), dtype=bool)
    d = t - o
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    inv = 1.0 / d
    t1 = (bmin[None, :, :] - o) * inv[:, None, :]
    t2 = (bmax[None, :, :] - o) * inv[:, None, :]
    t_near = np.minimum(t1, t2).max(axis=2)
    t_far = np.maximum(t1, t2).min(axis=2)
    return (t_far >= np.maximum(t_near, 0.0)) & (t_near < 1.0 - eps)
