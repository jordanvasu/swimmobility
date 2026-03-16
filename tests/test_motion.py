"""Tests for swimmobility.video.motion.compute_motion_energy."""
from __future__ import annotations

import numpy as np
import pytest

from swimmobility.video.motion import compute_motion_energy


def _bgr_frame(h: int, w: int, value: int = 0) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestComputeMotionEnergy:
    def test_single_frame_pair_no_roi(self):
        """Two frames with known difference → energy equals pixel sum."""
        h, w = 8, 8
        frame1 = _bgr_frame(h, w, 0)
        frame2 = _bgr_frame(h, w, 50)  # all pixels differ by 50

        # BGR [50,50,50] → grayscale ≈ 50 (coefficients sum to 1.0)
        raw, smoothed, centroids = compute_motion_energy(
            iter([frame1, frame2]), roi_mask=None, smoothing_kernel_size=1
        )

        assert len(raw) == 1
        assert len(smoothed) == 1
        assert len(centroids) == 1
        # gray diff per pixel = 50, total = h*w*50
        assert raw[0] == pytest.approx(h * w * 50, rel=0.02)

    def test_roi_restricts_energy(self):
        """Energy outside the ROI mask is excluded."""
        h, w = 10, 10
        frame1 = _bgr_frame(h, w, 0)
        frame2 = _bgr_frame(h, w, 100)

        # ROI covers only top-left 4×4
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[:4, :4] = 255

        raw, smoothed, centroids = compute_motion_energy(
            iter([frame1, frame2]), roi_mask=roi_mask, smoothing_kernel_size=1
        )

        assert len(raw) == 1
        # Only 4*4 = 16 pixels in ROI, each with diff = 100
        assert raw[0] == pytest.approx(16 * 100, rel=0.02)

    def test_output_length_n_minus_1(self):
        """N frames → N-1 energy values."""
        h, w = 6, 6
        frames = [_bgr_frame(h, w, v) for v in range(5)]
        raw, smoothed, centroids = compute_motion_energy(
            iter(frames), roi_mask=None, smoothing_kernel_size=1
        )
        assert len(raw) == 4
        assert len(smoothed) == 4
        assert len(centroids) == 4

    def test_zero_diff_produces_zero_energy(self):
        """Identical frames → zero energy, centroid is None."""
        h, w = 6, 6
        frame = _bgr_frame(h, w, 42)
        raw, smoothed, centroids = compute_motion_energy(
            iter([frame, frame.copy()]), roi_mask=None, smoothing_kernel_size=1
        )
        assert raw[0] == pytest.approx(0.0)
        assert centroids[0] is None

    def test_centroid_in_roi_region(self):
        """Centroid should be within the motion region."""
        h, w = 20, 20
        frame1 = _bgr_frame(h, w, 0)
        frame2 = _bgr_frame(h, w, 0)
        # Motion only in rows 10-14, cols 10-14
        frame2[10:15, 10:15] = 80

        raw, smoothed, centroids = compute_motion_energy(
            iter([frame1, frame2]), roi_mask=None, smoothing_kernel_size=1
        )
        cx, cy = centroids[0]
        # Centroid of a 5×5 block starting at (10,10) is at (12.0, 12.0)
        assert cx == pytest.approx(12.0, abs=0.5)
        assert cy == pytest.approx(12.0, abs=0.5)

    def test_smoothing_does_not_change_length(self):
        """Smoothed output has same length as raw."""
        frames = [_bgr_frame(4, 4, v * 20) for v in range(10)]
        raw, smoothed, _ = compute_motion_energy(
            iter(frames), roi_mask=None, smoothing_kernel_size=5
        )
        assert len(smoothed) == len(raw)

    def test_smoothed_reduces_variance(self):
        """Smoothed series should have lower or equal variance than raw."""
        rng = np.random.default_rng(0)
        h, w = 8, 8
        frames = [rng.integers(0, 200, (h, w, 3), dtype=np.uint8) for _ in range(20)]
        raw, smoothed, _ = compute_motion_energy(iter(frames), roi_mask=None, smoothing_kernel_size=5)
        assert smoothed.var() <= raw.var() + 1e-6
