"""Tests for swimmobility.detection.immobility.detect_immobility."""
from __future__ import annotations

import numpy as np
import pytest

from swimmobility.detection.immobility import detect_immobility


class TestDetectImmobility:
    def test_single_bout_detected(self):
        """One clear immobility bout surrounded by mobile frames."""
        fps = 10.0
        # 20 mobile frames, 30 immobile, 10 mobile  (indices 0-19, 20-49, 50-59)
        energy = np.array([5.0] * 20 + [0.5] * 30 + [5.0] * 10)
        bouts, mask = detect_immobility(energy, fps, threshold=2.0, min_bout_s=0.5)

        assert len(bouts) == 1
        start, end = bouts[0]
        assert start == 20
        assert end == 49
        assert mask[20:50].all()
        assert not mask[:20].any()
        assert not mask[50:].any()

    def test_short_bout_discarded(self):
        """Bouts shorter than min_bout_s are not returned."""
        fps = 10.0
        # 3 frames below threshold = 0.3 s < 0.5 s minimum
        energy = np.array([5.0] * 10 + [0.5] * 3 + [5.0] * 10)
        bouts, mask = detect_immobility(energy, fps, threshold=2.0, min_bout_s=0.5)

        assert len(bouts) == 0
        assert not mask.any()

    def test_minimum_bout_exactly_met(self):
        """A bout exactly at the minimum duration is kept."""
        fps = 10.0
        min_bout_s = 0.5  # 5 frames at 10 fps
        energy = np.array([5.0] * 5 + [0.5] * 5 + [5.0] * 5)
        bouts, mask = detect_immobility(energy, fps, threshold=2.0, min_bout_s=min_bout_s)

        assert len(bouts) == 1
        assert bouts[0] == (5, 9)

    def test_multiple_bouts(self):
        """Multiple separated bouts are all detected."""
        fps = 10.0
        # bout at 10-19, gap 20-24, bout at 25-39
        energy = np.array([5.0] * 10 + [0.5] * 10 + [5.0] * 5 + [0.5] * 15 + [5.0] * 5)
        bouts, mask = detect_immobility(energy, fps, threshold=2.0, min_bout_s=0.5)

        assert len(bouts) == 2
        assert bouts[0] == (10, 19)
        assert bouts[1] == (25, 39)

    def test_all_immobile(self):
        """Entire series below threshold → one bout spanning everything."""
        fps = 5.0
        energy = np.full(20, 0.1)
        bouts, mask = detect_immobility(energy, fps, threshold=1.0, min_bout_s=0.5)

        assert len(bouts) == 1
        assert bouts[0] == (0, 19)
        assert mask.all()

    def test_all_mobile(self):
        """No frames below threshold → no bouts."""
        energy = np.full(20, 10.0)
        bouts, mask = detect_immobility(energy, fps=10.0, threshold=2.0)

        assert len(bouts) == 0
        assert not mask.any()

    def test_empty_series(self):
        """Empty input returns empty outputs without error."""
        bouts, mask = detect_immobility(np.array([]), fps=10.0)
        assert bouts == []
        assert len(mask) == 0

    def test_immobile_mask_consistent_with_bouts(self):
        """immobile_mask True exactly where bouts say it should be."""
        fps = 10.0
        energy = np.array([5.0] * 5 + [0.3] * 8 + [5.0] * 5 + [0.1] * 12 + [5.0] * 5)
        bouts, mask = detect_immobility(energy, fps, threshold=1.0, min_bout_s=0.5)

        # Reconstruct expected mask from bouts
        expected = np.zeros(len(energy), dtype=bool)
        for s, e in bouts:
            expected[s : e + 1] = True

        np.testing.assert_array_equal(mask, expected)

    def test_bout_at_end_of_series(self):
        """A bout that reaches the last frame is correctly captured."""
        fps = 10.0
        energy = np.array([5.0] * 10 + [0.5] * 10)
        bouts, mask = detect_immobility(energy, fps, threshold=2.0, min_bout_s=0.5)

        assert len(bouts) == 1
        assert bouts[0] == (10, 19)
        assert mask[10:].all()
        assert not mask[:10].any()
