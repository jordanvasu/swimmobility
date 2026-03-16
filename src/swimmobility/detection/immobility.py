from __future__ import annotations

from typing import List, Tuple

import numpy as np


def detect_immobility(
    smoothed_energy: np.ndarray,
    fps: float,
    threshold: float = 2.0,
    min_bout_s: float = 0.5,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Detect immobility bouts from a smoothed motion-energy series.

    A frame is a candidate for immobility when its smoothed motion energy is
    strictly below *threshold*.  Contiguous candidate runs shorter than
    *min_bout_s* seconds are discarded.

    Parameters
    ----------
    smoothed_energy:
        1-D float array of smoothed motion energy, length N (one value per
        frame difference, i.e. N = total_frames - 1).
    fps:
        Frames per second of the source video.
    threshold:
        Motion-energy threshold (pixel-sum units). Frames below this value
        are considered immobile candidates.
    min_bout_s:
        Minimum duration in seconds for a bout to be retained.

    Returns
    -------
    bouts:
        List of (start_frame, end_frame) pairs, both indices inclusive,
        referencing positions in *smoothed_energy*.
    immobile_mask:
        Boolean array of length N; True where the frame belongs to a
        retained immobility bout.
    """
    n = len(smoothed_energy)
    if n == 0:
        return [], np.zeros(0, dtype=bool)

    min_bout_frames = max(1, int(min_bout_s * fps))
    candidates: np.ndarray = smoothed_energy < threshold

    bouts: List[Tuple[int, int]] = []
    immobile_mask = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if candidates[i]:
            start = i
            while i < n and candidates[i]:
                i += 1
            end = i - 1  # inclusive
            if (end - start + 1) >= min_bout_frames:
                bouts.append((start, end))
                immobile_mask[start : end + 1] = True
        else:
            i += 1

    return bouts, immobile_mask
