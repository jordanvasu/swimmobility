from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d as _scipy_gaussian

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _smooth(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply temporal smoothing to a 1-D float array."""
    if kernel_size <= 1 or len(x) == 0:
        return x.copy().astype(float)
    if _HAS_SCIPY:
        sigma = max(kernel_size / 4.0, 0.5)
        return _scipy_gaussian(x.astype(float), sigma=sigma)
    # Fallback: uniform convolution, same-length output
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(x.astype(float), kernel, mode="same")


def compute_motion_energy(
    frames: Iterable[np.ndarray],
    roi_mask: Optional[np.ndarray],
    smoothing_kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[Tuple[float, float]]]]:
    """
    Compute per-frame motion energy from a sequence of video frames.

    Converts each frame to grayscale, computes the absolute difference
    between consecutive frames, masks to the ROI, and sums the result to
    produce a scalar motion energy value per frame.

    Parameters
    ----------
    frames:
        Iterable of BGR (or grayscale) uint8 numpy arrays, all the same shape.
    roi_mask:
        Binary uint8 mask (255 = inside ROI, 0 = outside), same H×W as frames.
        Pass None to use the full frame.
    smoothing_kernel_size:
        Kernel size for temporal smoothing (Gaussian if scipy available,
        otherwise uniform). Set to 1 or 0 to disable.

    Returns
    -------
    raw_energy:
        Float64 array of length N-1 (one value per consecutive frame pair).
    smoothed_energy:
        Temporally smoothed version of raw_energy, same length.
    centroids:
        List of (cx, cy) float tuples giving the centre-of-mass of the
        motion-active region within the ROI for each frame difference.
        Entries are None where the masked difference is all-zero.
    """
    raw: List[float] = []
    centroids: List[Optional[Tuple[float, float]]] = []
    prev_gray: Optional[np.ndarray] = None

    for frame in frames:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)

            if roi_mask is not None:
                masked = cv2.bitwise_and(diff, roi_mask)
            else:
                masked = diff

            raw.append(float(masked.sum()))

            # Centre-of-mass (proxy position)
            M = cv2.moments(masked.astype(np.float32))
            if M["m00"] > 0:
                centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
            else:
                centroids.append(None)

        prev_gray = gray

    raw_arr = np.array(raw, dtype=np.float64)
    smoothed_arr = _smooth(raw_arr, smoothing_kernel_size)
    return raw_arr, smoothed_arr, centroids
