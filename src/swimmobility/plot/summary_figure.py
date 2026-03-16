from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend; must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _sample_background(
    video_path: str | Path,
    n_samples: int = 4,
) -> Optional[np.ndarray]:
    """
    Return a composite background by max-projecting *n_samples* evenly spaced
    frames. Returns None if the video cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None
    return np.max(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _fill_centroid_gaps(
    centroids: List[Optional[Tuple[float, float]]],
) -> List[Optional[Tuple[float, float]]]:
    """Forward-fill None centroid entries from the previous known value."""
    filled = list(centroids)
    last: Optional[Tuple[float, float]] = None
    for i, c in enumerate(filled):
        if c is not None:
            last = c
        elif last is not None:
            filled[i] = last
    return filled


def generate_summary_figure(
    video_path: str | Path,
    fly_id: int,
    roi_mask: Optional[np.ndarray],
    raw_energy: np.ndarray,
    smoothed_energy: np.ndarray,
    immobility_bouts: List[Tuple[int, int]],
    immobile_mask: np.ndarray,
    fps: float,
    threshold: float,
    centroids: List[Optional[Tuple[float, float]]],
    outdir: str | Path,
) -> Path:
    """
    Produce a composite PNG for one arena/fly and save it to *outdir*.

    Top panel — spatial trajectory:
        A max-projection of 4 evenly spaced video frames is used as the
        background.  The centroid of the motion-active region is plotted
        as a trajectory line coloured continuously from blue (mobile) to
        red (immobile).

    Bottom panel — motion energy time series:
        Raw energy in light gray, smoothed energy in black, horizontal
        threshold line in orange, and immobility bouts shaded in red.

    Parameters
    ----------
    video_path:
        Source video (re-read for background frames only).
    fly_id:
        1-based fly identifier used in the output filename.
    roi_mask:
        uint8 mask (255 = ROI); used to draw the ROI boundary. May be None.
    raw_energy, smoothed_energy:
        Motion energy arrays of length N-1.
    immobility_bouts:
        List of (start_frame, end_frame) inclusive pairs.
    immobile_mask:
        Boolean array of length N-1.
    fps, threshold:
        Used for axis scaling and the threshold line.
    centroids:
        List of (cx, cy) or None, length N-1.
    outdir:
        Directory where the figure is saved.

    Returns
    -------
    Path to the saved PNG.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"fly_{fly_id}_summary.png"

    n = len(smoothed_energy)
    times = np.arange(n) / fps  # seconds, one per frame-difference index

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ------------------------------------------------------------------
    # Top panel: spatial trajectory
    # ------------------------------------------------------------------
    ax_top = axes[0]
    bg = _sample_background(video_path, n_samples=4)
    if bg is not None:
        ax_top.imshow(bg, origin="upper", aspect="equal")
    else:
        ax_top.set_facecolor("#111111")

    # Draw ROI boundary
    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            xy = cnt[:, 0, :]
            ax_top.plot(
                np.append(xy[:, 0], xy[0, 0]),
                np.append(xy[:, 1], xy[0, 1]),
                color="yellow",
                linewidth=1.0,
                alpha=0.6,
            )

    # Build trajectory from centroids
    filled = _fill_centroid_gaps(centroids)
    valid_mask = np.array([c is not None for c in filled])
    if valid_mask.any():
        xs = np.array([c[0] if c is not None else np.nan for c in filled])
        ys = np.array([c[1] if c is not None else np.nan for c in filled])

        # Map immobility to colour: blue=mobile (0), red=immobile (1)
        color_vals = immobile_mask.astype(float)

        # Build line segments for LineCollection
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = color_vals[:-1] if len(color_vals) > 1 else color_vals

        lc = LineCollection(segments, cmap="RdBu_r", norm=plt.Normalize(0, 1), linewidth=1.5, alpha=0.8)
        lc.set_array(seg_colors)
        ax_top.add_collection(lc)

    ax_top.set_title(f"Fly {fly_id} — spatial trajectory (blue=mobile, red=immobile)")
    ax_top.set_xlabel("x (px)")
    ax_top.set_ylabel("y (px)")
    ax_top.autoscale()

    # ------------------------------------------------------------------
    # Bottom panel: motion energy time series
    # ------------------------------------------------------------------
    ax_bot = axes[1]

    ax_bot.plot(times, raw_energy, color="#cccccc", linewidth=0.8, label="raw")
    ax_bot.plot(times, smoothed_energy, color="black", linewidth=1.2, label="smoothed")
    ax_bot.axhline(threshold, color="orange", linewidth=1.0, linestyle="--", label=f"threshold={threshold}")

    # Shade immobility bouts
    for start_f, end_f in immobility_bouts:
        t0 = start_f / fps
        t1 = end_f / fps
        ax_bot.axvspan(t0, t1, color="red", alpha=0.2)

    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Motion energy (pixel-sum)")
    ax_bot.set_title(f"Fly {fly_id} — motion energy")
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.set_xlim(0, times[-1] if n > 0 else 1)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)

    return out_path
