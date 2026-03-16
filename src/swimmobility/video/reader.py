from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


def get_video_info(
    video_path: str | Path,
    fps_override: float | None = None,
) -> Tuple[float, int, Tuple[int, int]]:
    """
    Return (fps, n_frames, (height, width)) for a video file.

    Parameters
    ----------
    video_path:
        Path to the video file.
    fps_override:
        If given, returned FPS is this value instead of the metadata FPS.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    fps = fps_override if fps_override is not None else meta_fps
    if fps <= 0:
        raise ValueError(f"Invalid FPS {fps} from video {video_path}. Provide --fps override.")

    return fps, n_frames, (h, w)


def read_frames(
    video_path: str | Path,
    fps_override: float | None = None,
    trim_start_s: float = 0.0,
    trim_end_s: float | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames as uint8 numpy arrays.

    Parameters
    ----------
    video_path:
        Path to the video file.
    fps_override:
        If given, used for computing trim frame indices instead of metadata FPS.
    trim_start_s:
        Seconds to skip at the start of the video.
    trim_end_s:
        Seconds to skip at the end of the video (None = no trimming).
    """
    fps, n_frames, _ = get_video_info(video_path, fps_override=fps_override)

    start_frame = int(trim_start_s * fps)
    if trim_end_s is not None:
        end_frame = max(start_frame, n_frames - int(trim_end_s * fps))
    else:
        end_frame = n_frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield frame

    cap.release()
