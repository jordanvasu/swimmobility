from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class RoiArtifact:
    roi_polygon_xy: List[Point]
    roi_bbox_xywh: Tuple[int, int, int, int]  # x,y,w,h
    roi_json_path: str
    roi_mask_path: str


def _polygon_bbox(poly: List[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _mask_from_polygon(shape_hw: Tuple[int, int], poly: List[Point]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def draw_polygon_roi_on_first_frame(video_path: str | Path) -> tuple[np.ndarray, List[Point]]:
    """
    Draw polygon ROI on the first frame of a video.

    Mouse:
      - Left click: add vertex

    Keys:
      - Enter: finish (requires >= 3 points)
      - Backspace/Delete: remove last point
      - r: reset
      - q or Esc: cancel
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame.")

    points: List[Point] = []
    display = frame.copy()
    win = "Swimmobility ROI (L-click vertices, Enter=finish, r=reset, q=quit)"

    def redraw() -> None:
        nonlocal display
        display = frame.copy()

        for p in points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)

        if len(points) >= 2:
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(display, a, b, (0, 255, 0), 2)

        if len(points) >= 3:
            cv2.line(display, points[-1], points[0], (0, 255, 0), 1)

        cv2.putText(
            display,
            f"Points: {len(points)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(win)
            raise RuntimeError("ROI drawing cancelled by user.")
        if key == ord("r"):
            points.clear()
            redraw()
        if key in (8, 127):
            if points:
                points.pop()
                redraw()
        if key in (10, 13):
            if len(points) < 3:
                continue
            cv2.destroyWindow(win)
            return frame, points


def save_roi_artifacts(
    frame: np.ndarray,
    polygon: List[Point],
    outdir: str | Path,
    basename: str = "roi",
) -> RoiArtifact:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    bbox = _polygon_bbox(polygon)
    mask = _mask_from_polygon((h, w), polygon)

    json_path = outdir / f"{basename}.json"
    mask_path = outdir / f"{basename}_mask.png"

    payload = {
        "roi_polygon_xy": polygon,
        "roi_bbox_xywh": bbox,
        "frame_shape_hw": [h, w],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cv2.imwrite(str(mask_path), mask)

    return RoiArtifact(
        roi_polygon_xy=polygon,
        roi_bbox_xywh=bbox,
        roi_json_path=str(json_path),
        roi_mask_path=str(mask_path),
    )
