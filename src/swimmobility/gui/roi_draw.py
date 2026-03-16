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


def _draw_single_polygon(
    frame: np.ndarray,
    existing_polygons: List[List[Point]],
    fly_index: int,
    n_flies: int,
) -> List[Point]:
    """Interactively draw a single polygon on frame. Returns list of points."""
    points: List[Point] = []
    display = frame.copy()
    win = (
        f"Swimmobility ROI — Fly {fly_index + 1} of {n_flies} "
        "(L-click vertices, Enter=finish, r=reset, q=quit)"
    )

    def redraw() -> None:
        nonlocal display
        display = frame.copy()

        # Draw previously confirmed polygons in blue
        for prev_poly in existing_polygons:
            prev_pts = np.array(prev_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [prev_pts], isClosed=True, color=(255, 100, 0), thickness=2)

        # Draw current polygon in green
        for p in points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)

        if len(points) >= 2:
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(display, a, b, (0, 255, 0), 2)

        if len(points) >= 3:
            cv2.line(display, points[-1], points[0], (0, 255, 0), 1)

        cv2.putText(
            display,
            f"Fly {fly_index + 1}/{n_flies} — Points: {len(points)}",
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
            return points


def draw_polygon_roi_on_first_frame(video_path: str | Path) -> tuple[np.ndarray, List[Point]]:
    """
    Draw a single polygon ROI on the first frame of a video.

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

    points = _draw_single_polygon(frame, existing_polygons=[], fly_index=0, n_flies=1)
    return frame, points


def draw_multiple_polygon_rois(
    video_path: str | Path,
    n_flies: int,
) -> tuple[np.ndarray, List[List[Point]]]:
    """
    Draw N polygon ROIs sequentially on the first frame of a video.

    For each fly the user draws one polygon; previously confirmed polygons
    are displayed in blue so the user can avoid overlap.

    Returns (first_frame, list_of_polygons).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame.")

    all_polygons: List[List[Point]] = []
    for i in range(n_flies):
        poly = _draw_single_polygon(frame, existing_polygons=all_polygons, fly_index=i, n_flies=n_flies)
        all_polygons.append(poly)

    return frame, all_polygons


def save_roi_artifacts(
    frame: np.ndarray,
    polygon: List[Point],
    outdir: str | Path,
    basename: str = "roi",
) -> RoiArtifact:
    """Save a single-polygon ROI to roi.json and roi_mask.png (legacy / single-fly)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    bbox = _polygon_bbox(polygon)
    mask = _mask_from_polygon((h, w), polygon)

    json_path = outdir / f"{basename}.json"
    mask_path = outdir / f"{basename}_mask.png"

    payload = {
        "polygons": [
            {
                "fly_id": 1,
                "roi_polygon_xy": polygon,
                "roi_bbox_xywh": list(bbox),
                "frame_shape_hw": [h, w],
            }
        ]
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cv2.imwrite(str(mask_path), mask)

    return RoiArtifact(
        roi_polygon_xy=polygon,
        roi_bbox_xywh=bbox,
        roi_json_path=str(json_path),
        roi_mask_path=str(mask_path),
    )


def save_multiple_roi_artifacts(
    frame: np.ndarray,
    polygons: List[List[Point]],
    outdir: str | Path,
    basename: str = "roi",
) -> str:
    """Save multiple polygons to roi.json. Returns path to roi.json."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    json_path = outdir / f"{basename}.json"

    polygon_entries = []
    for idx, polygon in enumerate(polygons):
        fly_id = idx + 1
        bbox = _polygon_bbox(polygon)
        polygon_entries.append(
            {
                "fly_id": fly_id,
                "roi_polygon_xy": polygon,
                "roi_bbox_xywh": list(bbox),
                "frame_shape_hw": [h, w],
            }
        )

        # Write individual mask for each fly
        mask = _mask_from_polygon((h, w), polygon)
        mask_path = outdir / f"{basename}_fly{fly_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)

    payload = {"polygons": polygon_entries}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(json_path)


def load_roi_masks(
    roi_json_path: str | Path,
    frame_shape_hw: Tuple[int, int] | None = None,
) -> List[np.ndarray]:
    """
    Load ROI masks from a roi.json file.

    Supports the multi-polygon format: {"polygons": [...]}
    Falls back to legacy single-polygon format: {"roi_polygon_xy": [...], "frame_shape_hw": [...]}

    Returns list of uint8 masks (255 inside ROI, 0 outside), one per fly.
    """
    data = json.loads(Path(roi_json_path).read_text(encoding="utf-8"))

    if "polygons" in data:
        entries = data["polygons"]
    else:
        # Legacy single-polygon format
        entries = [data]

    masks = []
    for entry in entries:
        poly = entry["roi_polygon_xy"]
        shape = tuple(entry["frame_shape_hw"]) if "frame_shape_hw" in entry else frame_shape_hw
        if shape is None:
            raise ValueError("frame_shape_hw not found in roi.json and not provided as argument.")
        masks.append(_mask_from_polygon(shape, poly))

    return masks
