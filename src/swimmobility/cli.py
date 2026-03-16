from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from swimmobility._version import __version__
from swimmobility.config import SwimConfig, GuiConfig
from swimmobility.io.outputs import write_run_metadata
from swimmobility.gui.roi_draw import (
    draw_multiple_polygon_rois,
    draw_polygon_roi_on_first_frame,
    load_roi_masks,
    save_roi_artifacts,
    save_multiple_roi_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swimmobility")
    p.add_argument("--version", action="store_true", help="Print version and exit.")

    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run the immobility detection pipeline.")
    p_run.add_argument("--video", required=True, help="Path to input video.")
    p_run.add_argument("--outdir", required=True, help="Output directory.")
    p_run.add_argument("--trial-id", default=None, help="Optional trial identifier.")
    p_run.add_argument("--fps", type=float, default=None, help="FPS override (optional).")
    p_run.add_argument("--n-flies", type=int, default=1, help="Expected number of flies (1..10).")
    p_run.add_argument("--roi", default=None, help="Path to roi.json with polygon ROI(s).")

    p_roi = sub.add_parser("gui-roi", help="Draw polygon ROI(s) on first frame and save roi.json.")
    p_roi.add_argument("--video", required=True, help="Path to input video.")
    p_roi.add_argument("--outdir", required=True, help="Directory to save ROI artifacts.")
    p_roi.add_argument("--n-flies", type=int, default=1, help="Number of polygon ROIs to draw.")

    return p


def cmd_run(args: argparse.Namespace) -> int:
    if not (1 <= args.n_flies <= 10):
        raise SystemExit("--n-flies must be between 1 and 10.")

    cfg = SwimConfig(
        fps_override=args.fps,
        gui=GuiConfig(n_flies_expected=args.n_flies),
    )

    outdir = Path(args.outdir)
    video_path = Path(args.video)

    meta_path = write_run_metadata(
        outdir=outdir,
        video_path=video_path,
        cfg=cfg,
        trial_id=args.trial_id,
    )
    print(meta_path)
    print(f"config_hash={cfg.hash()}")

    # Import here to keep startup fast and allow the package to be imported
    # without optional dependencies installed.
    from swimmobility.video.reader import get_video_info, read_frames
    from swimmobility.video.motion import compute_motion_energy
    from swimmobility.detection.immobility import detect_immobility

    if not video_path.exists():
        print(f"WARNING: video path does not exist: {video_path}. Skipping pipeline.", file=sys.stderr)
        return 0

    fps, n_frames, frame_shape_hw = get_video_info(video_path, fps_override=cfg.fps_override)

    # Load ROI masks
    if args.roi is not None:
        roi_masks = load_roi_masks(args.roi, frame_shape_hw=frame_shape_hw)
    else:
        # Full-frame mask
        h, w = frame_shape_hw
        full_mask = np.full((h, w), 255, dtype=np.uint8)
        roi_masks = [full_mask] * args.n_flies

    events_rows = []
    summary_rows = []

    for fly_idx, roi_mask in enumerate(roi_masks):
        fly_id = fly_idx + 1

        frames = read_frames(
            video_path,
            fps_override=cfg.fps_override,
            trim_start_s=cfg.trim_start_s,
            trim_end_s=cfg.trim_end_s,
        )
        raw_energy, smoothed_energy, centroids = compute_motion_energy(
            frames,
            roi_mask=roi_mask,
            smoothing_kernel_size=cfg.smoothing_window_frames,
        )

        bouts, immobile_mask = detect_immobility(
            smoothed_energy,
            fps=fps,
            threshold=cfg.motion_threshold,
            min_bout_s=cfg.min_immobility_bout_s,
        )

        # Summary figure
        try:
            from swimmobility.plot.summary_figure import generate_summary_figure

            fig_path = generate_summary_figure(
                video_path=video_path,
                fly_id=fly_id,
                roi_mask=roi_mask,
                raw_energy=raw_energy,
                smoothed_energy=smoothed_energy,
                immobility_bouts=bouts,
                immobile_mask=immobile_mask,
                fps=fps,
                threshold=cfg.motion_threshold,
                centroids=centroids,
                outdir=outdir,
            )
            print(f"fly_{fly_id}_summary: {fig_path}")
        except ImportError:
            print("WARNING: matplotlib not installed; skipping summary figure.", file=sys.stderr)

        # Collect bout events
        for bout_idx, (start_f, end_f) in enumerate(bouts):
            events_rows.append(
                {
                    "trial_id": args.trial_id,
                    "video_path": str(video_path),
                    "fly_id": fly_id,
                    "fps": fps,
                    "bout_idx": bout_idx,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "start_time_s": start_f / fps,
                    "end_time_s": end_f / fps,
                    "duration_s": (end_f - start_f + 1) / fps,
                    "thr_value": cfg.motion_threshold,
                    "definition_version": cfg.definition_version,
                    "identity_confidence": None,
                    "n_identity_breaks": None,
                    "notes": None,
                }
            )

        # Per-fly summary stats
        n_energy = len(smoothed_energy)
        duration_s = n_energy / fps if fps > 0 else 0.0
        total_immobile_s = float(immobile_mask.sum()) / fps
        bout_durations_s = [(e - s + 1) / fps for s, e in bouts]

        summary_rows.append(
            {
                "trial_id": args.trial_id,
                "video_path": str(video_path),
                "fly_id": fly_id,
                "fps": fps,
                "duration_s": duration_s,
                "effective_duration_s": duration_s,
                "total_immobile_s": total_immobile_s,
                "immobile_fraction": total_immobile_s / duration_s if duration_s > 0 else None,
                "n_bouts": len(bouts),
                "latency_to_first_immobility_s": bouts[0][0] / fps if bouts else None,
                "mean_bout_duration_s": float(np.mean(bout_durations_s)) if bout_durations_s else None,
                "median_bout_duration_s": float(np.median(bout_durations_s)) if bout_durations_s else None,
                "max_bout_duration_s": float(max(bout_durations_s)) if bout_durations_s else None,
                "track_coverage_fraction": None,
                "n_gaps": None,
                "max_gap_s": None,
                "n_merge_events": None,
                "n_split_events": None,
                "identity_confidence": None,
                "warnings": None,
            }
        )

    # Write events.csv
    from swimmobility.io.schemas import EVENTS_SCHEMA, SUMMARY_BY_FLY_SCHEMA

    events_df = pd.DataFrame(events_rows, columns=EVENTS_SCHEMA.columns) if events_rows else pd.DataFrame(columns=EVENTS_SCHEMA.columns)
    events_path = outdir / "events.csv"
    events_df.to_csv(events_path, index=False)
    print(f"events: {events_path}")

    # Write summary_by_fly.csv
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_BY_FLY_SCHEMA.columns) if summary_rows else pd.DataFrame(columns=SUMMARY_BY_FLY_SCHEMA.columns)
    summary_path = outdir / "summary_by_fly.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"summary_by_fly: {summary_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.cmd == "run":
        return cmd_run(args)

    if args.cmd == "gui-roi":
        if not (1 <= args.n_flies <= 10):
            raise SystemExit("--n-flies must be between 1 and 10.")

        if args.n_flies == 1:
            frame, poly = draw_polygon_roi_on_first_frame(args.video)
            art = save_roi_artifacts(frame, poly, args.outdir)
            print(art.roi_json_path)
            print(art.roi_mask_path)
        else:
            frame, polygons = draw_multiple_polygon_rois(args.video, n_flies=args.n_flies)
            roi_json_path = save_multiple_roi_artifacts(frame, polygons, args.outdir)
            print(roi_json_path)

        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
