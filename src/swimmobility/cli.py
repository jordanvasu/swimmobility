from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swimmobility._version import __version__
from swimmobility.config import SwimConfig, GuiConfig
from swimmobility.io.outputs import write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swimmobility")
    p.add_argument("--version", action="store_true", help="Print version and exit.")

    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Create deterministic run artifacts (MVP scaffold).")
    p_run.add_argument("--video", required=True, help="Path to input video.")
    p_run.add_argument("--outdir", required=True, help="Output directory.")
    p_run.add_argument("--trial-id", default=None, help="Optional trial identifier.")
    p_run.add_argument("--fps", type=float, default=None, help="FPS override (optional).")
    p_run.add_argument("--n-flies", type=int, default=1, help="Expected number of flies (1..10).")

    return p


def cmd_run(args: argparse.Namespace) -> int:
    if not (1 <= args.n_flies <= 10):
        raise SystemExit("--n-flies must be between 1 and 10.")

    cfg = SwimConfig(
        fps_override=args.fps,
        gui=GuiConfig(n_flies_expected=args.n_flies),
    )

    meta_path = write_run_metadata(
        outdir=Path(args.outdir),
        video_path=Path(args.video),
        cfg=cfg,
        trial_id=args.trial_id,
    )

    print(meta_path)
    print(f"config_hash={cfg.hash()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.cmd == "run":
        return cmd_run(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
