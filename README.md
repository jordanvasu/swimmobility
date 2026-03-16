# swimmobility

A lightweight, deterministic tool for measuring immobility in the *Drosophila* forced swim task (FST) from top-down video. No machine learning used, detection is based on frame differencing and  smoothing, making results fully reproducible from a fixed camera position.

## What it does

swimmobility processes video of one or more flies on a water surface and detects immobility bouts, periods where pixel displacement drops below a configurable threshold for a minimum sustained duration. For each fly, it outputs bout timing, summary statistics, and a verification figure showing the trajectory colored by behavioral state alongside the motion.
<img width="1800" height="1200" alt="fly_1_summary" src="https://github.com/user-attachments/assets/bea2d7d7-7b3c-4680-8a5f-dd17a0fe8ce7" />

## Intended use

Designed for *Drosophila* FST experiments filmed from above with a static camera. Each fly is filmed in its own arena; up to ten arenas can be processed from a single video. Arena boundaries are drawn manually using the built-in ROI tool before analysis. The pipeline is consistent in that the same video and configuration will always produce the same output.

## Install

git clone https://github.com/jordanvasu/swimmobility
cd swimmobility
pip install -e ".[video,plot,signal]"

## Basic usage

Draw arena ROIs:

swimmobility gui-roi --video your_video.mp4 --outdir out_trial01

Run analysis:

swimmobility run --video your_video.mp4 --outdir out_trial01 --trial-id t001 --n-flies 3

## Output

Each run produces a metadata file, per-fly immobility bout tables, a summary CSV, and a composite verification figure for each arena.

## Parameters

The key parameters to tune for your setup are `motion_threshold` (pixel-sum units, default 2.0), `smoothing_window_frames` (default 5), and `min_immobility_bout_s` (default 0.5 seconds). These can be set via config and are recorded in the run metadata for reproducibility. Expect to calibrate `motion_threshold` against pilot data before committing to a value.

## Requirements

Python 3.10+, OpenCV, NumPy, pandas, matplotlib. See `pyproject.toml` for full dependency details.

## License

MIT

## Contributors

Developed by Jordan Vasu with implementation assistance from Claude Code (Anthropic).
