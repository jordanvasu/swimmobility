from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class TableSchema:
    name: str
    columns: List[str]


EVENTS_SCHEMA = TableSchema(
    name="events.csv",
    columns=[
        "trial_id",
        "video_path",
        "fly_id",
        "fps",
        "bout_idx",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "thr_value",
        "definition_version",
        "identity_confidence",
        "n_identity_breaks",
        "notes",
    ],
)

SUMMARY_BY_FLY_SCHEMA = TableSchema(
    name="summary_by_fly.csv",
    columns=[
        "trial_id",
        "video_path",
        "fly_id",
        "fps",
        "duration_s",
        "effective_duration_s",
        "total_immobile_s",
        "immobile_fraction",
        "n_bouts",
        "latency_to_first_immobility_s",
        "mean_bout_duration_s",
        "median_bout_duration_s",
        "max_bout_duration_s",
        "track_coverage_fraction",
        "n_gaps",
        "max_gap_s",
        "n_merge_events",
        "n_split_events",
        "identity_confidence",
        "warnings",
    ],
)

SUMMARY_SCHEMA = TableSchema(
    name="summary.csv",
    columns=[
        "trial_id",
        "video_path",
        "analysis_version",
        "definition_version",
        "config_hash",
        "video_hash_sha256",
        "fps",
        "n_frames",
        "duration_s",
        "effective_duration_s",
        "n_flies_expected",
        "n_flies_tracked",
        "mean_track_coverage_fraction",
        "total_immobile_s_sum_flies",
        "warnings",
    ],
)

TRACKING_EVENTS_SCHEMA = TableSchema(
    name="tracking_events.csv",
    columns=[
        "trial_id",
        "frame",
        "event_type",
        "fly_id",
        "details_json",
    ],
)
