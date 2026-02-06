from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import os

from swimmobility._version import __version__
from swimmobility.config import SwimConfig


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_run_metadata(
    outdir: str | Path,
    video_path: str | Path,
    cfg: SwimConfig,
    trial_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vp = Path(video_path)
    payload: dict[str, Any] = {
        "analysis_version": __version__,
        "definition_version": cfg.definition_version,
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "video_path": str(vp),
        "trial_id": trial_id,
        "cwd": os.getcwd(),
    }

    if vp.exists():
        payload["video_hash_sha256"] = sha256_file(vp)
    else:
        payload["video_hash_sha256"] = None
        payload["warnings"] = ["video_path_does_not_exist"]

    if extra:
        payload["extra"] = extra

    p = outdir / "run_metadata.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p
