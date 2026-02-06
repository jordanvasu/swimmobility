from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal
import hashlib
import json


def _canonical_json(obj: Any) -> str:
    """Stable JSON for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RoiConfig:
    mode: Literal["full_frame", "polygon_mask"] = "full_frame"
    roi_json_path: str | None = None
    roi_mask_path: str | None = None


@dataclass(frozen=True, slots=True)
class GuiConfig:
    n_flies_expected: int = 1  # 1..10
    init_window_s: float = 2.0


@dataclass(frozen=True, slots=True)
class SwimConfig:
    definition_version: str = "immobility_v1"

    fps_override: float | None = None
    trim_start_s: float = 0.0
    trim_end_s: float | None = None

    roi: RoiConfig = field(default_factory=RoiConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def hash(self) -> str:
        return sha256_text(self.to_canonical_json())

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_canonical_json(), encoding="utf-8")
