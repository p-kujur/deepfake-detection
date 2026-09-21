"""Metrics.json writer for artifacts/runs/<id>/metrics.json."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MetricsRecord:
    dataset: str
    split: str
    model_id: str
    acc: float | None
    ap: float | None
    auc: float | None
    n: int
    threshold: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    aug: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    config_hash: str | None = None
    notes: str | None = None
    config_path: str | None = None
    created_at: str | None = None

    def finalize(self) -> dict[str, Any]:
        if self.git_commit is None:
            self.git_commit = _git_commit()
        if self.config_hash is None and self.config_path:
            self.config_hash = _file_hash(self.config_path)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        d = asdict(self)
        d.pop("config_path", None)
        return d


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return "missing"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def write_metrics(
    record: MetricsRecord,
    run_id: str | None = None,
    output_dir: str | Path = "artifacts/runs",
) -> Path:
    """Write metrics.json under artifacts/runs/<id>/ and return the path."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    payload = record.finalize()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
