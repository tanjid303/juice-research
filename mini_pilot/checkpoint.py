"""Atomic JSONL checkpoints so a killed Kaggle session does not lose generations."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_utc": datetime.now(timezone.utc).isoformat()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def done_ids(records: list[dict], precision: str) -> set[str]:
    return {str(r["id"]) for r in records if r.get("precision") == precision}


def find_resume_checkpoint(primary: Path, extra_roots: list[Path] | None = None) -> Path | None:
    """Prefer /kaggle/working, then a previously uploaded dataset under /kaggle/input."""
    candidates: list[Path] = [primary]
    for root in extra_roots or []:
        if not root.exists():
            continue
        if root.is_file() and root.name.endswith(".jsonl"):
            candidates.append(root)
            continue
        candidates.extend(root.rglob("checkpoint_8b.jsonl"))
        candidates.extend(root.rglob("detail_*.jsonl"))
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None
