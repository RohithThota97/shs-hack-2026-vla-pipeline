"""IDs, checksums and fingerprints for traceability."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Fixed namespace so the same episode always gets the same demo_id (re-imports do not duplicate).
DEMO_NAMESPACE = uuid.UUID("6f1c3a52-6a3e-4d5e-9b1a-1f6c2b7d9e10")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def demo_id(source: str, episode_index: int) -> str:
    """Deterministic UUID for one episode of one source dataset."""
    return str(uuid.uuid5(DEMO_NAMESPACE, f"{source}:{episode_index}"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(items: Iterable[str], config: dict[str, Any] | None = None) -> str:
    """Hash of sorted item checksums plus a config dict. Same inputs → same fingerprint."""
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(item.encode())
        h.update(b"\n")
    h.update(canonical_json(config or {}).encode())
    return h.hexdigest()[:16]


def next_version(existing: Iterable[str], prefix: str) -> str:
    """next_version(["ds-v1", "ds-v2"], "ds") -> "ds-v3"."""
    nums = []
    for v in existing:
        if v.startswith(f"{prefix}-v"):
            try:
                nums.append(int(v.split("-v", 1)[1]))
            except ValueError:
                pass
    return f"{prefix}-v{max(nums, default=0) + 1}"
