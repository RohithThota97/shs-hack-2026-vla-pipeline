"""Datasets that exist on this machine but aren't linked into data/raw yet: the organisers' download folder and
Google Drive (Colab mount, Drive for desktop), from `dataset.watch_dirs` in configs/project.yaml.

A dataset found there (a folder with meta/info.json, or a zip holding one) is added by linking it as
data/raw/<name> — the same layout the Colab notebook builds — so demo IDs are equal on every machine. A zip is
unzipped to data/cache/<name> first. Copies are recognised by their meta files (info.json + stats.json), so a
dataset already linked, or the zip of a folder that is already there, is not listed again.
"""
from __future__ import annotations

import glob
import hashlib
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from hackpipe.common.config import REPO_ROOT, project_config, source_key
from hackpipe.pipeline.sources import SEARCH_DIRS, dataset_name, extract_zip, find_roots

RAW = "data/raw"
CACHE = "data/cache"


def watch_dirs() -> list[Path]:
    """dataset.watch_dirs (~ and * expanded, relative to the repo); HACKPIPE_WATCH_DIRS (os.pathsep-separated,
    empty = none) overrides it."""
    env = os.environ.get("HACKPIPE_WATCH_DIRS")
    pats = ([p for p in env.split(os.pathsep) if p] if env is not None
            else project_config().get("dataset", {}).get("watch_dirs") or [])
    out: list[Path] = []
    for pat in pats:
        p = os.path.expanduser(str(pat))
        p = p if os.path.isabs(p) else str(REPO_ROOT / p)
        out += [Path(m) for m in sorted(glob.glob(p)) if os.path.isdir(m)]
    return out


def pinned_names() -> dict[tuple[int, int], str]:
    """dataset.pinned_names: {name: [episodes, frames]} → {(episodes, frames): name} — names that keep ds-v1's
    demo IDs. Both numbers must match, so another dataset with 84 episodes doesn't take the name."""
    pins = project_config().get("dataset", {}).get("pinned_names") or {}
    return {(int(v[0]), int(v[1])): name for name, v in pins.items()}


def _fp(info: bytes, stats: bytes | None) -> str:
    return hashlib.sha256(info + b"\0" + (stats or b"")).hexdigest()[:16]


def folder_fingerprint(root: Path) -> str:
    stats = root / "meta" / "stats.json"
    return _fp((root / "meta" / "info.json").read_bytes(), stats.read_bytes() if stats.exists() else None)


def zip_datasets(path: Path) -> list[dict[str, Any]]:
    """Datasets inside a zip (read from its directory, nothing extracted): [{inner, fingerprint, info}]."""
    import json
    out = []
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for m in sorted(n for n in names if n.endswith("meta/info.json") and "__MACOSX" not in n):
                inner = str(PurePosixPath(m).parent.parent)
                stats = f"{inner}/meta/stats.json" if inner != "." else "meta/stats.json"
                info = z.read(m)
                out.append({"inner": "" if inner == "." else inner,
                            "fingerprint": _fp(info, z.read(stats) if stats in names else None),
                            "info": json.loads(info)})
    except (zipfile.BadZipFile, OSError, ValueError):
        pass
    return out


def _zips(top: Path, max_depth: int = 4) -> list[Path]:
    out = []
    for dirpath, dirs, files in os.walk(top):
        depth = len(Path(dirpath).relative_to(top).parts)
        dirs[:] = [] if depth >= max_depth or (Path(dirpath) / "meta" / "info.json").exists() else \
            [d for d in dirs if not d.startswith(".")]
        out += [Path(dirpath) / f for f in sorted(files) if f.endswith(".zip") and not f.startswith("._")]
    return out


def linked(base: Path = REPO_ROOT) -> dict[str, str]:
    """fingerprint → source key of every dataset already under data/raw and data/incoming."""
    out = {}
    for d in SEARCH_DIRS:
        for r in find_roots(base / d):
            try:
                out.setdefault(folder_fingerprint(r), source_key(r))
            except OSError:
                continue
    return out


def link_name(info: dict[str, Any], path: str | Path) -> str:
    key = (int(info.get("total_episodes", -1)), int(info.get("total_frames", -1)))
    return pinned_names().get(key) or dataset_name(path)


def unlinked(base: Path = REPO_ROOT, dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    """Datasets in the watch dirs that aren't in data/raw / data/incoming yet, one entry per distinct dataset:
    {path, kind: folder | zip, inner, name, episodes, frames, fps, fingerprint}. Folders win over zips."""
    import json
    have = linked(base)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    dirs = watch_dirs() if dirs is None else dirs
    for top in dirs:
        for r in find_roots(top):
            try:
                fp, info = folder_fingerprint(r), json.loads((r / "meta" / "info.json").read_text())
            except (OSError, ValueError):
                continue
            if fp in have or fp in seen:
                continue
            seen.add(fp)
            out.append({"path": str(r), "kind": "folder", "inner": "", "name": link_name(info, r),
                        "episodes": info.get("total_episodes"), "frames": info.get("total_frames"),
                        "fps": info.get("fps"), "fingerprint": fp})
    for top in dirs:
        for z in _zips(top):
            for d in zip_datasets(z):
                if d["fingerprint"] in have or d["fingerprint"] in seen:
                    continue
                seen.add(d["fingerprint"])
                info = d["info"]
                out.append({"path": str(z), "kind": "zip", "inner": d["inner"],
                            "name": link_name(info, d["inner"] or z), "episodes": info.get("total_episodes"),
                            "frames": info.get("total_frames"), "fps": info.get("fps"),
                            "fingerprint": d["fingerprint"]})
    return out


def link(path: str | Path, name: str | None = None, base: Path = REPO_ROOT) -> Path:
    """Add a found dataset as data/raw/<name> (zip: unzipped to data/cache/<name> first). Returns the link."""
    import json
    p = Path(os.path.abspath(Path(path).expanduser()))
    if p.is_file() and p.suffix == ".zip":
        found = zip_datasets(p)
        if not found:
            raise FileNotFoundError(f"no LeRobot dataset (meta/info.json) in {p}")
        if len(found) > 1:
            raise ValueError(f"{p} holds {len(found)} datasets — unzip it and add the folders one by one")
        name = name or link_name(found[0]["info"], found[0]["inner"] or p)
        cache = base / CACHE / name
        if not (cache / ".complete").exists():
            import shutil
            shutil.rmtree(cache, ignore_errors=True)          # half-finished earlier attempt
            cache.mkdir(parents=True)
            extract_zip(p, cache)
            (cache / ".complete").touch()
        root = cache / found[0]["inner"] if found[0]["inner"] else cache
    else:
        if not (p / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"not a LeRobot v3 dataset folder (no meta/info.json): {p}")
        root = p
        name = name or link_name(json.loads((p / "meta" / "info.json").read_text()), p)
    dest = base / RAW / name
    if dest.is_symlink() and dest.resolve() == root.resolve():
        return dest                                           # already linked
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"{RAW}/{name} already exists and is a different dataset — pick another name")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(os.path.relpath(root, dest.parent) if root.is_relative_to(base) else root,
                    target_is_directory=True)
    return dest
