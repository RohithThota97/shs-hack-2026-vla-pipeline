"""Move the pipeline's state between machines that don't share a folder (a laptop without Drive sync ↔ Colab).

`export` writes one zip: a consistent copy of registry.db (SQLite backup API) plus the small outputs that
versions, labels and QC refer to (outputs/datasets, labels, qc, runs, eval). No checkpoints, no data. `import`
unpacks it on the other machine, after backing up whatever registry is there; the writer role moves with it
(the exporting machine must not write until it imports the state back).

Paths inside the registry are repo-relative (or fall back to <outputs>/datasets/<version>/), so the same zip
works in /content/shs-hack-2026-vla-pipeline and /Users/…/shs-hack-2026-vla-pipeline.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.ids import now_iso

OUTPUT_PARTS = ("datasets", "labels", "qc", "runs", "eval")


def _paths() -> tuple[Path, Path]:
    cfg = project_config()["project"]
    return resolve(cfg["registry_path"]), resolve(cfg["outputs_dir"])


def summary(db: Path) -> dict[str, Any]:
    con = sqlite3.connect(db)
    try:
        q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
        return {"demos": q("SELECT COUNT(*) FROM demos")[0][0],
                "datasets": [{"version": v, "status": s, "fingerprint": f}
                             for v, s, f in q("SELECT version, status, fingerprint FROM datasets ORDER BY created_at")],
                "models": [v for (v,) in q("SELECT version FROM models ORDER BY created_at")]}
    finally:
        con.close()


def export(out_zip: str | Path) -> dict[str, Any]:
    from hackpipe.common import writelock
    db, outputs = _paths()
    if not db.exists():
        raise FileNotFoundError(f"no registry at {db}")
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "registry.db"
        src, dst = sqlite3.connect(db), sqlite3.connect(copy)
        with dst:
            src.backup(dst)                               # consistent even while the console is open
        src.close()
        dst.close()
        info = {"exported_at": now_iso(), "from_machine": writelock.machine_id(), **summary(copy)}
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(copy, "registry/registry.db")
            for part in OUTPUT_PARTS:
                for f in sorted((outputs / part).rglob("*")) if (outputs / part).is_dir() else []:
                    if f.is_file():
                        z.write(f, f"outputs/{f.relative_to(outputs).as_posix()}")
            z.writestr("state.json", json.dumps(info, indent=1))
    return {"zip": str(out_zip), "size_mb": round(out_zip.stat().st_size / 1e6, 1), **info}


def import_state(zip_path: str | Path, replace: bool = False) -> dict[str, Any]:
    from hackpipe.common import writelock
    db, outputs = _paths()
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if "registry/registry.db" not in names or "state.json" not in names:
            raise ValueError(f"{zip_path} is not a `hackpipe state export` zip")
        bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
        if bad:
            raise ValueError(f"unsafe paths in {zip_path}: {bad[:3]}")
        info = json.loads(z.read("state.json"))
        backup = None
        if db.exists():
            have = summary(db)
            if (have["demos"] or have["datasets"]) and not replace:
                raise FileExistsError(f"{db} already has {have['demos']} demos and versions "
                                      f"{[d['version'] for d in have['datasets']]}; import with --replace "
                                      "(it is backed up first)")
            backup = db.parent / f"backup-{now_iso().replace(':', '')}"
            backup.mkdir(parents=True)
            shutil.copy2(db, backup / "registry.db")
            for part in OUTPUT_PARTS:
                if (outputs / part).is_dir():
                    shutil.copytree(outputs / part, backup / "outputs" / part)
        db.parent.mkdir(parents=True, exist_ok=True)
        with z.open("registry/registry.db") as f, open(db, "wb") as out:
            shutil.copyfileobj(f, out)
        for n in names:
            if n.startswith("outputs/") and not n.endswith("/"):
                target = outputs / Path(n).relative_to("outputs")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as f, open(target, "wb") as out:
                    shutil.copyfileobj(f, out)
    writelock.acquire(force=True)            # the exporting machine handed the registry over
    return {"imported": str(zip_path), "registry": str(db), "backup": str(backup) if backup else None,
            "writer": writelock.machine_id(), **{k: info[k] for k in ("exported_at", "from_machine", "datasets")}}
