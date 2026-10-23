"""SQLite registry linking demos → datasets → models → evaluations → deployments (R9).

One file (registry/registry.db), no server. Every row keeps its full metadata as JSON.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.ids import now_iso

def portable(path: str | Path) -> str:
    """Repo-relative inside the repo (the same on a laptop and on Colab), absolute otherwise."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    from hackpipe.common.config import REPO_ROOT
    for root in {REPO_ROOT, REPO_ROOT.resolve()}:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.as_posix()


def local_path(p: str | None) -> str | None:
    """A stored path on this machine: as is if it exists; a path from another checkout (…/outputs/… or
    …/registry/…, e.g. /content/repo/outputs/train/m-v3 read on a laptop) under this checkout's folders."""
    if not p:
        return p
    q = resolve(p)
    if q.exists():
        return str(q)
    parts = Path(p).parts
    if "outputs" in parts:
        rest = parts[len(parts) - 1 - parts[::-1].index("outputs") + 1:]
        return str(resolve(project_config()["project"]["outputs_dir"]).joinpath(*rest))
    return str(q)


MODEL_META_PATHS = ("output_dir", "train_config_file")


def _model_paths(row: dict[str, Any]) -> dict[str, Any]:
    meta = {**row["meta"], **{k: local_path(row["meta"][k]) for k in MODEL_META_PATHS if row["meta"].get(k)}}
    return {**row, "checkpoint_dir": local_path(row["checkpoint_dir"]),
            "run_record_path": local_path(row["run_record_path"]), "meta": meta}


def _dataset_paths(row: dict[str, Any]) -> dict[str, Any]:
    """Absolute manifest / card paths for this machine: this machine's <outputs>/datasets/<version>/ copy when
    there is one (registry and outputs move together, `hackpipe state`), else the stored path — so rows written
    on another checkout (e.g. /Users/… read on Colab) still resolve."""
    out_dir = resolve(project_config()["project"]["outputs_dir"]) / "datasets" / row["version"]

    def here(p: str | None, name: str) -> str | None:
        if p is None:
            return None
        return str(out_dir / name if (out_dir / name).exists() else resolve(p))

    row = {**row, "manifest_path": here(row["manifest_path"], "manifest.json")}
    if row["meta"].get("card"):
        row["meta"] = {**row["meta"], "card": here(row["meta"]["card"], "dataset_card.md")}
    return row


SCHEMA = """
CREATE TABLE IF NOT EXISTS demos (
    demo_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    episode_index INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'imported',
    meta TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    version TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    meta TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    status TEXT NOT NULL,
    checkpoint_dir TEXT,
    run_record_path TEXT,
    meta TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evals (
    eval_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    test_suite TEXT NOT NULL,
    env_version TEXT NOT NULL,
    results_dir TEXT NOT NULL,
    metrics TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    needs TEXT NOT NULL,
    machine TEXT NOT NULL,
    pid INTEGER,
    log_path TEXT,
    work_units REAL,
    estimate_s REAL,
    exit_code INTEGER,
    started_at TEXT,
    ended_at TEXT,
    meta TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deployments (
    dep_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    meta TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


JOB_FIELDS = ("status", "pid", "log_path", "work_units", "estimate_s", "exit_code", "started_at", "ended_at")


class Registry:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = project_config()["project"]["registry_path"]
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)   # job workers write from other processes
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- generic helpers ----------
    def _rows(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            d = dict(r)
            for k in ("meta", "metrics"):
                if k in d and isinstance(d[k], str):
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    def close(self) -> None:
        self.conn.close()

    # ---------- demos ----------
    def upsert_demo(self, demo_id: str, source: str, episode_index: int, meta: dict[str, Any],
                    status: str | None = None) -> None:
        ts = now_iso()
        existing = self.get_demo(demo_id)
        if existing:
            merged = {**existing["meta"], **meta}
            self.conn.execute(
                "UPDATE demos SET meta=?, status=COALESCE(?, status), updated_at=? WHERE demo_id=?",
                (json.dumps(merged, default=str), status, ts, demo_id))
        else:
            self.conn.execute(
                "INSERT INTO demos VALUES (?,?,?,?,?,?,?)",
                (demo_id, source, episode_index, status or "imported", json.dumps(meta, default=str), ts, ts))
        self.conn.commit()

    def set_demo_status(self, demo_id: str, status: str, extra: dict[str, Any] | None = None) -> None:
        d = self.get_demo(demo_id)
        if d is None:
            raise KeyError(demo_id)
        meta = {**d["meta"], **(extra or {})}
        self.conn.execute("UPDATE demos SET status=?, meta=?, updated_at=? WHERE demo_id=?",
                          (status, json.dumps(meta, default=str), now_iso(), demo_id))
        self.conn.commit()

    def get_demo(self, demo_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM demos WHERE demo_id=?", (demo_id,))
        return rows[0] if rows else None

    def demos(self, status: str | None = None, source: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM demos WHERE 1=1", []
        if status:
            sql += " AND status=?"
            args.append(status)
        if source:
            sql += " AND source=?"
            args.append(source)
        return self._rows(sql + " ORDER BY source, episode_index", tuple(args))

    # ---------- datasets ----------
    def add_dataset(self, version: str, fingerprint: str, manifest_path: str, meta: dict[str, Any]) -> None:
        meta = {**meta, **({"card": portable(meta["card"])} if meta.get("card") else {})}
        self.conn.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?)",
                          (version, fingerprint, "draft", portable(manifest_path), json.dumps(meta, default=str),
                           now_iso()))
        self.conn.commit()

    def set_dataset_status(self, version: str, status: str) -> None:
        if not self.get_dataset(version):
            raise KeyError(version)
        self.conn.execute("UPDATE datasets SET status=? WHERE version=?", (status, version))
        self.conn.commit()

    def update_dataset_meta(self, version: str, extra: dict[str, Any]) -> None:
        rows = self._rows("SELECT * FROM datasets WHERE version=?", (version,))   # stored meta, not this machine's paths
        if not rows:
            raise KeyError(version)
        d = rows[0]
        self.conn.execute("UPDATE datasets SET meta=? WHERE version=?",
                          (json.dumps({**d["meta"], **extra}, default=str), version))
        self.conn.commit()

    def get_dataset(self, version: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM datasets WHERE version=?", (version,))
        return _dataset_paths(rows[0]) if rows else None

    def datasets(self) -> list[dict[str, Any]]:
        return [_dataset_paths(r) for r in self._rows("SELECT * FROM datasets ORDER BY created_at")]

    # ---------- models ----------
    def add_model(self, version: str, dataset_version: str, policy_name: str, status: str,
                  checkpoint_dir: str | None, run_record_path: str | None, meta: dict[str, Any]) -> None:
        meta = {**meta, **{k: portable(meta[k]) for k in MODEL_META_PATHS if meta.get(k)}}
        self.conn.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?)",
                          (version, dataset_version, policy_name, status,
                           portable(checkpoint_dir) if checkpoint_dir else checkpoint_dir,
                           portable(run_record_path) if run_record_path else run_record_path,
                           json.dumps(meta, default=str), now_iso()))
        self.conn.commit()

    def set_model_status(self, version: str, status: str) -> None:
        self.conn.execute("UPDATE models SET status=? WHERE version=?", (status, version))
        self.conn.commit()

    def update_model_meta(self, version: str, extra: dict[str, Any]) -> None:
        rows = self._rows("SELECT * FROM models WHERE version=?", (version,))
        if not rows:
            raise KeyError(version)
        m = rows[0]
        self.conn.execute("UPDATE models SET meta=? WHERE version=?",
                          (json.dumps({**m["meta"], **extra}, default=str), version))
        self.conn.commit()

    def get_model(self, version: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM models WHERE version=?", (version,))
        return _model_paths(rows[0]) if rows else None

    def models(self) -> list[dict[str, Any]]:
        return [_model_paths(r) for r in self._rows("SELECT * FROM models ORDER BY created_at")]

    # ---------- evaluations ----------
    def add_eval(self, eval_id: str, model_version: str, test_suite: str, env_version: str,
                 results_dir: str, metrics: dict[str, Any]) -> None:
        self.conn.execute("INSERT INTO evals VALUES (?,?,?,?,?,?,?)",
                          (eval_id, model_version, test_suite, env_version, results_dir,
                           json.dumps(metrics, default=str), now_iso()))
        self.conn.commit()

    def get_eval(self, eval_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM evals WHERE eval_id=?", (eval_id,))
        return rows[0] if rows else None

    def evals(self, model_version: str | None = None) -> list[dict[str, Any]]:
        if model_version:
            return self._rows("SELECT * FROM evals WHERE model_version=? ORDER BY created_at", (model_version,))
        return self._rows("SELECT * FROM evals ORDER BY created_at")

    # ---------- deployments ----------
    def add_deployment(self, dep_id: str, model_version: str, status: str, meta: dict[str, Any]) -> None:
        self.conn.execute("INSERT INTO deployments VALUES (?,?,?,?,?)",
                          (dep_id, model_version, status, json.dumps(meta, default=str), now_iso()))
        self.conn.commit()

    def deployments(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM deployments ORDER BY created_at")

    def current_deployed_model(self) -> str | None:
        rows = self._rows("SELECT * FROM deployments WHERE status='deployed' ORDER BY created_at DESC LIMIT 1")
        return rows[0]["model_version"] if rows else None

    # ---------- jobs ----------
    def add_job(self, job_id: str, kind: str, needs: str, machine: str, meta: dict[str, Any],
                work_units: float | None = None, estimate_s: float | None = None) -> None:
        self.conn.execute(
            "INSERT INTO jobs (job_id, kind, status, needs, machine, work_units, estimate_s, meta, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, kind, "queued", needs, machine, work_units, estimate_s, json.dumps(meta, default=str), now_iso()))
        self.conn.commit()

    def update_job(self, job_id: str, meta: dict[str, Any] | None = None, **fields: Any) -> None:
        bad = set(fields) - set(JOB_FIELDS)
        if bad:
            raise ValueError(f"unknown job fields {bad}")
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        sets = [f"{k}=?" for k in fields] + ["meta=?"]
        args = list(fields.values()) + [json.dumps({**job["meta"], **(meta or {})}, default=str), job_id]
        self.conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", args)
        self.conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        return rows[0] if rows else None

    def jobs(self, kind: str | None = None, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM jobs WHERE 1=1", []
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if status:
            sql += " AND status=?"
            args.append(status)
        return self._rows(sql + " ORDER BY created_at DESC, job_id DESC LIMIT ?", (*args, limit))
