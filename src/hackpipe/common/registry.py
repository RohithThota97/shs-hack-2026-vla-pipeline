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
CREATE TABLE IF NOT EXISTS deployments (
    dep_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    meta TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Registry:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = project_config()["project"]["registry_path"]
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
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
        self.conn.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?)",
                          (version, fingerprint, "draft", manifest_path, json.dumps(meta, default=str), now_iso()))
        self.conn.commit()

    def set_dataset_status(self, version: str, status: str) -> None:
        if not self.get_dataset(version):
            raise KeyError(version)
        self.conn.execute("UPDATE datasets SET status=? WHERE version=?", (status, version))
        self.conn.commit()

    def get_dataset(self, version: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM datasets WHERE version=?", (version,))
        return rows[0] if rows else None

    def datasets(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM datasets ORDER BY created_at")

    # ---------- models ----------
    def add_model(self, version: str, dataset_version: str, policy_name: str, status: str,
                  checkpoint_dir: str | None, run_record_path: str | None, meta: dict[str, Any]) -> None:
        self.conn.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?)",
                          (version, dataset_version, policy_name, status, checkpoint_dir, run_record_path,
                           json.dumps(meta, default=str), now_iso()))
        self.conn.commit()

    def set_model_status(self, version: str, status: str) -> None:
        self.conn.execute("UPDATE models SET status=? WHERE version=?", (status, version))
        self.conn.commit()

    def get_model(self, version: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM models WHERE version=?", (version,))
        return rows[0] if rows else None

    def models(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM models ORDER BY created_at")

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
