"""`hackpipe auto --root <folder>`: new demonstrations in, draft dataset version out — no manual step until
the human approval.

import (incremental, duplicates recognised) → QC of new/changed demos only → next draft version based on the
latest approved version (inherits its validated demos). It doesn't create a version when nothing changed, or
when a draft is already waiting for approval and this folder brought nothing new.
"""
from __future__ import annotations

from typing import Any

from hackpipe.common.config import source_key
from hackpipe.common.registry import Registry
from hackpipe.dataset.versioning import NothingNewError, create_dataset_version, latest_approved
from hackpipe.ingest.importer import import_demos
from hackpipe.quality.report import run_qc


def _waiting_draft(reg: Registry) -> str | None:
    base = latest_approved(reg)
    base_n = int(base.split("-v", 1)[1]) if base else 0
    drafts = [d["version"] for d in reg.datasets()
              if d["status"] == "draft" and int(d["version"].split("-v", 1)[1]) > base_n]
    return max(drafts, key=lambda v: int(v.split("-v", 1)[1])) if drafts else None


def run_auto(root: str, registry: Registry | None = None, notes: str = "") -> dict[str, Any]:
    reg = registry or Registry()
    imp = import_demos(root, reg)
    source = source_key(root)
    pending = [d for d in reg.demos(source=source) if d["status"] == "imported"]
    out: dict[str, Any] = {"source": source,
                           "import": {k: (len(v) if isinstance(v, list) else v) for k, v in imp.items()
                                      if k != "source"},
                           "qc": None, "version": None, "base": latest_approved(reg)}
    if pending:
        r = run_qc(root, reg)
        out["qc"] = {"checked_now": r["checked_now"], "reused": r["reused"], "status_counts": r["status_counts"],
                     "report": r["paths"]["md"]}
    changed = bool(imp["new"] or imp["changed"] or pending)
    waiting = _waiting_draft(reg)
    if waiting and not changed:
        out["reason"] = f"{waiting} is waiting for approval and this folder brought nothing new."
        return out
    try:
        m = create_dataset_version(reg, notes=notes or f"auto: {source}", base=out["base"])
    except NothingNewError as e:
        out["reason"] = str(e)
        return out
    except RuntimeError as e:
        out["reason"] = f"no version created: {e}"
        return out
    lin = m.get("lineage") or {}
    out.update({"version": m["version"], "fingerprint": m["fingerprint"],
                "counts": {k: len(v) for k, v in m["demo_ids"].items()},
                "inherited": len(lin.get("inherited", [])), "added": len(lin.get("added", [])),
                "removed": len(lin.get("removed", [])), "needs_materialize": m["needs_materialize"],
                "reason": f"{m['version']} is waiting for approval."})
    return out
