"""Submit pipeline steps as background jobs that survive page reloads.

submit() records the job (with its expected time) and starts `python -m hackpipe.jobs.worker <job_id>` in its own
session. The worker waits for the GPU lock if the job needs the GPU, runs the command with output to a log file,
and records the exit code. reconcile() marks jobs whose process vanished (e.g. a Colab restart) as failed.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.ids import now_iso
from hackpipe.common.registry import Registry
from hackpipe.common.writelock import acquire, machine_id
from hackpipe.jobs.estimates import estimate

GPU_LOCK = "/tmp/hackpipe-gpu.lock"


def hackpipe_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "hackpipe", *args]


def _next_job_id(reg: Registry) -> str:
    nums = [int(j["job_id"].split("-")[1]) for j in reg.jobs(limit=100000) if j["job_id"].startswith("job-")]
    return f"job-{max(nums, default=0) + 1:04d}"


def log_dir() -> Path:
    d = resolve(Path(project_config()["project"]["outputs_dir"]) / "logs" / "jobs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def submit(kind: str, argv: list[str], needs: str = "cpu", work_units: float | None = None, label: str = "",
           registry: Registry | None = None, start: bool = True, estimate_override: dict | None = None,
           extra_meta: dict | None = None, env: dict[str, str] | None = None) -> str:
    """Record and start a job. Claims the registry writer role for this machine first.
    estimate_override: {"seconds", "basis"} measured elsewhere (e.g. training: timing run × steps).
    env: extra environment for the job only (e.g. access tokens) — not recorded anywhere."""
    if needs not in ("cpu", "gpu"):
        raise ValueError("needs must be 'cpu' or 'gpu'")
    reg = registry or Registry()
    acquire()
    me = machine_id()
    job_id = _next_job_id(reg)
    est = estimate_override or estimate(kind, work_units, reg, me)
    reg.add_job(job_id, kind, needs, me, {"argv": argv, "label": label or kind, "estimate_basis": est["basis"],
                                          **(extra_meta or {})},
                work_units=work_units, estimate_s=est["seconds"])
    reg.update_job(job_id, log_path=str(log_dir() / f"{job_id}.log"))
    if start:
        subprocess.Popen([sys.executable, "-m", "hackpipe.jobs.worker", job_id], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, **(env or {})})
    return job_id


def run_job(job_id: str, registry: Registry | None = None) -> int:
    """Worker body: wait for the GPU if needed, run the command, record the result."""
    reg = registry or Registry()
    job = reg.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    log = Path(job["log_path"])
    gpu_lock = None
    if job["needs"] == "gpu":
        import fcntl
        gpu_lock = open(GPU_LOCK, "w")
        reg.update_job(job_id, pid=os.getpid(), meta={"waiting_for": "GPU"})
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)              # blocks while another GPU job runs
    reg.update_job(job_id, status="running", pid=os.getpid(), started_at=now_iso(), meta={"waiting_for": None})
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        with open(log, "a") as f:
            f.write(f"$ {' '.join(job['meta']['argv'])}\n")
            f.flush()
            code = subprocess.run(job["meta"]["argv"], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    except BaseException as e:  # noqa: BLE001 — record any failure, then re-raise
        reg.update_job(job_id, status="failed", ended_at=now_iso(), meta={"error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        if gpu_lock:
            gpu_lock.close()
    reg.update_job(job_id, status="done" if code == 0 else "failed", exit_code=code, ended_at=now_iso())
    return code


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reconcile(registry: Registry | None = None) -> list[str]:
    """Jobs of this machine that say queued/running but whose worker is gone → failed (lost)."""
    reg = registry or Registry()
    lost = []
    for status in ("running", "queued"):
        for j in reg.jobs(status=status):
            if j["machine"] == machine_id() and j["pid"] and not _alive(j["pid"]):
                reg.update_job(j["job_id"], status="failed", ended_at=now_iso(),
                               meta={"error": "worker process ended without reporting (machine restarted?)"})
                lost.append(j["job_id"])
    return lost


def cancel(job_id: str, registry: Registry | None = None) -> bool:
    reg = registry or Registry()
    j = reg.get_job(job_id)
    if not j or j["status"] not in ("queued", "running") or not _alive(j["pid"]):
        return False
    os.killpg(j["pid"], signal.SIGTERM)                   # worker leads its own session: stops the child too
    reg.update_job(job_id, status="failed", ended_at=now_iso(), meta={"error": "cancelled by user"})
    return True


def log_tail(job: dict[str, Any], lines: int = 15) -> str:
    p = Path(job.get("log_path") or "")
    if not p.exists():
        return ""
    return "\n".join(p.read_text(errors="replace").splitlines()[-lines:])
