"""Jobs, expected-time estimates, machine info and the registry writer lock."""
import sys

import pytest

from hackpipe.common import writelock
from hackpipe.jobs import runner
from hackpipe.jobs.estimates import NO_ESTIMATE, estimate, fmt_duration


@pytest.fixture
def me(monkeypatch):
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop-test")
    return "laptop-test"


def _done_job(reg, job_id, kind, work, seconds, machine="laptop-test"):
    reg.add_job(job_id, kind, "cpu", machine, {"label": kind}, work_units=work)
    reg.update_job(job_id, status="done", started_at="2026-10-23T10:00:00Z",
                   ended_at=f"2026-10-23T10:{seconds // 60:02d}:{seconds % 60:02d}Z", exit_code=0)


def test_no_estimate_without_history(workspace, me):
    _, reg = workspace
    assert estimate("qc", 1000, reg, me) == {"seconds": None, "basis": NO_ESTIMATE}
    assert estimate("qc", None, reg, me)["seconds"] is None


def test_estimate_from_measured_runs(workspace, me):
    _, reg = workspace
    _done_job(reg, "job-0001", "qc", 1000, 10)          # 100 units/s
    _done_job(reg, "job-0002", "qc", 3000, 20)          # 150 units/s
    _done_job(reg, "job-0003", "qc", 9999, 1, machine="colab")
    e = estimate("qc", 1250, reg, me)
    assert e["seconds"] == pytest.approx(1250 / 125) and "2 earlier run(s) on laptop-test" in e["basis"]
    other = estimate("qc", 1000, reg, "elsewhere")       # no run on this machine → all machines, labelled
    assert "other machine" in other["basis"]


def test_fmt_duration():
    assert fmt_duration(None) == "n/a" and fmt_duration(42) == "42s"
    assert fmt_duration(125) == "2m 05s" and fmt_duration(3725) == "1h 02m"


def test_job_runs_and_records(workspace, me):
    _, reg = workspace
    jid = runner.submit("echo", [sys.executable, "-c", "print('hello from job')"], work_units=5,
                        registry=reg, start=False)
    assert reg.get_job(jid)["status"] == "queued" and writelock.holder()["machine"] == me
    assert runner.run_job(jid, reg) == 0
    j = reg.get_job(jid)
    assert j["status"] == "done" and j["exit_code"] == 0 and j["started_at"] and j["ended_at"]
    assert "hello from job" in runner.log_tail(j)
    assert runner.submit("echo", ["true"], registry=reg, start=False) == "job-0002"


def test_failed_job(workspace, me):
    _, reg = workspace
    jid = runner.submit("boom", [sys.executable, "-c", "raise SystemExit(3)"], registry=reg, start=False)
    assert runner.run_job(jid, reg) == 3 and reg.get_job(jid)["status"] == "failed"


def test_reconcile_marks_lost_jobs(workspace, me):
    _, reg = workspace
    jid = runner.submit("lost", ["true"], registry=reg, start=False)
    reg.update_job(jid, status="running", pid=999999)   # no such process
    assert runner.reconcile(reg) == [jid] and reg.get_job(jid)["status"] == "failed"


def test_writer_lock(workspace, monkeypatch):
    monkeypatch.setenv("HACKPIPE_MACHINE", "colab")
    writelock.acquire()
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop")
    with pytest.raises(writelock.WriterLockError, match="'colab'"):
        writelock.acquire()
    assert writelock.release() is False                  # not the holder
    rec = writelock.acquire(force=True)
    assert rec["machine"] == "laptop" and rec["took_over_from"] == "colab"
    assert writelock.release() is True and writelock.holder() is None


def test_submit_refused_when_other_machine_writes(workspace, monkeypatch):
    _, reg = workspace
    monkeypatch.setenv("HACKPIPE_MACHINE", "colab")
    writelock.acquire()
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop")
    with pytest.raises(writelock.WriterLockError):
        runner.submit("x", ["true"], registry=reg, start=False)
    assert reg.jobs() == []


def test_machine_info_shape():
    from hackpipe.jobs.machine import info
    m = info()
    assert {"machine", "cpus", "ram_gib", "disk_free_gib", "gpus", "label"} <= set(m)
