"""Train page numbers (measured from logs) and the page itself."""
import json
from pathlib import Path

import pytest

from hackpipe.training import progress

LOG = """INFO 2026-10-23 10:00:40 ot_train.py:298 Creating dataset
INFO 2026-10-23 10:01:30 ot_train.py:400 step:50 smpl:400 ep:0 epch:0.00 loss:0.910 grdn:3.1 lr:1.0e-04 updt_s:0.400 data_s:0.100 smp/s:16 mem_gb:9.10
INFO 2026-10-23 10:01:55 ot_train.py:400 step:100 smpl:800 ep:0 epch:0.00 loss:0.700 grdn:2.9 lr:1.0e-04 updt_s:0.400 data_s:0.100 smp/s:16 mem_gb:9.40
INFO 2026-10-23 10:02:20 ot_train.py:400 step:150 smpl:1K ep:0 epch:0.01 loss:0.550 grdn:2.5 lr:1.0e-04 updt_s:0.400 data_s:0.100 smp/s:16 mem_gb:9.40
INFO 2026-10-23 10:02:45 ot_train.py:400 step:200 smpl:2K ep:0 epch:0.01 loss:0.480 grdn:2.2 lr:1.0e-04 updt_s:0.400 data_s:0.100 smp/s:16 mem_gb:9.50
"""


def _job(tmp_path, log=LOG, started="2026-10-23T10:00:00Z", status="done", **meta):
    p = tmp_path / "job.log"
    p.write_text(log)
    return {"job_id": "job-0007", "status": status, "log_path": str(p), "started_at": started,
            "meta": {"log_freq": 50, **meta}}


def test_timing_summary_from_log(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    import time
    time.tzset()
    t = progress.timing_summary(_job(tmp_path), 50)
    assert t["seconds_per_step"] == 0.5                     # 50 s between steps 100 and 200, warm-up line skipped
    assert t["peak_mem_gb"] == 9.5 and t["data_share"] == 0.2 and t["last_step"] == 200
    assert t["startup_s"] == pytest.approx(90 - 50 * 0.5)   # first log at +90 s, minus 50 steps
    assert progress.timing_summary(_job(tmp_path, log="no training lines"), 50)["seconds_per_step"] is None


def test_budget_epochs_expected():
    assert progress.budget(3.5, 0.5, 1000, startup_s=65) == 25000      # (12600-65)/0.5 = 25070 → 25000
    assert progress.budget(0.01, 0.5, 1000) == 0
    assert progress.epochs(25000, 8, 170772) == pytest.approx(1.17, abs=0.01)
    assert progress.expected_seconds(1000, 0.5, 60) == 560


def test_live_eta(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    import time
    time.tzset()
    lv = progress.live(_job(tmp_path, status="running"), target_steps=1200, log_freq=50)
    assert lv["step"] == 200 and lv["loss"] == 0.48 and lv["eta_s"] == pytest.approx(1000 * 0.5)
    assert progress.live(_job(tmp_path, log=""), 1200)["step"] is None


def test_run_record_diff_flags_dataset_fields():
    a = {"dataset_version": "ds-v1", "dataset_fingerprint": "aa", "train_config": {"seed": 1, "args": {"steps": 9}},
         "started_at": "x", "environment": {"gpu": "T4"}}
    b = {**a, "dataset_version": "ds-v2", "dataset_fingerprint": "bb", "started_at": "y",
         "environment": {"gpu": "A100"}}
    d = progress.run_record_diff(a, b)
    assert {x["field"] for x in d} == {"dataset_version", "dataset_fingerprint"} and all(x["dataset_field"] for x in d)
    c = {**b, "train_config": {"seed": 2, "args": {"steps": 9}}}
    assert any(x["field"] == "train_config.seed" and not x["dataset_field"] for x in progress.run_record_diff(a, c))


def test_set_config_steps(tmp_path):
    from hackpipe.training.train import set_config_steps
    p = tmp_path / "c.yaml"
    p.write_text("args:\n  batch_size: 8\n  steps: 20000                # placeholder — set from timing\n")
    set_config_steps(p, 25000)
    assert "  steps: 25000                # placeholder" in p.read_text()
    (tmp_path / "bad.yaml").write_text("args: {}\n")
    with pytest.raises(ValueError):
        set_config_steps(tmp_path / "bad.yaml", 1)


def test_train_page_renders_with_timing_run(workspace, dataset, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from hackpipe.dataset.versioning import approve, create_dataset_version
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop-test")
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg, by="t")
    log = tmp / "timing.log"
    log.write_text(LOG)
    reg.add_job("job-0001", "train_timing", "gpu", "colab", {"label": "Timing", "log_freq": 50}, work_units=200)
    reg.update_job("job-0001", status="done", log_path=str(log), started_at="2026-10-23T10:00:00Z",
                   ended_at="2026-10-23T10:03:00Z", exit_code=0)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/train.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert [t.label for t in at.tabs] == ["Setup & dry run", "Timing run", "Step budget", "Full run", "Runs"]
    labels = {m.label: m.value for m in at.metric}
    assert labels["Seconds / step"] == "0.5" and labels["Peak GPU memory"] == "9.5 GB"
    start = next(b for b in at.button if b.label == "Start timing run (200 steps)")
    assert start.disabled                                         # no GPU on the test machine


def test_purpose_recorded(workspace, dataset, monkeypatch):
    import subprocess
    from hackpipe.dataset.versioning import approve, create_dataset_version
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    from hackpipe.training.train import train
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg)
    monkeypatch.setattr("hackpipe.training.train.preflight", lambda: None)
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: real(cmd, **kw) if cmd[0] != "lerobot-train"
                        else subprocess.CompletedProcess(cmd, 0))
    rec = train("configs/train/smolvla_fr3_duo.yaml", "ds-v1", registry=reg, purpose="timing",
                overrides={"steps": 200})
    assert rec["purpose"] == "timing" and reg.get_model("m-v1")["meta"]["purpose"] == "timing"
    assert json.loads(Path(reg.get_model("m-v1")["run_record_path"]).read_text())["purpose"] == "timing"
