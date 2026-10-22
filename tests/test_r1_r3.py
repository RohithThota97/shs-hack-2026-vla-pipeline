"""R1–R3 against the spec: metadata, video / final-state checks, held-out source, frozen test, card."""
import json
from pathlib import Path

import pytest
import yaml

from conftest import FPS, add_fr3_extras, make_dataset, write_test_configs
from hackpipe.common.config import REPO_ROOT, source_key
from hackpipe.common.ids import sha256_file
from hackpipe.dataset.versioning import LeakageError, approve, create_dataset_version
from hackpipe.ingest.importer import build_demo_records, ingest, watch
from hackpipe.quality.checks import INVALID, REVIEW, VALID, check_dataset
from hackpipe.quality.report import run_qc


def fr3_dataset(root: Path, n=6, frames=30, **kw) -> Path:
    return add_fr3_extras(make_dataset(root, n_episodes=n, frames=frames), n, frames, **kw)


def use_config(test_configs: Path, **overrides) -> None:
    write_test_configs(test_configs, **overrides)


# ---------- R1 ----------

def test_source_key_through_symlinked_repo_path(tmp_path):
    link = tmp_path / "repo_link"
    link.symlink_to(REPO_ROOT)
    assert source_key(link / "data" / "raw" / "x") == "data/raw/x"


def test_report_name_inside_and_outside_repo():
    from hackpipe.quality.report import report_name
    assert report_name("data/raw/fr3_duo_handover_84") == "fr3_duo_handover_84"
    a, b = report_name("/mnt/a/fr3_duo_handover_84"), report_name("/mnt/b/fr3_duo_handover_84")
    assert a != b and a.startswith("fr3_duo_handover_84-")


def test_source_key_is_repo_relative_inside_repo(tmp_path):
    assert source_key(REPO_ROOT / "data" / "raw" / "x") == "data/raw/x"
    assert source_key("data/raw/x") == "data/raw/x"
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert source_key(tmp_path / "link").endswith("/link")          # symlink not followed


def test_demo_record_has_spec_metadata(tmp_path):
    root = fr3_dataset(tmp_path / "ds")
    rec = build_demo_records(root)[2]
    m = rec["meta"]
    assert rec["source"] == source_key(root)
    assert m["recording"]["start_s"] == 0.0 and m["recording"]["end_s"] == pytest.approx(29 / FPS)
    assert set(m["robot"]) == {"type", "embodiment", "end_effector", "controller"}
    assert m["task"] == m["instruction"] == "pick red box"
    env = m["environment"]
    assert env["seed"] == 7002 and env["variation"] == "syn_s7002" and env["layout_id"] == "syn_root_v1"
    assert env["lighting"]["color_temperature_K"] == 5020.0
    assert m["outcome"]["status"] == "PASS" and m["outcome"]["footprint_margin_mm"] == 50.0
    assert m["cameras"]["observation.images.front"]["fps"] == FPS
    files = m["files_sha256"]
    assert files["data/chunk-000/file-000.parquet"] == sha256_file(root / "data/chunk-000/file-000.parquet")
    assert "videos/observation.images.front/chunk-000/file-000.mp4" in files
    assert m["pipeline"]["hackpipe"] and "git_commit" in m["pipeline"]


def test_watch_imports_and_refreshes_qc(workspace):
    tmp, reg = workspace
    root = fr3_dataset(tmp / "ds")
    watch(root, interval_s=0, registry=reg, max_polls=1)
    assert len(reg.demos()) == 6
    assert all(d["status"] == VALID for d in reg.demos())
    from hackpipe.quality.report import report_name
    assert (tmp / "outputs" / "qc" / report_name(source_key(root)) / "qc_report.md").exists()


# ---------- R2 ----------

def test_videos_ok_all_valid(tmp_path):
    res = check_dataset(fr3_dataset(tmp_path / "ds"))
    assert [e.status for e in res["episodes"]] == [VALID] * 6
    assert res["episodes"][0].stats["video_frames:front"] == 30


def test_missing_video_frames_is_review(tmp_path):
    res = check_dataset(fr3_dataset(tmp_path / "ds", video="short:3"))
    by = {e.episode_index: e for e in res["episodes"]}
    assert by[3].status == REVIEW and any(s.startswith("missing_frames:front (27") for s in by[3].soft)
    assert by[2].status == VALID


def test_missing_video_found_without_decoding(tmp_path, test_configs):
    use_config(test_configs, quality__check_videos=False)
    res = check_dataset(fr3_dataset(tmp_path / "ds", video="missing"))
    assert all(e.status == INVALID and "missing_video:front" in e.hard for e in res["episodes"])


@pytest.mark.parametrize("video,reason", [("corrupt", "video_corrupted:front"), ("missing", "missing_video:front")])
def test_broken_video_is_invalid(tmp_path, video, reason):
    res = check_dataset(fr3_dataset(tmp_path / "ds", video=video))
    assert all(e.status == INVALID and reason in e.hard for e in res["episodes"])


def test_final_state_from_outcome_records(tmp_path):
    root = fr3_dataset(tmp_path / "ds", video=None,
                       outcome={1: {"status": "FAIL"}, 4: {"footprint_margin_mm": -3.0}})
    by = {e.episode_index: e for e in check_dataset(root)["episodes"]}
    assert by[1].status == REVIEW and by[1].soft[0].startswith("final_state_not_reached")
    assert by[4].status == REVIEW and by[4].soft[0].startswith("object_outside_target_zone")
    assert by[0].status == VALID


def test_robot_config_dimension_mismatch(tmp_path, test_configs):
    proj = yaml.safe_load((test_configs / "project.yaml").read_text())
    proj["robot"]["state_dim"] = 16
    (test_configs / "project.yaml").write_text(yaml.safe_dump(proj))
    res = check_dataset(make_dataset(tmp_path / "ds", n_episodes=2))
    assert any("robot config 16" in i for i in res["dataset_issues"])
    assert all(e.status == INVALID for e in res["episodes"])


def test_joint_limits(tmp_path, test_configs):
    use_config(test_configs, quality__joint_limits={"action": {"min": [-1] * 4, "max": [1] * 4}})
    by = {e.episode_index: e for e in check_dataset(make_dataset(tmp_path / "ds", n_episodes=2, frames=30))["episodes"]}
    assert "action_outside_joint_limits" in by[0].hard      # synthetic action ramps to 2.9


def test_redundant_trajectories_reported(workspace):
    tmp, reg = workspace
    root = fr3_dataset(tmp / "ds", n=4, video=None)          # synthetic episodes share one trajectory
    ingest(root, reg)
    r = run_qc(root, reg)
    assert r["redundancy"]["distinct_trajectories"] == 1 and r["redundancy"]["largest_identical_group"] == 4
    assert "1 distinct state/action trajectories in 4 episodes" in Path(r["paths"]["md"]).read_text()


def test_qc_warns_when_not_ingested(workspace):
    tmp, reg = workspace
    r = run_qc(fr3_dataset(tmp / "ds", n=2, video=None), reg)
    assert any("hackpipe ingest" in i for i in r["dataset_issues"])


# ---------- R3 ----------

def _two_sources(tmp, reg, test_configs, test_seeds=None):
    train_root = fr3_dataset(tmp / "night", n=10, video=None)
    test_root = fr3_dataset(tmp / "lighting", n=3, video=None, seeds=test_seeds or [9000, 9001, 9002])
    use_config(test_configs, splits__test_sources=[str(test_root)], splits__leakage_keys=["environment.seed"],
               splits__train=0.9, splits__val=0.1, splits__test=0.0)
    for r in (train_root, test_root):
        ingest(r, reg)
        run_qc(r, reg)
    return train_root, test_root


def test_held_out_source_is_the_test_split(workspace, test_configs):
    tmp, reg = workspace
    train_root, test_root = _two_sources(tmp, reg, test_configs)
    m = create_dataset_version(reg)
    assert m["source"] == source_key(train_root) and m["test_sources"] == [source_key(test_root)]
    assert m["splits"]["test"] == [0, 1, 2] and len(m["splits"]["train"]) + len(m["splits"]["val"]) == 10
    assert len(m["splits"]["val"]) == 1
    test_ids = set(m["demo_ids"]["test"])
    assert test_ids == {d["demo_id"] for d in reg.demos(source=source_key(test_root))}
    reports = sorted(p.name for p in (tmp / "outputs" / "qc").iterdir())
    assert len(reports) == 2 and reports[0].startswith("lighting-") and reports[1].startswith("night-")


def test_leakage_guard_on_seed(workspace, test_configs):
    tmp, reg = workspace
    _two_sources(tmp, reg, test_configs, test_seeds=[9000, 7003, 9002])    # 7003 is a train seed
    with pytest.raises(LeakageError, match="environment.seed"):
        create_dataset_version(reg)


def test_test_split_is_frozen(workspace, test_configs, dataset):
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    use_config(test_configs, splits__seed=4242)          # different seed → different ratio-based test split
    create_dataset_version(reg)                          # ds-v1 is only a draft: nothing frozen yet
    approve("ds-v1", reg)
    with pytest.raises(LeakageError, match="frozen since ds-v1"):
        create_dataset_version(reg)


def test_manifest_preprocessing_and_card(workspace, test_configs):
    tmp, reg = workspace
    _two_sources(tmp, reg, test_configs)
    m = create_dataset_version(reg, notes="v1 test")
    pre = m["preprocessing"]
    assert pre["schema"]["fps"] == FPS and pre["schema"]["state"]["shape"] == [4]
    assert pre["normalisation_stats"]["sha256"]
    out = tmp / "outputs" / "datasets" / "ds-v1"
    assert json.loads((out / "stats.json").read_text()) == {"action": {"mean": [0.0] * 4}}
    card = (out / "dataset_card.md").read_text()
    assert m["fingerprint"] in card and "Held-out test sources" in card and "lighting" in card
    assert "**test**" in card and "[0, 1, 2]" in card
    assert "All 13 demos are successful" in card
    assert "13 demos share only 1 distinct state/action trajectories" in card
    assert "vary: lighting, seed, variation" in card
    assert "environment.seed — passed" in card


def test_fingerprint_changes_with_video_files(workspace, test_configs):
    tmp, reg = workspace
    train_root = fr3_dataset(tmp / "night", n=10)
    use_config(test_configs, splits__test=0.0, splits__val=0.1)
    ingest(train_root, reg)
    run_qc(train_root, reg)
    fp1 = create_dataset_version(reg)["fingerprint"]
    video = train_root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    video.write_bytes(video.read_bytes() + b"\0")             # images changed, state/action identical
    ingest(train_root, reg)                                    # changed demos go back to "imported" …
    assert {d["status"] for d in reg.demos()} == {"imported"}
    run_qc(train_root, reg)                                    # … and must pass QC again
    assert create_dataset_version(reg)["fingerprint"] != fp1


def test_fingerprint_changes_with_stats(workspace, test_configs):
    tmp, reg = workspace
    train_root, _ = _two_sources(tmp, reg, test_configs)
    fp1 = create_dataset_version(reg)["fingerprint"]
    (train_root / "meta" / "stats.json").write_text(json.dumps({"action": {"mean": [1.0] * 4}}))
    assert create_dataset_version(reg)["fingerprint"] != fp1


# ---------- R4 inputs ----------

def test_train_set_overrides_and_rename_map(workspace, test_configs, dataset):
    from hackpipe.training.train import parse_overrides, train
    tmp, reg = workspace
    use_config(test_configs, preprocessing__rename_map={"observation.images.front": "observation.images.camera1"})
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg)
    ov = parse_overrides(["steps=200", "save_freq=200", "seed=7", "policy.device=cpu"])
    assert ov == {"steps": 200, "save_freq": 200, "seed": 7, "policy.device": "cpu"}
    rec = train("configs/train/smolvla_fr3_duo.yaml", "ds-v1", dry_run=True, registry=reg, overrides=ov)
    cmd = rec["command"]
    assert "--steps=200" in cmd and "--save_freq=200" in cmd and "--seed=7" in cmd
    assert "--policy.path=lerobot/smolvla_base" in cmd and "--policy.device=cpu" in cmd
    assert '--rename_map={"observation.images.front":"observation.images.camera1"}' in cmd
    assert rec["overrides"] == ov
    with pytest.raises(ValueError):
        parse_overrides(["steps"])


def test_model_registered_before_training_finishes(workspace, dataset, monkeypatch):
    monkeypatch.setattr("hackpipe.training.train.preflight", lambda: None)
    import subprocess
    from hackpipe.training.train import train
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg)
    seen = {}
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] != "lerobot-train":
            return real_run(cmd, **kw)                             # git / pip calls from envinfo
        seen["status"] = reg.get_model("m-v1")["status"]          # what a crashed session would leave behind
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rec = train("configs/train/smolvla_fr3_duo.yaml", "ds-v1", registry=reg)
    assert seen["status"] == "training"
    assert reg.get_model("m-v1")["status"] == "failed" and rec["exit_code"] == 1


def test_loss_log_parsing(tmp_path):
    from hackpipe.training.loss import export, parse_log
    log = ("INFO 2026-10-23 step:200 smpl:2K ep:1 epch:0.01 loss:0.512 grdn:1.2 lr:1e-04\n"
           "noise line\nINFO step:1K smpl:8K loss:0.21 grdn:0.9\nINFO step:2.5K loss:1.5e-01\n")
    assert parse_log(log) == [(200, 0.512), (1000, 0.21), (2500, 0.15)]
    (tmp_path / "train.log").write_text(log)
    r = export(tmp_path / "train.log", tmp_path / "run")
    assert r["points"] == 3 and r["final_loss"] == 0.15
    assert (tmp_path / "run" / "loss.csv").read_text().splitlines()[-1] == "2500,0.15"


def _approved(workspace, dataset):
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg)
    return reg


def test_resume_rejects_overrides(workspace, dataset):
    from hackpipe.training.train import train
    reg = _approved(workspace, dataset)
    with pytest.raises(ValueError, match="--resume"):
        train("configs/train/smolvla_fr3_duo.yaml", "ds-v1", registry=reg, resume_model="m-v1",
              overrides={"steps": 40000})


def test_launch_failure_marks_model_failed(workspace, dataset, monkeypatch):
    monkeypatch.setattr("hackpipe.training.train.preflight", lambda: None)
    import subprocess
    from hackpipe.training.train import train
    reg = _approved(workspace, dataset)
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] != "lerobot-train":
            return real_run(cmd, **kw)
        raise FileNotFoundError("lerobot-train")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FileNotFoundError):
        train("configs/train/smolvla_fr3_duo.yaml", "ds-v1", registry=reg)
    assert reg.get_model("m-v1")["status"] == "failed"
    rec = json.loads(Path(reg.get_model("m-v1")["run_record_path"]).read_text())
    assert rec["error"].startswith("FileNotFoundError")


def test_preflight_blocks_broken_lerobot_before_registering(workspace, dataset, monkeypatch):
    import subprocess
    from hackpipe.training import train as t
    reg = _approved(workspace, dataset)
    monkeypatch.setattr(t.shutil, "which", lambda name: "/usr/bin/lerobot-train")
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[1:] == ["--help"]:
            return subprocess.CompletedProcess(cmd, 1, "", "ImportError: cannot import name 'FqnToConfig' "
                                                           "from 'torchao.quantization'")
        return real_run(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(t.PreflightError, match="pip uninstall -y torchao"):
        t.train("configs/train/smolvla_fr3_duo.yaml", "ds-v1", registry=reg)
    assert reg.models() == []                                   # no model ID used up


def test_preflight_needs_lerobot(monkeypatch):
    from hackpipe.training import train as t
    monkeypatch.setattr(t.shutil, "which", lambda name: None)
    with pytest.raises(t.PreflightError, match="not found"):
        t.preflight()


def test_log_steps_rebuilt_from_log_freq(tmp_path):
    from hackpipe.training.loss import export, parse_metrics
    lines = [f"INFO step:{s if s < 1000 else str(round(s / 1000)) + 'K'} smpl:1 ep:0 epch:0.01 loss:{1 / s:.4f} "
             f"grdn:1.0 lr:1.0e-04 updt_s:0.420 data_s:0.031 smp/s:19 mem_gb:9.87" for s in range(500, 3001, 500)]
    rows = parse_metrics("\n".join(lines), log_freq=500)
    assert [r["step"] for r in rows] == [500, 1000, 1500, 2000, 2500, 3000]   # printed: 500,1K,2K,2K,2K,3K
    assert rows[0]["updt_s"] == 0.42 and rows[0]["data_s"] == 0.031 and rows[0]["mem_gb"] == 9.87
    assert rows[0]["smp/s"] == 19 and rows[0]["lr"] == 1e-4
    assert [r["step"] for r in parse_metrics("\n".join(lines))][2] == 2000       # no log_freq: as printed
    (tmp_path / "t.log").write_text("\n".join(lines))
    export(tmp_path / "t.log", tmp_path / "run", log_freq=500)
    assert (tmp_path / "run" / "loss.csv").read_text().splitlines()[3].startswith("1500,")
    assert (tmp_path / "run" / "metrics.csv").read_text().splitlines()[0].startswith("step,loss,grdn,lr,updt_s")
