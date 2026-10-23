"""Isaac Sim side of the pipeline, without Isaac Sim: the reference file (held-out lighting), recordings → LeRobot
v3, derived demos keeping their original's split, importing eval_isaac results, curriculum source filter."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import add_fr3_extras, make_dataset, write_test_configs

CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera",
        "observation.images.right_wrist_camera"]


def _organiser_meta(root: Path, seeds: list[int]) -> Path:
    """The extra fields the sim reference reads (hold cut, box pose, gripper drive, conversion file)."""
    extras = [json.loads(l) for l in (root / "meta" / "fr3_episodes_extra.jsonl").read_text().splitlines()]
    for e, seed in zip(extras, seeds):
        e.update({"seed": seed, "hold_cut": {"dropped_frames": [12, 13]}, "final_box_pose": [0.1] * 7,
                  "gripper_drive": {"stiffness": 0.13, "damping": 0.007, "maxForce": 1.0}})
    (root / "meta" / "fr3_episodes_extra.jsonl").write_text("".join(json.dumps(e) + "\n" for e in extras))
    (root / "meta" / "fr3_conversion.json").write_text(json.dumps({
        "dataset_cfg": {"task": "pick red box", "gripper_contract": {"closed_joint_rad": 0.8, "hold_joint_rad": 0.459,
                                                                     "hold_norm": 0.51}},
        "recorder": {"camera_params": {"head": {"focal_length": 1.0}}}}))
    return root


@pytest.fixture
def sim_setup(workspace, tmp_path, monkeypatch):
    """main source (10 ep) + held-out test source (4 ep, other seeds), ds-v1 approved."""
    from hackpipe.dataset.versioning import approve
    from hackpipe.pipeline.auto import run_auto
    tmp, reg = workspace
    main = _organiser_meta(add_fr3_extras(make_dataset(tmp / "main", n_episodes=10, frames=30), 10, 30, video=None),
                           [100 + i for i in range(10)])
    test = _organiser_meta(add_fr3_extras(make_dataset(tmp / "test", n_episodes=4, frames=30), 4, 30, video=None,
                                          seeds=[900, 901, 902, 903]), [900, 901, 902, 903])
    cfg = write_test_configs(tmp_path / "cfg", splits__test_sources=[str(test)], splits__leakage_keys=["environment.seed"])
    proj = yaml.safe_load((cfg / "project.yaml").read_text())
    proj["dataset"]["root"] = str(main)
    (cfg / "project.yaml").write_text(yaml.safe_dump(proj))
    monkeypatch.setenv("HACKPIPE_CONFIG_DIR", str(cfg))
    from hackpipe.pipeline.auto import _import_and_check
    _import_and_check(str(test), reg)                 # held-out source first, then the version with both
    out = run_auto(str(main), reg)
    approve(out["version"], reg, by="t")
    suite = tmp / "suite.yaml"
    suite.write_text(yaml.safe_dump({"version": "sim-t1", "env_version": "isaac-t", "sim": {
        "lighting_source": str(test), "n_action_steps": 5}, "scenarios": [{"id": "s", "category": "environment_variation",
                                                                            "seen": False, "n_runs": 4}]}))
    return tmp, reg, main, test, suite, out["version"]


def test_reference_uses_held_out_lighting(sim_setup):
    from hackpipe.common.config import source_key
    from hackpipe.common.config import load_yaml
    from hackpipe.sim.reference import build_reference
    tmp, reg, main, test, suite, ver = sim_setup
    ref = build_reference(ver, load_yaml(suite), reg)
    assert [x["episode_index"] for x in ref["val_lighting"]] == [0, 1, 2, 3]        # the test source's episodes
    assert ref["hackpipe"]["lighting_source"] == source_key(test)
    assert len(ref["record_episodes"]) == 10 and {e["split"] for e in ref["record_episodes"]} <= {"train", "val"}
    assert ref["hold_cut_frame"] == 12 and len(ref["demo_actions"]) == 30 and ref["task"] == "pick red box"
    bad = {**load_yaml(suite), "sim": {"lighting_source": str(main)}}
    with pytest.raises(ValueError, match="not a held-out test source"):
        build_reference(ver, bad, reg)


def _results(lights, n_action_steps=5, stages=(4, 1, 0, 4)):
    return {"policy": {"n_action_steps": n_action_steps, "type": "smolvla"},
            "episodes": [{"episode": i, "lighting_from_val_episode": l, "success": s == 4, "stage": s,
                          "stage_name": "x", "dist_to_demo_final_mm": 12.0, "max_lift_mm": 50.0 if s else 0.0,
                          "latency_ms": {"median": 40.0, "p95": 80.0}} for i, (l, s) in enumerate(zip(lights, stages))]}


def test_import_eval_isaac_results(sim_setup):
    from hackpipe.evaluation.sim_import import ImportRefused, import_results
    from hackpipe.training.train import register_external
    tmp, reg, main, test, suite, ver = sim_setup
    m = register_external(str(tmp / "ckpt"), ver, "smolvla_team", reg)
    p = tmp / "results.json"
    p.write_text(json.dumps(_results([0, 1, 2, 3])))
    gifs = tmp / "gifs"
    gifs.mkdir()
    (gifs / "policy_ep00.gif").write_bytes(b"GIF89a")
    r = import_results(m, str(suite), p, gifs, reg)
    met = r["metrics"]
    assert met["task_success"]["successes"] == 2 and met["task_success"]["n"] == 4 and met["stage_counts_0to4"] == [1, 1, 0, 0, 2]
    assert met["fake_results"] is False and met["task_success"]["ci95"][0] < 0.5 < met["task_success"]["ci95"][1]
    assert reg.get_model(m)["status"] == "evaluated" and (Path(r["results_dir"]) / "policy_ep00.gif").exists()
    eps = [json.loads(l) for l in (Path(r["results_dir"]) / "episodes.jsonl").read_text().splitlines()]
    assert [e["failure_stage"] for e in eps] == [None, "transfer", "grasp", None] and eps[0]["video_path"]

    p.write_text(json.dumps(_results([0, 1, 7, 3])))                         # lit like a non-test episode
    with pytest.raises(ImportRefused, match="not .* episodes"):
        import_results(m, str(suite), p, None, reg)
    p.write_text(json.dumps(_results([0, 1, 2, 3], n_action_steps=20)))      # other re-planning setting
    with pytest.raises(ImportRefused, match="executed 20 actions"):
        import_results(m, str(suite), p, None, reg)


def _template(root: Path, fps: int = 10) -> Path:
    """A source-like folder with 3 cameras at 224 x 224 (what the converter copies its schema from)."""
    (root / "meta").mkdir(parents=True)
    feats = {c: {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"],
                 "info": {"video.fps": fps, "video.codec": "av1"}} for c in CAMS}
    feats |= {"observation.state": {"dtype": "float32", "shape": [16]}, "action": {"dtype": "float32", "shape": [16]}}
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": fps, "features": feats, "robot_type": "fr3_duo",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["hand over"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    return root


def _recording(root: Path, name: str, n: int, success: bool, source: str, episode: int):
    from PIL import Image
    d = root / name
    for c in CAMS:
        (d / "cams" / c.split(".")[-1]).mkdir(parents=True)
        for k in range(n):
            Image.fromarray(np.full((224, 224, 3), 10 * k, np.uint8)).save(d / "cams" / c.split(".")[-1] / f"{k:06d}.jpg")
    np.save(d / "state.npy", np.full((n, 16), 0.1, np.float32))
    np.save(d / "action.npy", np.full((n, 16), 0.2, np.float32))
    (d / "meta.json").write_text(json.dumps({"name": f"s{episode}", "mode": "render", "source": source,
                                             "source_episode": episode, "seed": 100 + episode,
                                             "light": {"rect": {}}, "success": success, "stage": 4 if success else 0}))


def test_convert_recordings_to_lerobot_v3(tmp_path):
    import av

    from hackpipe.common.lerobot_io import DatasetFolder
    from hackpipe.ingest.importer import build_demo_records
    from hackpipe.ingest.sim_recording import convert
    tpl = _template(tmp_path / "tpl")
    rec = tmp_path / "rec"
    _recording(rec, "render_002_0", 6, True, "data/raw/main", 2)
    _recording(rec, "render_003_0", 4, False, "data/raw/main", 3)
    out = convert(rec, tmp_path / "out" / "sim_render", template=tpl)
    assert out["episodes"] == 1 and out["frames"] == 6 and out["left_out_failed"] == 1
    ds = DatasetFolder(out["root"])
    assert ds.info["total_episodes"] == 1 and ds.fps == 10 and len(ds.frames()) == 6
    assert ds.frames()["raw_action_row"].tolist() == [2, 4, 6, 8, 10, 12]
    ex = ds.episode_extras()[0]
    assert ex["derived_from"] == {"source": "data/raw/main", "episode_index": 2} and ex["seed"] == 102
    stats = json.loads((Path(out["root"]) / "meta" / "stats.json").read_text())
    assert np.array(stats[CAMS[0]]["mean"]).shape == (3, 1, 1) and len(stats["action"]["mean"]) == 16
    with av.open(str(ds.episode_video_file(CAMS[1], 0, 0))) as c:
        assert sum(1 for _ in c.decode(video=0)) == 6
    rec_ = build_demo_records(out["root"])[0]
    from hackpipe.common.ids import demo_id
    assert rec_["meta"]["derived_from"] == demo_id("data/raw/main", 2)
    with pytest.raises(FileExistsError):
        convert(rec, tmp_path / "out" / "sim_render", template=tpl)


def test_derived_demos_keep_their_originals_split(approved_v1):
    from hackpipe.common.config import source_key
    from hackpipe.dataset.versioning import LeakageError, create_dataset_version, load_manifest
    from hackpipe.pipeline.auto import _import_and_check
    tmp, reg, src = approved_v1
    v1 = load_manifest("ds-v1", reg)
    by_split = {k: sorted(i for s, i in v1["episodes"][k]) for k in ("train", "val", "test")}
    copy = make_dataset(tmp / "copy", n_episodes=3, frames=30)
    picks = [by_split["train"][0], by_split["val"][0], by_split["train"][1]]
    (copy / "meta" / "fr3_episodes_extra.jsonl").write_text("".join(json.dumps(
        {"episode_index": i, "seed": 500 + i, "derived_from": {"source": source_key(src), "episode_index": e}}) + "\n"
        for i, e in enumerate(picks)))
    _import_and_check(str(copy), reg)
    m = create_dataset_version(reg, base="ds-v1")
    ids = {d["episode_index"]: d["demo_id"] for d in reg.demos(source=source_key(copy))}
    assert ids[0] in m["demo_ids"]["train"] and ids[1] in m["demo_ids"]["val"] and ids[2] in m["demo_ids"]["train"]

    bad = make_dataset(tmp / "bad", n_episodes=1, frames=30)
    (bad / "meta" / "fr3_episodes_extra.jsonl").write_text(json.dumps(
        {"episode_index": 0, "seed": 777, "derived_from": {"source": source_key(src),
                                                           "episode_index": by_split["test"][0]}}) + "\n")
    _import_and_check(str(bad), reg)
    with pytest.raises(LeakageError, match="copies of test demos"):
        create_dataset_version(reg, base="ds-v1")


@pytest.fixture
def approved_v1(workspace, tmp_path):
    from hackpipe.dataset.versioning import approve
    from hackpipe.pipeline.auto import run_auto
    tmp, reg = workspace
    src = make_dataset(tmp / "src", n_episodes=20, frames=30)
    run_auto(str(src), reg)
    approve("ds-v1", reg, by="t")
    return tmp, reg, src


def test_curriculum_train_sources_filter():
    from hackpipe.training import curriculum as cur
    m = {"version": "ds-v9", "episodes": {"train": [["data/raw/a", 0], ["data/raw/b", 1]], "val": [["data/raw/a", 2]]}}
    assert cur.split_episodes(m, "train", ["a"]) == {"data/raw/a": [0]}
    assert cur.split_episodes(m, "val", ["a"]) == {"data/raw/a": [2]}
    with pytest.raises(ValueError, match="not train sources"):
        cur.split_episodes(m, "train", ["c"])
    s = cur.settings({"args": {"train_sources": ["a"], "image_aug": True}})
    assert s["train_sources"] == ["a"] and s["image_aug"] and s["state_noise_std"] == 0.0


def test_augment_keeps_shape_and_range():
    torch = pytest.importorskip("torch")
    from hackpipe.training.curriculum import augment
    torch.manual_seed(0)
    img = torch.rand(3, 224, 224)
    out = augment(img)
    assert out.shape == img.shape and float(out.min()) >= 0 and float(out.max()) <= 1 and not torch.equal(out, img)


def test_import_refuses_short_or_repeated_runs(sim_setup):
    from hackpipe.evaluation.sim_import import ImportRefused, import_results
    from hackpipe.training.train import register_external
    tmp, reg, main, test, suite, ver = sim_setup
    m = register_external(str(tmp / "ckpt"), ver, "smolvla_team", reg)
    p = tmp / "results.json"
    p.write_text(json.dumps(_results([0, 1, 2], stages=(4, 1, 0))))
    with pytest.raises(ImportRefused, match="3 episodes, the suite has 4"):
        import_results(m, str(suite), p, None, reg)
    p.write_text(json.dumps(_results([0, 1, 1, 3])))
    with pytest.raises(ImportRefused, match="repeat"):
        import_results(m, str(suite), p, None, reg)


def test_train_sources_as_a_single_name():
    from hackpipe.training import curriculum as cur
    assert cur.settings({"args": {"train_sources": "fr3_duo_handover_84"}})["train_sources"] == ["fr3_duo_handover_84"]
    assert cur.source_names(["data/raw/a", "b"]) == ["a", "b"] and cur.source_names(None) is None


def _copies(tmp, name, src, picks, seed0):
    from hackpipe.common.config import source_key
    d = make_dataset(tmp / name, n_episodes=len(picks), frames=32)     # other content: a copy, not a duplicate
    (d / "meta" / "fr3_episodes_extra.jsonl").write_text("".join(json.dumps(
        {"episode_index": i, "seed": seed0 + i, "derived_from": {"source": source_key(src), "episode_index": e}}) + "\n"
        for i, e in enumerate(picks)))
    return d


def test_copies_follow_originals_without_a_base_and_when_added_together(workspace):
    from hackpipe.common.config import source_key
    from hackpipe.dataset.versioning import create_dataset_version
    from hackpipe.pipeline.auto import _import_and_check
    tmp, reg = workspace
    src = make_dataset(tmp / "src", n_episodes=20, frames=30)
    cp = _copies(tmp, "cp", src, list(range(20)), 500)
    _import_and_check(str(src), reg)
    _import_and_check(str(cp), reg)
    m = create_dataset_version(reg)                      # no base: originals + copies split together
    orig = {d["episode_index"]: d["demo_id"] for d in reg.demos(source=source_key(src))}
    copy = {d["episode_index"]: d["demo_id"] for d in reg.demos(source=source_key(cp))}
    where = {i: k for k in ("train", "val", "test") for i in m["demo_ids"][k]}
    assert all(where[orig[e]] == where[copy[e]] for e in range(20))
    assert len({where[orig[e]] for e in range(20)}) > 1          # still a real split
