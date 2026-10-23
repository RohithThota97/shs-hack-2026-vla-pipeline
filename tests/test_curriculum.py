"""Sub-task labels (robot signals → boundaries + checks), a dataset version that adds them without changing the
split, and the curriculum trainer's plan / sampling / data rows (the torch loop itself runs on Colab)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hackpipe.dataset import subtasks as sbt
from hackpipe.training import curriculum as cur


# ---------------------------------------------------------------- labelling
def _handover_signals(T=500, fps=30):
    """Left picks (close 95), carries (120-180), right reaches (200-260) and closes (270), left opens (300) and
    retracts (320-360), right places (380-450) and opens (460)."""
    S, A = np.zeros((T, 16)), np.zeros((T, 16))
    A[:, 7] = A[:, 15] = S[:, 7] = S[:, 15] = 1.0

    def move(j, a, b):
        S[a:b, j] = np.linspace(0, 0.01 * (b - a), b - a) + S[a - 1, j]
        S[b:, j] = S[b - 1, j]

    def grip(g, t, close):
        lo, hi = (1.0, 0.0) if close else (0.0, 1.0)
        A[t:t + 6, g] = np.linspace(lo, hi, 6)
        A[t + 6:, g] = hi
        S[t + 1:t + 11, g] = np.linspace(lo, 0.3 if close else 1.0, 10)
        S[t + 11:, g] = 0.3 if close else 1.0

    for j, a, b in ((0, 30, 90), (0, 120, 180), (8, 200, 260), (0, 320, 360), (8, 380, 450)):
        move(j, a, b)
    grip(7, 95, True)
    grip(15, 270, True)
    grip(7, 300, False)
    grip(15, 460, False)
    A[:, :7], A[:, 8:15] = S[:, :7], S[:, 8:15]
    return S, A


def test_handover_segmentation_and_checks():
    S, A = _handover_signals()
    j = sbt.Joints({})
    b, ev, still = sbt.segment("handover", S, A, 30, j)
    assert b is not None and len(b) == 6 and b == sorted(b) and b[0] == 0 and b[-1] == len(S)
    assert ev["left_close"] == 95 and ev["right_close"] == 270 and ev["left_open"] == 300
    assert sbt.check("handover", b, ev, still, A, j) == []
    full, subs = sbt.texts("handover")
    assert len(subs) == 5 and "serving tray" in full


def test_episode_without_grasp_is_flagged():
    S, A = _handover_signals()
    A[:, 15] = 1.0                                         # right gripper never closes
    j = sbt.Joints({})
    b, ev, still = sbt.segment("handover", S, A, 30, j)
    assert b is None and sbt.check("handover", b, ev, still, A, j)[0].startswith("an event was not found")


def test_cube_texts_name_colours():
    full, subs = sbt.texts("cubes", {"cube_left": "red", "cube_base": "unknown", "cube_right": "blue"})
    assert full == "Stack the red cube on the other cube, then put the blue cube on top."
    assert subs[2] == "Pick up the blue cube with the right gripper."
    assert sbt.color_name([200, 30, 30]) == "red" and sbt.color_name([240, 240, 240]) == "white"


# ---------------------------------------------------------------- dataset version with labels
def _fake_labels(directory: Path, manifest: dict, flagged=(), frames=30, skip=()) -> Path:
    """labels.json + tables for every train/val episode of the version (2 sub-tasks of 'handover' rules)."""
    directory.mkdir(parents=True, exist_ok=True)
    eps: dict = {}
    for split in ("train", "val"):
        for s, i in manifest["episodes"][split]:
            if (s, i) not in skip:
                eps.setdefault(s, []).append(i)
    index = {"label_version": 1, "code_sha256": "c" * 64, "base": manifest["version"], "sources": {}}
    for s, e in eps.items():
        name = Path(s).name
        seg = pd.DataFrame([{"episode_index": i, "subtask_order": k, "start_frame": k * frames // 2,
                             "end_frame": (k + 1) * frames // 2, "duration_s": 1.5,
                             "text": ["Pick up the box with the left gripper.", "Bring the box to the handover position."][k],
                             "skill": ["reach_grasp", "transport_place"][k]} for i in e for k in range(2)])
        epi = pd.DataFrame([{"episode_index": i, "n_frames": frames, "full_text": sbt.HANDOVER_FULL, "colors": "{}",
                             "events": "{}", "problems": "[]", "flagged": i in flagged} for i in e])
        seg.to_csv(directory / f"{name}_segments.csv", index=False)
        epi.to_csv(directory / f"{name}_episodes.csv", index=False)
        index["sources"][s] = {"kind": "handover", "fps": 10.0, "episodes": len(e), "flagged": sorted(flagged),
                               "unlabelled": [], "segments": f"{name}_segments.csv",
                               "episodes_file": f"{name}_episodes.csv", "sha256": "s" * 64}
    index["sha256"] = "a1b2c3" + "0" * 58
    (directory / "labels.json").write_text(json.dumps(index))
    return directory


@pytest.fixture
def approved(workspace, dataset):
    from hackpipe.dataset.versioning import approve, create_dataset_version
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg, by="t")
    return tmp, reg


def test_labels_make_a_new_version_with_the_same_split(approved):
    from hackpipe.dataset.versioning import NothingNewError, approve, create_dataset_version, load_manifest
    tmp, reg = approved
    v1 = load_manifest("ds-v1", reg)
    with pytest.raises(NothingNewError):
        create_dataset_version(reg, base="ds-v1")
    labels = _fake_labels(tmp / "labels", v1, flagged=(v1["episodes"]["train"][0][1],))
    m = create_dataset_version(reg, base="ds-v1", subtasks=labels)
    assert m["version"] == "ds-v2" and m["demo_ids"] == v1["demo_ids"]          # same demos, same split
    assert m["fingerprint"] != v1["fingerprint"]                                 # labels are part of it
    assert m["subtasks"]["sha256"].startswith("a1b2c3") and Path(m["subtasks"]["dir"]).name == "subtasks"
    assert (Path(reg.get_dataset("ds-v2")["manifest_path"]).parent / "subtasks" / "labels.json").exists()
    card = open(reg.get_dataset("ds-v2")["meta"]["card"]).read()
    assert "## Sub-task labels (curriculum training)" in card
    approve("ds-v2", reg, by="t")
    with pytest.raises(NothingNewError):                     # labels alone don't repeat: nothing new now
        create_dataset_version(reg, base="ds-v2")


def test_labels_must_cover_every_train_val_episode(approved):
    from hackpipe.dataset.versioning import create_dataset_version, load_manifest
    tmp, reg = approved
    v1 = load_manifest("ds-v1", reg)
    s, i = v1["episodes"]["val"][0]
    labels = _fake_labels(tmp / "labels", v1, skip={(s, i)})
    with pytest.raises(RuntimeError, match="don't cover"):
        create_dataset_version(reg, base="ds-v1", subtasks=labels)


# ---------------------------------------------------------------- curriculum plan + sampling
def test_plan_steps_and_resume_point():
    plan = cur.plan_of({})
    steps = cur.stage_steps(plan, 20000)
    assert steps == [3000, 4000, 4000, 9000] and sum(cur.stage_steps(plan, 7)) == 7
    assert cur.stage_at(0, steps) == (0, 0) and cur.stage_at(3000, steps) == (1, 0)
    assert cur.stage_at(3500, steps) == (1, 500) and cur.stage_at(20000, steps) == (3, 9000)
    assert cur.plan_of({"args": {"no_curriculum": True}})["stages"][0]["skills"] == cur.SKILLS
    with pytest.raises(ValueError):
        cur.plan_of({"curriculum": {"stages": [{"name": "x", "frac": 1, "skills": ["juggling"]}]}})


def test_weights_balance_datasets_and_rehearse():
    groups = np.array(["h:reach_grasp"] * 10 + ["c:reach_grasp"] * 90 + ["h:transport_place"] * 5 +
                      ["c:transport_place"] * 95)
    w = cur.weights(groups, ["reach_grasp"], ["reach_grasp"], 0.7)
    assert w[groups == "h:reach_grasp"].sum() == pytest.approx(0.5) and w[groups == "h:transport_place"].sum() == 0
    w = cur.weights(groups, ["reach_grasp", "transport_place"], ["transport_place"], 0.7)
    assert w[groups == "h:transport_place"].sum() == pytest.approx(0.35)       # new skill: 70% over 2 datasets
    assert w[groups == "c:reach_grasp"].sum() == pytest.approx(0.15)           # rehearsal: 30% over 2 datasets
    mix = cur.group_mix(["h:reach_grasp", "h:transport_place", "h:full_task"], cur.plan_of({}))
    assert mix[0] == {"h:reach_grasp": 1.0} and mix[3]["h:full_task"] == pytest.approx(0.7)


def test_draws_are_seeded_and_resume_exactly():
    w = np.array([1, 1, 2, 0], float)
    a = cur.stage_draws(w, 200, seed=1000, stage=2, paraphrase_prob=0.3)
    assert (a == cur.stage_draws(w, 200, 1000, 2, 0.3)).all()
    assert (cur.stage_draws(w, 200, 1000, 2, 0.3, skip=40) == a[40:]).all()      # resume continues the same order
    assert not (a // 3 == 3).any() and set(np.unique(a % 3)) <= {0, 1, 2}         # weight 0 never drawn
    assert not (cur.stage_draws(w, 200, 1000, 3, 0.3) == a).all()


def test_reword_and_gripper_lag_fix():
    assert cur.reword("Pick up the red cube with the left gripper.", 1) == "Grab the red cube using the left hand."
    assert cur.reword("Pick up the box with the left gripper.", 0) == "Pick up the box with the left gripper."
    S, A = np.zeros((20, 16)), np.ones((20, 16))
    A[10:, 7] = 0.0                                   # command flips at 10 …
    S[7:, 7] = np.linspace(1, 0.4, 13)                # … but the gripper starts closing at 7
    S[:7, 7] = 1.0
    fixed, shifts = cur.fix_gripper_lag(S, A, [7, 15])
    assert shifts == [3] and fixed[7, 7] == 0.0 and fixed[6, 7] == 1.0


def test_tables_rows_and_chunks(approved):
    from hackpipe.dataset.versioning import load_manifest
    tmp, reg = approved
    m = load_manifest("ds-v1", reg)
    train = cur.split_episodes(m, "train")
    (src, eps), = train.items()
    labels = _fake_labels(tmp / "labels", m, flagged=(eps[0],))
    lab = sbt.load_labels(labels)
    s = cur.settings({"args": {"rate_hz": 5}})                       # synthetic data: 10 fps → every 2nd frame
    t = cur.build_table(src, eps, {**lab[src], "kind": "handover", "flagged": [eps[0]]}, s, [7, 15])
    assert t["stride"] == 2 and eps[0] in t["flagged"]
    assert set(t["ep"][t["rows"]].tolist()) == set(eps) - {eps[0]}
    r = t["rows"][0]
    idx, pad, text = cur.chunk_for(t, r, 0, 10, "hold")
    assert text.startswith("Pick up the box") and idx.max() == t["sub_end"][r] - 1 and not pad.any()
    _, pad_m, _ = cur.chunk_for(t, r, 0, 10, "mask")
    assert pad_m.any()
    idx_f, pad_f, text_f = cur.chunk_for(t, r, 1, 10, "hold")
    assert text_f == sbt.HANDOVER_FULL and idx_f.max() < t["ep_end"][r]
    with pytest.raises(ValueError):
        cur.split_episodes(m, "test")                               # never reads the test split


# ---------------------------------------------------------------- hackpipe train
def _curriculum_cfg(tmp):
    p = tmp / "curr.yaml"
    p.write_text("trainer: curriculum\npolicy_name: smolvla_curriculum\nseed: 1\nkeep_checkpoints: 2\n"
                 "args:\n  policy.path: lerobot/smolvla_base\n  steps: 100\n  rate_hz: 5\n")
    return p


def test_dry_run_records_the_plan_and_needs_labels(approved):
    from hackpipe.dataset.versioning import approve, create_dataset_version, load_manifest
    from hackpipe.training.train import train
    tmp, reg = approved
    cfg = _curriculum_cfg(tmp)
    with pytest.raises(RuntimeError, match="no sub-task labels"):
        train(str(cfg), "ds-v1", dry_run=True, registry=reg)
    create_dataset_version(reg, base="ds-v1", subtasks=_fake_labels(tmp / "labels", load_manifest("ds-v1", reg)))
    approve("ds-v2", reg, by="t")
    rec = train(str(cfg), "ds-v2", dry_run=True, registry=reg, overrides={"no_curriculum": True})
    assert rec["trainer"] == "curriculum" and "hackpipe.training.curriculum" in rec["command"]
    assert rec["curriculum"]["stage_steps"] == [100] and rec["curriculum"]["no_curriculum"] is True
    written = Path(rec["command"][rec["command"].index("--config") + 1])
    assert written.name == "train_config_dryrun.json" and json.loads(written.read_text())["args"]["no_curriculum"]
    assert reg.models() == []                                       # a dry run registers nothing


def test_resume_without_config_uses_the_runs_config(approved, tmp_path):
    from hackpipe.training.train import train
    tmp, reg = approved
    out = tmp / "outputs" / "train" / "m-v1"
    reg.add_model("m-v1", "ds-v1", "smolvla", "failed", str(out), "", {"output_dir": str(out),
                  "train_config_file": "configs/train/smolvla_fr3_duo.yaml", "trainer": "lerobot-train"})
    rec = train(None, "ds-v1", dry_run=True, registry=reg, resume_model="m-v1")
    assert rec["command"][:2] == ["lerobot-train", "--resume=true"]


def test_curriculum_frames(approved):
    from hackpipe.dataset.versioning import approve, create_dataset_version, load_manifest
    from hackpipe.training.progress import curriculum_frames
    tmp, reg = approved
    m1 = load_manifest("ds-v1", reg)
    flagged = m1["episodes"]["train"][0][1]
    create_dataset_version(reg, base="ds-v1", subtasks=_fake_labels(tmp / "labels", m1, flagged=(flagged,)))
    approve("ds-v2", reg, by="t")
    n_train = len(m1["demo_ids"]["train"])
    frames = {d["demo_id"]: int(d["meta"]["n_frames"]) for d in reg.demos()}
    want = sum(-(-frames[i] // 2) for i in m1["demo_ids"]["train"]) - (-(-30 // 2))
    assert curriculum_frames("ds-v2", reg, 5) == want and n_train > 1


def test_train_page_curriculum_config(approved, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from hackpipe.dataset.versioning import approve, create_dataset_version, load_manifest
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop-test")
    tmp, reg = approved
    page = str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/train.py")
    at = AppTest.from_file(page, default_timeout=30)
    at.run()
    at.selectbox[1].select("configs/train/smolvla_curriculum.yaml").run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("has none" in w.value for w in at.warning)                       # ds-v1 has no labels
    assert any(b.label == "Label ds-v1 and create the draft" for b in at.button)
    assert next(b for b in at.button if b.label == "Start training").disabled

    create_dataset_version(reg, base="ds-v1", subtasks=_fake_labels(tmp / "labels", load_manifest("ds-v1", reg)))
    approve("ds-v2", reg, by="t")
    at = AppTest.from_file(page, default_timeout=30)
    at.run()
    at.selectbox[1].select("configs/train/smolvla_curriculum.yaml").run()
    assert not at.exception, [e.message for e in at.exception]
    plan_table = lambda: next(d.value for d in at.dataframe if "steps" in d.value.columns)  # noqa: E731
    stages = plan_table()
    assert list(stages["stage"]) == [s["name"] for s in cur.DEFAULT_PLAN["stages"]]
    assert {m.label for m in at.metric} >= {"Train samples (6 Hz)"}
    at.toggle[0].set_value(True).run()                                            # baseline: one mixed stage
    assert not at.exception and list(plan_table()["stage"]) == ["all_skills_mixed"]
