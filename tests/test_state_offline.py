"""Moving the registry between machines (laptop <-> Colab) and offline evaluation scoring."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest


def test_state_export_import_roundtrip(workspace, dataset, monkeypatch, tmp_path):
    from hackpipe.common import state
    from hackpipe.dataset.versioning import approve, create_dataset_version, load_manifest
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg, by="t")
    fp = load_manifest("ds-v1", reg)["fingerprint"]
    out = state.export(tmp / "state.zip")
    assert out["datasets"] == [{"version": "ds-v1", "status": "approved", "fingerprint": fp}]

    # "another machine": empty registry + outputs elsewhere
    monkeypatch.setenv("HACKPIPE_OUTPUTS_DIR", str(tmp_path / "colab" / "outputs"))
    monkeypatch.setenv("HACKPIPE_REGISTRY", str(tmp_path / "colab" / "registry" / "registry.db"))
    monkeypatch.setenv("HACKPIPE_MACHINE", "colab")
    r = state.import_state(tmp / "state.zip")
    assert r["backup"] is None and r["writer"] == "colab"
    from hackpipe.common.registry import Registry
    reg2 = Registry()
    m = load_manifest("ds-v1", reg2)                       # the stored path is another machine's: falls back here
    assert m["fingerprint"] == fp and m["status"] == "approved"
    assert Path(reg2.get_dataset("ds-v1")["meta"]["card"]).is_relative_to(tmp_path / "colab")
    with pytest.raises(FileExistsError, match="--replace"):
        state.import_state(tmp / "state.zip")
    r = state.import_state(tmp / "state.zip", replace=True)
    assert Path(r["backup"], "registry.db").exists()
    reg2.close()


def test_import_refuses_other_zips(tmp_path):
    import zipfile

    from hackpipe.common import state
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("hello.txt", "hi")
    with pytest.raises(ValueError, match="not a `hackpipe state export`"):
        state.import_state(z)


def test_portable_paths():
    from hackpipe.common.config import REPO_ROOT
    from hackpipe.common.registry import portable
    assert portable(REPO_ROOT / "outputs" / "datasets" / "ds-v1" / "manifest.json") == "outputs/datasets/ds-v1/manifest.json"
    assert portable("/elsewhere/x.json") == "/elsewhere/x.json" and portable("outputs/a") == "outputs/a"


def _table(n=60, stride=1):
    S = np.cumsum(np.full((n, 16), 0.01, np.float32), 0)
    A = np.roll(S, -1, 0)
    sub = np.array(["Pick" if i < n // 2 else "Place" for i in range(n)], object)
    return {"name": "h", "stride": stride, "S": S, "A": A, "rows": np.arange(0, n, 2),
            "sub_end": np.where(np.arange(n) < n // 2, n // 2, n), "ep_end": np.full(n, n),
            "sub_text": sub, "full_text": np.full(n, "Do it", object), "ep": np.zeros(n, int)}


def test_offline_score_perfect_and_hold_still():
    from hackpipe.evaluation.offline import score
    t = _table()

    def oracle(tab, rows, mode):
        from hackpipe.training.curriculum import chunk_for
        return np.stack([tab["A"][chunk_for(tab, r, 1 if mode == "full" else 0, 5, "mask")[0]] for r in rows])

    res = score([t], oracle, chunk=5, every=1, batch=4)
    allrow = next(r for r in res["rows"] if r["instruction"] == "full" and r["subtask"] == "h · all")
    assert allrow["first_arm_rad"] == 0 and allrow["chunk_arm_rad"] == 0 and allrow["hold_still_arm_rad"] > 0
    assert {r["subtask"] for r in res["rows"]} == {"h · Pick", "h · Place", "h · all"}
    assert res["predictions"] == 2 * len(t["rows"])

    still = score([t], lambda tab, rows, mode: np.stack([np.repeat(tab["S"][r][None], 5, 0) for r in rows]), 5, 1)
    a = next(r for r in still["rows"] if r["subtask"] == "h · all")
    assert a["chunk_arm_rad"] == pytest.approx(a["hold_still_arm_rad"])


def test_offline_rate_from_run(tmp_path):
    from hackpipe.evaluation.offline import model_rate
    out = tmp_path / "m"
    out.mkdir()
    (out / "train_config.json").write_text(json.dumps({"args": {"rate_hz": 6, "chunk": 20}}))
    assert model_rate({"meta": {"trainer": "curriculum", "output_dir": str(out)}}, 30) == (6.0, 20)
    assert model_rate({"meta": {"trainer": "lerobot-train", "output_dir": str(out)}}, 30) == (30.0, None)


def test_eval_cli_keeps_adapter_flag(workspace, dataset):
    from hackpipe.cli import main
    from hackpipe.dataset.versioning import approve, create_dataset_version
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    from hackpipe.training.train import register_external
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg, by="t")
    m = register_external(str(tmp / "ck"), "ds-v1", "p", reg)
    main(["eval", "--model", m, "--adapter", "dummy"])           # was parsed onto `state` by mistake
    assert reg.evals()


def test_model_and_card_paths_stay_portable(workspace, monkeypatch):
    import sqlite3

    from hackpipe.common.config import project_config, resolve
    tmp, reg = workspace
    out = resolve(project_config()["project"]["outputs_dir"])
    reg.add_model("m-v1", "ds-v1", "p", "trained", "/content/repo/outputs/train/m-v1/checkpoints/last/pretrained_model",
                  "/content/repo/outputs/runs/m-v1/run_record.json", {"output_dir": "/content/repo/outputs/train/m-v1"})
    m = reg.get_model("m-v1")                                     # a Colab row read here → this machine's outputs
    assert m["checkpoint_dir"] == str(out / "train" / "m-v1" / "checkpoints" / "last" / "pretrained_model")
    assert m["meta"]["output_dir"] == str(out / "train" / "m-v1")
    reg.update_model_meta("m-v1", {"x": 1})
    raw = sqlite3.connect(reg.path).execute("SELECT meta FROM models").fetchone()[0]
    assert "/content/repo/outputs/train/m-v1" in raw               # stored value untouched by the update


def test_state_replace_drops_stale_outputs_and_prunes_backups(workspace, tmp_path, monkeypatch):
    from hackpipe.common import state
    from hackpipe.common.config import project_config, resolve
    tmp, reg = workspace
    reg.close()
    z = state.export(tmp / "s.zip")["zip"]
    out = resolve(project_config()["project"]["outputs_dir"])
    (out / "runs" / "m-old").mkdir(parents=True)
    (out / "runs" / "m-old" / "offline_eval_test.json").write_text("{}")
    for _ in range(5):
        state.import_state(z, replace=True)
    assert not (out / "runs" / "m-old").exists()                   # replaced, not mixed in
    assert len(list(resolve(project_config()["project"]["registry_path"]).parent.glob("backup-*"))) == state.KEEP_BACKUPS
