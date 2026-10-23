"""Every console page renders without errors, on an empty and on a filled registry (headless AppTest)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from hackpipe.dataset.versioning import create_dataset_version  # noqa: E402
from hackpipe.ingest.importer import ingest  # noqa: E402
from hackpipe.quality.report import run_qc  # noqa: E402
from hackpipe.status import overview  # noqa: E402

CONSOLE = Path(__file__).resolve().parents[1] / "src" / "hackpipe" / "console"
PAGES = ["overview", "data", "train", "evaluate", "deploy", "improve"]


def _run(path):
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_on_empty_registry(workspace, page):
    _run(CONSOLE / "views" / f"{page}.py")


def test_app_shell_renders(workspace):
    at = _run(CONSOLE / "app.py")
    text = " ".join(m.value for m in at.markdown)
    assert "FR3 Duo · Demo-to-Deployment" in text and "Writer" in text


def test_overview_reflects_registry(workspace, dataset):
    _, reg = workspace
    assert overview(reg)["next"][0].startswith("Add demonstrations")
    ingest(dataset, reg)
    assert "not quality-checked" in overview(reg)["next"][0]
    run_qc(dataset, reg)
    create_dataset_version(reg)
    ov = overview(reg)
    assert ov["next"][0].startswith("Review and approve ds-v1")
    assert ov["stages"][1]["status"] == "waiting" and ov["versions"]["dataset"] == "ds-v1"
    at = _run(CONSOLE / "views" / "overview.py")
    assert any("Review and approve ds-v1" in m.value for m in at.markdown)


def test_data_page_on_filled_registry(workspace):
    from conftest import add_fr3_extras, make_dataset
    from hackpipe.pipeline.auto import run_auto
    tmp, reg = workspace
    root = add_fr3_extras(make_dataset(tmp / "ds", n_episodes=6, frames=30), 6, 30)
    assert run_auto(str(root), reg)["version"] == "ds-v1"
    at = _run(CONSOLE / "views" / "data.py")
    assert [t.label for t in at.tabs] == ["Add demonstrations", "Demos", "Episode viewer", "Quality", "Versions"]
    assert any("ds-v1" in e.label for e in at.expander)
    assert any(b.label == "Approve ds-v1" for b in at.button)


def test_preview_counts_and_estimate(workspace):
    from conftest import add_fr3_extras, make_dataset
    from hackpipe.ingest.importer import import_demos
    from hackpipe.pipeline.preview import preview
    tmp, reg = workspace
    root = add_fr3_extras(make_dataset(tmp / "ds", n_episodes=6, frames=30), 6, 30)
    p = preview(str(root), reg)
    assert (p["episodes"], p["new"], p["registered"]) == (6, 6, 0)
    assert p["work_units"] == 6 * 30 * 1 and p["estimate"]["basis"].startswith("no estimate yet")
    import_demos(root, reg)
    p = preview(str(root), reg)
    assert (p["new"], p["registered"], p["pending_qc"]) == (0, 6, 6)


def test_candidate_folders(tmp_path):
    from conftest import make_dataset
    from hackpipe.pipeline.preview import candidate_folders
    make_dataset(tmp_path / "data" / "incoming" / "batch1", n_episodes=1)
    make_dataset(tmp_path / "data" / "raw" / "zipdir" / "inner", n_episodes=1)
    (tmp_path / "data" / "incoming" / "not_a_dataset").mkdir()
    found = candidate_folders(tmp_path)
    assert any(f.endswith("data/incoming/batch1") for f in found)
    assert any(f.endswith("data/raw/zipdir/inner") for f in found) and len(found) == 2


def test_token_protects_console(workspace, monkeypatch):
    monkeypatch.setenv("HACKPIPE_CONSOLE_TOKEN", "s3cret")
    at = AppTest.from_file(str(CONSOLE / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception and "protected" in at.info[0].value
    assert not any("FR3 Duo · Demo-to-Deployment" in m.value for m in at.markdown)
    at.text_input[0].input("wrong").run()
    assert not any("FR3 Duo · Demo-to-Deployment" in m.value for m in at.markdown)
    at.text_input[0].input("s3cret").run()
    assert any("FR3 Duo · Demo-to-Deployment" in m.value for m in at.markdown)


def test_add_tab_every_source_renders(workspace):
    at = _run(CONSOLE / "views" / "data.py")
    def radio():
        return next(r for r in at.radio if r.label == "From")
    for opt in radio().options:
        radio().set_value(opt).run()
        assert not at.exception, (opt, [e.message for e in at.exception])
