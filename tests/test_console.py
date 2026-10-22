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
    _run(CONSOLE / "pages" / f"{page}.py")


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
    at = _run(CONSOLE / "pages" / "overview.py")
    assert any("Review and approve ds-v1" in m.value for m in at.markdown)
