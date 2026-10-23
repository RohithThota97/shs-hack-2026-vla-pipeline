"""Watch folders (organiser download, Drive): datasets not in data/raw yet are found once and linked as
data/raw/<name>; copies (already linked, or zips of a folder that is there) are not listed again."""
import zipfile

import pytest
from conftest import make_dataset

from hackpipe.pipeline import watch


def _zip(ds, z, prefix=""):
    z.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z, "w") as f:
        for p in ds.rglob("*"):
            f.write(p, prefix + p.relative_to(ds).as_posix())
    return z


@pytest.fixture
def layout(tmp_path):
    """repo/data/raw/night84 → organiser/night84; organiser also has rand (folder + zip) and a zip-only set."""
    repo, org = tmp_path / "repo", tmp_path / "organiser"
    night = make_dataset(org / "Datasets" / "night_ep" / "night84", n_episodes=4)
    (repo / "data" / "raw").mkdir(parents=True)
    (repo / "data" / "raw" / "night84").symlink_to(night, target_is_directory=True)
    _zip(night, org / "Datasets" / "night_ep.zip", "night84/")
    rand = make_dataset(org / "Datasets" / "rand" / "rand", n_episodes=6)
    _zip(rand, org / "Dataset" / "rand.zip", "rand/")
    only = make_dataset(tmp_path / "elsewhere" / "zip_only", n_episodes=3, frames=20)
    _zip(only, org / "Datasets" / "zip_only.zip")
    return repo, org


def test_unlinked_lists_each_new_dataset_once(layout, monkeypatch):
    repo, org = layout
    monkeypatch.setattr(watch, "pinned_names", lambda: {})
    found = watch.unlinked(repo, [org])
    assert [(d["name"], d["kind"]) for d in found] == [("rand", "folder"), ("zip_only", "zip")]


def test_link_folder_and_zip_then_nothing_left(layout, monkeypatch):
    repo, org = layout
    monkeypatch.setattr(watch, "pinned_names", lambda: {})
    for d in watch.unlinked(repo, [org]):
        dest = watch.link(d["path"], d["name"], base=repo)
        assert (dest / "meta" / "info.json").is_file()
    assert (repo / "data" / "cache" / "zip_only" / ".complete").exists()
    assert sorted(p.name for p in (repo / "data" / "raw").iterdir()) == ["night84", "rand", "zip_only"]
    assert watch.unlinked(repo, [org]) == []
    assert watch.link(org / "Datasets" / "rand" / "rand", "rand", base=repo).name == "rand"   # again: no-op


def test_link_refuses_a_different_dataset_with_the_same_name(layout):
    repo, org = layout
    with pytest.raises(FileExistsError):
        watch.link(org / "Datasets" / "rand" / "rand", "night84", base=repo)


def test_pinned_names_by_episode_count(layout, monkeypatch):
    repo, org = layout
    monkeypatch.setattr(watch, "pinned_names", lambda: {6: "fr3_rand_pinned"})
    assert watch.unlinked(repo, [org])[0]["name"] == "fr3_rand_pinned"


def test_watch_dirs_env_and_globs(tmp_path, monkeypatch):
    (tmp_path / "GoogleDrive-a@b" / "My Drive").mkdir(parents=True)
    monkeypatch.setenv("HACKPIPE_WATCH_DIRS", str(tmp_path / "GoogleDrive-*" / "My Drive") + ":/nope")
    assert watch.watch_dirs() == [tmp_path / "GoogleDrive-a@b" / "My Drive"]
    monkeypatch.setenv("HACKPIPE_WATCH_DIRS", "")
    assert watch.watch_dirs() == []


def test_add_tab_lists_and_links_watched_dataset(workspace, monkeypatch, layout):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    repo, org = layout
    monkeypatch.setenv("HACKPIPE_WATCH_DIRS", str(org))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/data.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    sel = next(s for s in at.selectbox if s.label.startswith("Dataset ("))
    opt = next(o for o in sel.options if "not linked yet" in o and "rand" in o)
    sel.set_value(opt).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(b.label.startswith("Link as data/raw/") for b in at.button)
