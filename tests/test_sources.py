"""Add demonstrations from anywhere: discovery of nested folders, local folder / zip fetch, naming, one version
for several new folders."""
import base64
import zipfile

import pytest
from conftest import make_dataset

from hackpipe.pipeline import sources
from hackpipe.pipeline.auto import run_auto_many
from hackpipe.pipeline.preview import candidate_folders, new_folders


def test_find_roots_nested_and_not_inside_a_dataset(tmp_path):
    a = make_dataset(tmp_path / "top" / "fr3_duo_rand500" / "fr3_duo_rand500", n_episodes=2)
    b = make_dataset(tmp_path / "top" / "b", n_episodes=2)
    assert sources.find_roots(tmp_path / "top") == [b, a]
    assert sources.find_roots(tmp_path / "missing") == []


@pytest.mark.parametrize("src,name", [
    ("/content/drive/MyDrive/shs/data/fr3_duo_rand500/fr3_duo_rand500", "fr3_duo_rand500"),
    ("FR3Duo_Handover_Dataset_lerobot3.zip", "FR3Duo_Handover_Dataset_lerobot3"),
    ("lerobot/aloha sim", "aloha_sim"), ("https://github.com/owner/repo.git", "repo"), ("s3://b/x/ds/", "ds")])
def test_dataset_name(src, name):
    assert sources.dataset_name(src) == name


def test_fetch_local_folder_is_linked_zip_is_extracted(tmp_path):
    ds = make_dataset(tmp_path / "elsewhere" / "night", n_episodes=2)
    repo = tmp_path / "repo"
    roots = sources.fetch("local", str(ds.parent), name="night_set", base=repo)
    link = repo / "data" / "incoming" / "night_set"
    assert link.is_symlink() and roots == [link / "night"]

    z = tmp_path / "up.zip"
    with zipfile.ZipFile(z, "w") as f:
        for p in ds.rglob("*"):
            f.write(p, "inner/" + p.relative_to(ds).as_posix())
    roots = sources.fetch("local", str(z), base=repo)
    assert roots == [repo / "data" / "incoming" / "up" / "inner"]
    with pytest.raises(FileExistsError):
        sources.fetch("local", str(z), base=repo)


def test_fetch_local_inside_repo(tmp_path):
    repo = tmp_path / "repo"
    raw = make_dataset(repo / "data" / "raw" / "a", n_episodes=2)
    assert sources.fetch("local", "data/raw/a", base=repo) == [raw]          # already in place
    other = make_dataset(repo / "data" / "organiser" / "b" / "b", n_episodes=2)
    roots = sources.fetch("local", str(other.parent), base=repo)              # elsewhere in the repo → linked
    assert roots == [repo / "data" / "incoming" / "b" / "b"]


def test_fetch_without_dataset_leaves_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    repo = tmp_path / "repo"
    with pytest.raises(FileNotFoundError):
        sources.fetch("local", str(tmp_path / "empty"), name="x", copy=True, base=repo)
    assert not (repo / "data" / "incoming" / "x").exists()


def test_zip_with_unsafe_paths_is_refused(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        sources.extract_zip(z, tmp_path / "out")


def test_github_token_goes_in_env_not_url(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok123")
    env = sources._git_env()
    assert sources.github_url("owner/repo") == "https://github.com/owner/repo.git"
    assert base64.b64decode(env["GIT_CONFIG_VALUE_0"].split()[-1]).decode() == "x-access-token:tok123"


def test_gdrive_ids():
    assert sources.gdrive_id("https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrSt") == \
        ("folder", "1AbCdEfGhIjKlMnOpQrSt")
    assert sources.gdrive_id("https://drive.google.com/file/d/1XyZ0123456789abcdefg/view?usp=sharing") == \
        ("file", "1XyZ0123456789abcdefg")


def test_new_folders_and_one_version_for_all(workspace, monkeypatch):
    tmp, reg = workspace
    import hackpipe.pipeline.preview as pv
    monkeypatch.setattr(pv, "REPO_ROOT", tmp)
    make_dataset(tmp / "data" / "raw" / "a", n_episodes=6)
    make_dataset(tmp / "data" / "incoming" / "rand" / "rand", n_episodes=6)
    folders = candidate_folders(tmp)
    assert len(folders) == 2 and any(f.endswith("incoming/rand/rand") for f in folders)
    assert len(new_folders(reg, tmp)) == 2
    out = run_auto_many(folders, reg)
    assert out["version"] == "ds-v1" and len(out["sources"]) == 2
    assert new_folders(reg, tmp) == []
    assert run_auto_many(folders, reg)["version"] is None       # nothing new → no second version
