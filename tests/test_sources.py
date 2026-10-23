"""Add demonstrations from anywhere: discovery of nested folders, local folder / zip fetch, naming, one version
for several new folders."""
import base64
import shutil
import subprocess
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
    ("lerobot/aloha sim", "aloha_sim"), ("https://drive.google.com/drive/folders/1QVc?usp=drive_link", "1QVc"), ("https://github.com/owner/repo.git", "repo"), ("s3://b/x/ds/", "ds")])
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
    assert sources.fetch("local", str(raw), base=repo) == [raw]               # already in place
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


@pytest.mark.parametrize("bad", ["../raw/foo", "a/b", "..", ".hidden", "x y"])
def test_bad_names_are_refused(tmp_path, bad):
    ds = make_dataset(tmp_path / "ds", n_episodes=2)
    with pytest.raises(ValueError):
        sources.fetch("local", str(ds), name=bad, base=tmp_path / "repo")
    assert not (tmp_path / "repo" / "data" / "raw").exists()


def test_one_bad_folder_does_not_stop_the_others(workspace):
    tmp, reg = workspace
    good = make_dataset(tmp / "good", n_episodes=6)
    bad = tmp / "bad" / "meta"
    bad.mkdir(parents=True)
    (bad / "info.json").write_text("{not json")
    out = run_auto_many([str(tmp / "bad"), str(good)], reg)
    assert out["version"] == "ds-v1" and len(out["sources"]) == 1 and len(out["errors"]) == 1


def _git_repo(path, files):
    path.mkdir(parents=True)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    g = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*g[:3], "init", "-q"], check=True)
    subprocess.run([*g[:3], "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "x"], check=True)
    return subprocess.run([*g[:3], "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def test_github_fetch_by_commit(tmp_path):
    src = tmp_path / "src"
    ds = make_dataset(tmp_path / "tmpds", n_episodes=2)
    files = {str(p.relative_to(ds.parent)): "" for p in ds.rglob("*") if p.is_file()}
    files = {k: (ds.parent / k).read_text(errors="replace") if k.endswith(".json") else "x" for k in files}
    sha = _git_repo(src, files)
    roots = sources.fetch("github", f"file://{src}", name="g", revision=sha, base=tmp_path / "repo")
    assert [r.name for r in roots] == ["tmpds"] and not (tmp_path / "repo" / "data" / "incoming" / "g" / ".git").exists()
    roots = sources.fetch("github", f"file://{src}", name="g7", revision=sha[:7], base=tmp_path / "repo")
    assert [r.name for r in roots] == ["tmpds"]                       # short SHA as GitHub shows it


def test_github_lfs_without_git_lfs_fails_clearly(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _git_repo(src, {".gitattributes": "*.mp4 filter=lfs diff=lfs merge=lfs -text\n", "meta/info.json": "{}"})
    real = sources.shutil.which
    monkeypatch.setattr(sources.shutil, "which", lambda n: None if n == "git-lfs" else real(n))
    with pytest.raises(RuntimeError, match="git-lfs"):
        sources.fetch("github", f"file://{src}", name="g", base=tmp_path / "repo")
    assert not (tmp_path / "repo" / "data" / "incoming" / "g").exists()


def _fake_gdown(monkeypatch, tmp_path, *, folder_name=None, zip_of=None, error=None, refuse=()):
    """Like gdown 6: download_folder(skip_download=True) lists files, download(id, output) gets one.
    refuse: file suffixes Google refuses ('too many accesses'); calls records every per-file download."""
    import sys
    import types
    exc = types.ModuleType("gdown.exceptions")

    class DownloadError(Exception):
        pass

    class FileURLRetrievalError(DownloadError):
        pass
    exc.DownloadError, exc.FileURLRetrievalError = DownloadError, FileURLRetrievalError
    g = types.ModuleType("gdown")
    g.calls = []
    src = tmp_path / "drive" / folder_name if folder_name else None
    if src and not src.exists():
        make_dataset(src, n_episodes=2)
    by_id = {}

    def download_folder(id, output, skip_download, quiet):
        if error:
            raise DownloadError(f"Failed to retrieve folder contents for folder ID: {id} (status code 401).")
        out = []
        for i, p in enumerate(sorted(q for q in src.rglob("*") if q.is_file())):
            rel = f"{folder_name}/{p.relative_to(src).as_posix()}"
            by_id[f"id{i}"] = p
            out.append(types.SimpleNamespace(id=f"id{i}", path=rel, local_path=output + rel))
        return out

    def download(id, output, quiet, resume):
        g.calls.append(id)
        if error:
            raise FileURLRetrievalError("Cannot retrieve the public link of the file.")
        if id in by_id:
            if by_id[id].suffix in refuse:
                raise FileURLRetrievalError("Failed to retrieve file url:\n\n\tCannot retrieve the public link")
            shutil.copy(by_id[id], output)
            return output
        z = output + zip_of
        with zipfile.ZipFile(z, "w") as f:
            ds = make_dataset(tmp_path / "zsrc", n_episodes=2)
            for p in ds.rglob("*"):
                f.write(p, p.relative_to(ds.parent).as_posix())
        return z
    g.download_folder, g.download, g.exceptions = download_folder, download, exc
    monkeypatch.setitem(sys.modules, "gdown", g)
    monkeypatch.setitem(sys.modules, "gdown.exceptions", exc)
    return g


def test_gdrive_folder_link_named_after_drive_folder(tmp_path, monkeypatch):
    _fake_gdown(monkeypatch, tmp_path, folder_name="night_set")
    repo = tmp_path / "repo"
    roots = sources.fetch("gdrive", "https://drive.google.com/drive/folders/1QVcgaNooB0MHn6Dj?usp=drive_link",
                          base=repo)
    assert roots == [repo / "data" / "incoming" / "night_set" / "night_set"]
    assert [p.name for p in (repo / "data" / "incoming").iterdir()] == ["night_set"]


def test_gdrive_zip_link_named_after_zip(tmp_path, monkeypatch):
    _fake_gdown(monkeypatch, tmp_path, zip_of="fr3_more.zip")
    repo = tmp_path / "repo"
    roots = sources.fetch("gdrive", "https://drive.google.com/file/d/1XyZ0123456789abcdefg/view", base=repo)
    assert roots == [repo / "data" / "incoming" / "fr3_more" / "zsrc"]


@pytest.mark.parametrize("link", ["https://drive.google.com/drive/folders/1QVcgaNooB0MHn6Dj",
                                  "https://drive.google.com/file/d/1XyZ0123456789abcdefg/view"])
def test_gdrive_no_access_says_what_to_do(tmp_path, monkeypatch, link):
    _fake_gdown(monkeypatch, tmp_path, error=True)
    with pytest.raises(RuntimeError, match="Anyone with the link"):
        sources.fetch("gdrive", link, base=tmp_path / "repo")
    assert list((tmp_path / "repo" / "data" / "incoming").iterdir()) == []


def test_missing_packages(monkeypatch):
    monkeypatch.setitem(sources.CLOUD_NEEDS, "s3", "definitely_not_installed_xyz")
    assert sources.missing_packages("local") == []
    assert "definitely_not_installed_xyz" in sources.missing_packages("cloud", "s3://b/x")


def test_symlinked_folder_in_data_raw_is_used_in_place(tmp_path):
    real = make_dataset(tmp_path / "cache" / "night", n_episodes=2)
    repo = tmp_path / "repo"
    (repo / "data" / "raw").mkdir(parents=True)
    link = repo / "data" / "raw" / "night"
    link.symlink_to(real, target_is_directory=True)
    assert sources.fetch("local", str(link), base=repo) == [link]
    assert not (repo / "data" / "incoming").exists()


def test_uploaded_zip_kept_when_it_holds_no_dataset(tmp_path):
    repo = tmp_path / "repo"
    up = repo / "data" / "incoming" / "_uploads" / "bad.zip"
    up.parent.mkdir(parents=True)
    with zipfile.ZipFile(up, "w") as f:
        f.writestr("readme.txt", "no dataset here")
    with pytest.raises(FileNotFoundError):
        sources.fetch("local", str(up), name="bad", base=repo)
    assert up.exists()                                   # the user can fix the name and retry
    ds = make_dataset(tmp_path / "ds", n_episodes=2)
    good = up.with_name("good.zip")
    with zipfile.ZipFile(good, "w") as f:
        for p in ds.rglob("*"):
            f.write(p, p.relative_to(ds).as_posix())
    sources.fetch("local", str(good), name="good", base=repo)
    assert not good.exists()                             # removed only after a successful check


@pytest.mark.parametrize("src", ["--upload-pack=touch /tmp/x", "-c core.sshCommand=x"])
def test_git_options_are_not_accepted_as_repo(src):
    with pytest.raises(ValueError):
        sources.github_url(src)


def test_console_key_removed_when_fetch_fails_early(workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "KEY_DIR", tmp_path / "keys")
    (tmp_path / "keys").mkdir()
    key = tmp_path / "keys" / "gcs-1.json"
    key.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    from hackpipe.cli import main
    with pytest.raises(ValueError):                      # bad name: fails before fetch_cloud runs
        main(["fetch", "--from", "cloud", "gs://b/x", "--name", "../x"])
    assert not key.exists()


def test_partial_auto_is_not_a_failed_job(capsys):
    from hackpipe.cli import _exit_for
    _exit_for({"sources": [{"source": "a"}], "errors": [{"source": "b", "error": "bad"}]})   # no exit
    assert "skipped b" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _exit_for({"sources": [], "errors": [{"source": "b", "error": "bad"}]})


def test_changed_files_are_found_without_hashing(workspace, monkeypatch):
    import os
    import hackpipe.pipeline.preview as pv
    from hackpipe.ingest.importer import import_demos
    tmp, reg = workspace
    monkeypatch.setattr(pv, "REPO_ROOT", tmp)
    root = make_dataset(tmp / "data" / "raw" / "a", n_episodes=4)
    import_demos(str(root), reg)
    from hackpipe.quality.report import run_qc
    run_qc(str(root), reg)
    assert pv.preview(str(root), reg)["changed_files"] == 0
    f = next((root / "data").rglob("*.parquet"))
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    assert pv.preview(str(root), reg)["changed_files"] == 1
    assert [p["source"] for p in pv.new_folders(reg, tmp)] == [pv.source_key(root)]


def test_gdrive_throttled_files_are_kept_and_retry_gets_only_the_rest(tmp_path, monkeypatch):
    link = "https://drive.google.com/drive/folders/1QVcgaNooB0MHn6Dj"
    repo = tmp_path / "repo"
    g = _fake_gdown(monkeypatch, tmp_path, folder_name="rand", refuse={".parquet"})
    with pytest.raises(RuntimeError, match="could not be downloaded"):
        sources.fetch("gdrive", link, base=repo)
    part = sources.partial_dir(link, repo)
    kept = [p for p in part.rglob("*") if p.is_file() and p.name != ".listing.json"]
    n_refused = len(g.calls) - len(kept)
    assert n_refused >= 1 and kept                        # everything except the refused files is kept
    assert sources.find_roots(repo / "data" / "incoming") == []      # an unfinished download is not a dataset
    g2 = _fake_gdown(monkeypatch, tmp_path, folder_name="rand")
    roots = sources.fetch("gdrive", link, base=repo)
    assert len(g2.calls) == n_refused                     # only the missing files
    assert roots == [repo / "data" / "incoming" / "rand" / "rand"] and not part.exists()


def test_gdrive_retry_starts_over_when_folder_changed(tmp_path, monkeypatch):
    link = "https://drive.google.com/drive/folders/1QVcgaNooB0MHn6Dj"
    repo = tmp_path / "repo"
    _fake_gdown(monkeypatch, tmp_path, folder_name="rand", refuse={".parquet"})
    with pytest.raises(RuntimeError):
        sources.fetch("gdrive", link, base=repo)
    part = sources.partial_dir(link, repo)
    stale = part / "rand" / "old_file.json"
    stale.write_text("{}")
    (part / ".listing.json").write_text("[]")              # pretend the first attempt saw another listing
    g = _fake_gdown(monkeypatch, tmp_path, folder_name="rand")
    roots = sources.fetch("gdrive", link, base=repo)
    assert len(g.calls) == 3 and not (roots[0] / "old_file.json").exists()
    assert not (roots[0].parent / ".listing.json").exists()


def test_gdrive_no_access_leaves_no_partial_folder(tmp_path, monkeypatch):
    _fake_gdown(monkeypatch, tmp_path, error=True)
    repo = tmp_path / "repo"
    with pytest.raises(RuntimeError):
        sources.fetch("gdrive", "https://drive.google.com/drive/folders/1QVcgaNooB0MHn6Dj", base=repo)
    assert not (repo / "data" / "incoming" / "_partial").exists()


def test_github_branch_named_like_a_commit(tmp_path):
    src = tmp_path / "src"
    ds = make_dataset(tmp_path / "tmpds", n_episodes=2)
    files = {str(p.relative_to(ds.parent)): (p.read_text(errors="replace") if p.suffix == ".json" else "x")
             for p in ds.rglob("*") if p.is_file()}
    _git_repo(src, files)
    subprocess.run(["git", "-C", str(src), "branch", "cafe123"], check=True)
    subprocess.run(["git", "-C", str(src), "branch", "release/abc1234"], check=True)
    assert sources.fetch("github", f"file://{src}", name="b", revision="cafe123", base=tmp_path / "repo")
    with pytest.raises(subprocess.CalledProcessError):     # abc1234 is not a commit here, nor an exact ref
        sources.fetch("github", f"file://{src}", name="c", revision="abc1234", base=tmp_path / "repo")
