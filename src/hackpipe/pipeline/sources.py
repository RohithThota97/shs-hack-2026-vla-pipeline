"""Bring demonstrations in from wherever they are: a folder or zip on this machine, a zip uploaded in the
browser, a Hugging Face dataset repo, a GitHub repo, a Google Drive link or a cloud bucket (S3 / GCS / Azure).

Everything lands in data/incoming/<name> (folders on this machine are linked, not copied, unless asked), so the
demo IDs are the same on every machine with the same layout. Credentials come from the environment
(HF_TOKEN, GITHUB_TOKEN, AWS_*, GOOGLE_APPLICATION_CREDENTIALS, AZURE_STORAGE_CONNECTION_STRING): the console
passes what the user typed to the fetch job as environment variables only — never in the command line, the
registry or a log.
"""
from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from hackpipe.common.config import REPO_ROOT

INCOMING = "data/incoming"
SEARCH_DIRS = (INCOMING, "data/raw")      # where new dataset folders are looked for
KINDS = ("local", "hf", "github", "gdrive", "cloud")
KEY_DIR = Path.home() / ".cache" / "hackpipe" / "keys"
MAX_DEPTH = 4          # how deep below a folder to look for meta/info.json
_SKIP = {".git", "node_modules", "__pycache__", ".cache", "_partial"}   # _partial: unfinished Drive downloads


def find_roots(top: str | Path, max_depth: int = MAX_DEPTH) -> list[Path]:
    """LeRobot dataset folders (with meta/info.json) at or below top; doesn't look inside a dataset folder."""
    top = Path(top)
    out: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if (d / "meta" / "info.json").is_file():
            out.append(d)
            return
        if depth >= max_depth:
            return
        try:
            kids = sorted(p for p in d.iterdir() if p.is_dir() and p.name not in _SKIP)
        except OSError:
            return
        for k in kids:
            walk(k, depth + 1)

    if top.is_dir():
        walk(top, 0)
    return out


def dataset_name(path: str | Path) -> str:
    """Folder-safe name from a path, zip, repo ID or URL: fr3_duo_rand500/fr3_duo_rand500 → fr3_duo_rand500."""
    s = re.sub(r"[?#].*$", "", str(path)).rstrip("/")          # URL query (?usp=drive_link) is not a name
    s = re.sub(r"(\.git|\.zip|\.tar\.gz|\.tgz)$", "", s)
    parts = [p for p in re.split(r"[/\\:]+", s) if p and p not in (".", "..")]
    name = parts[-1] if parts else "dataset"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "dataset"


def _dest(name: str, base: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or ".." in name:
        raise ValueError(f"name {name!r}: use letters, digits, '.', '_' or '-' only (one folder, no '/' or '..')")
    d = base / INCOMING / name
    if d.exists() or d.is_symlink():
        raise FileExistsError(f"{d} already exists — pick another name or remove it")
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


def _check(dest: Path) -> list[Path]:
    roots = find_roots(dest)
    if not roots:
        raise FileNotFoundError(f"no LeRobot dataset (meta/info.json) found in what was fetched → {dest}")
    return roots


def extract_zip(zip_path: str | Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():                  # no paths outside dest
            if Path(m).is_absolute() or ".." in Path(m).parts:
                raise ValueError(f"unsafe path in zip: {m}")
        z.extractall(dest)
    for junk in dest.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)


# ---------- one fetcher per kind; each fills dest ----------
def local_path(src: str) -> Path:
    """~ expanded, relative to the current directory (the console runs from the repo root)."""
    return Path(os.path.abspath(Path(src).expanduser()))


def fetch_local(src: str, dest: Path, copy: bool = False) -> None:
    p = local_path(src)
    if not p.exists():
        raise FileNotFoundError(src)
    if p.is_file() and p.suffix == ".zip":
        extract_zip(p, dest)
    elif copy:
        shutil.copytree(p, dest)
    else:
        dest.symlink_to(p.resolve(), target_is_directory=True)


def fetch_hf(repo_id: str, dest: Path, revision: str | None = None, subdir: str = "") -> None:
    from huggingface_hub import snapshot_download     # [hub] extra
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision or None, local_dir=str(dest),
                      token=os.environ.get("HF_TOKEN") or None,
                      allow_patterns=[f"{subdir.strip('/')}/**"] if subdir else None)


def _git_env() -> dict[str, str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:   # header via env, so the token is not in argv or .git/config
        basic = base64.b64encode(f"x-access-token:{tok}".encode()).decode()
        env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"})
    return env


def github_url(src: str) -> str:
    """owner/repo, github.com/owner/repo or a full URL → https clone URL."""
    s = src.strip().removesuffix("/")
    if s.startswith("-"):
        raise ValueError(f"not a repository: {src!r}")
    if re.fullmatch(r"[\w.-]+/[\w.-]+", s):
        return f"https://github.com/{s.removesuffix('.git')}.git"
    if s.startswith("github.com/"):
        return f"https://{s}"
    return s


def _uses_lfs(repo: Path) -> bool:
    return any("filter=lfs" in f.read_text(errors="replace") for f in repo.rglob(".gitattributes"))


def fetch_github(src: str, dest: Path, revision: str | None = None) -> None:
    env = _git_env()
    git = ["git", "-C", str(dest)]
    if revision and revision.startswith("-"):
        raise ValueError(f"not a branch / tag / commit: {revision!r}")
    url = github_url(src)
    hexrev = bool(revision) and re.fullmatch(r"[0-9a-f]{7,40}", revision) is not None
    is_ref = True
    if hexrev:              # a branch / tag named like a commit? ls-remote matches ref name endings: compare exactly
        refs = subprocess.run(["git", "ls-remote", "--heads", "--tags", "--", url], check=True, env=env,
                              capture_output=True, text=True).stdout.split()
        is_ref = any(r in (f"refs/heads/{revision}", f"refs/tags/{revision}") for r in refs)
    if revision and not is_ref and re.fullmatch(r"[0-9a-f]{40}", revision):   # full commit: shallow fetch by SHA
        dest.mkdir(parents=True)
        subprocess.run([*git, "init", "-q"], check=True, env=env)
        subprocess.run([*git, "remote", "add", "--", "origin", url], check=True, env=env)
        subprocess.run([*git, "fetch", "--depth", "1", "--", "origin", revision], check=True, env=env)
        subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True, env=env)
    elif revision and not is_ref and re.fullmatch(r"[0-9a-f]{7,39}", revision):
        # short commit (as GitHub shows it): servers only fetch full SHAs. History without file contents
        # (--filter=blob:none, no LFS smudge), then only that commit's files are downloaded on checkout.
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--", url, str(dest)],
                       check=True, env={**env, "GIT_LFS_SKIP_SMUDGE": "1"})
        subprocess.run([*git, "checkout", "-q", revision], check=True, env={**env, "GIT_LFS_SKIP_SMUDGE": "1"})
    else:
        subprocess.run(["git", "clone", "--depth", "1", *(["--branch", revision] if revision else []), "--",
                        url, str(dest)], check=True, env=env)
    if _uses_lfs(dest):                       # datasets keep videos/parquet in LFS
        if not shutil.which("git-lfs"):
            raise RuntimeError("this repo stores files in Git LFS: install git-lfs (brew / apt install git-lfs), "
                               "otherwise only small pointer files would be imported")
        subprocess.run([*git, "lfs", "pull"], check=True, env=env)
    shutil.rmtree(dest / ".git", ignore_errors=True)


def gdrive_id(src: str) -> tuple[str, str]:
    """A Drive share link or ID → ("file" | "folder", id)."""
    m = re.search(r"/folders/([\w-]+)", src)
    if m:
        return "folder", m.group(1)
    m = re.search(r"(?:/d/|[?&]id=)([\w-]+)", src)
    if m:
        return "file", m.group(1)
    if re.fullmatch(r"[\w-]{20,}", src.strip()):
        return "file", src.strip()
    raise ValueError("not a Google Drive link or ID")


NO_ACCESS = ("Google Drive refused access to this link. Either share it as 'Anyone with the link' (Share → "
             "General access), or — better for team-only data — install Google Drive for desktop and use "
             "'This machine' with the folder's path (~/Library/CloudStorage/GoogleDrive-…/My Drive/…).")


def _rmdir_if_empty(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass


def partial_dir(src: str, base: Path = REPO_ROOT) -> Path:
    """Where an unfinished Drive download is kept, so running the same fetch again only gets the missing files."""
    return base / INCOMING / "_partial" / f"gdrive-{gdrive_id(src)[1]}"


def _gdrive_folder(gdown, fid: str, part: Path) -> None:
    """List the folder once, then download only files that aren't on disk yet. Google refuses some files after
    many downloads of a shared link ('too many accesses'); those are counted, the rest is kept for a retry."""
    from gdown.exceptions import DownloadError
    import json
    files = gdown.download_folder(id=fid, output=str(part) + "/", skip_download=True, quiet=True)
    if not files:
        raise RuntimeError(NO_ACCESS)
    listing = sorted([f.id, f.path] for f in files)
    mark = part / ".listing.json"                    # what the kept files belong to
    try:
        before = json.loads(mark.read_text())
    except (OSError, ValueError):
        before = None
    if before is not None and before != listing:     # the folder changed on Drive since the first attempt
        print("Drive folder changed since the last attempt: starting over", flush=True)
        for p in part.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    mark.write_text(json.dumps(listing))
    missing = [f for f in files if not Path(f.local_path).is_file()]
    print(f"Drive folder: {len(files)} files, {len(files) - len(missing)} already here, {len(missing)} to download",
          flush=True)
    failed = []
    for f in missing:
        Path(f.local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(id=f.id, output=f.local_path, quiet=False, resume=True)
        except DownloadError as e:
            failed.append(f.path)
            print(f"failed: {f.path}: {str(e).strip().splitlines()[0]}", flush=True)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(files)} files could not be downloaded. Usually Google's limit on downloads of "
            "shared links ('too many accesses') — or those files aren't shared themselves (check their sharing). "
            f"The other {len(files) - len(failed)} are kept in {part}: run the same fetch again later (only the "
            "missing files are fetched), or share the dataset as one zip instead.")


def fetch_gdrive(src: str, dest: Path, base: Path = REPO_ROOT) -> str:
    """Shared link ('anyone with the link'). A zip is extracted. Returns the name on Drive (folder or zip
    without .zip). A mounted Drive (Colab, Drive for desktop) is a folder on this machine — use kind 'local'.
    Downloads into partial_dir(src) first and moves it to dest when complete."""
    try:
        import gdown
        from gdown.exceptions import DownloadError, FileURLRetrievalError
    except ImportError as e:
        raise RuntimeError("Google Drive links need gdown: pip install -e '.[sources]'") from e
    kind, fid = gdrive_id(src)
    part = partial_dir(src, base)
    part.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "folder":
            _gdrive_folder(gdown, fid, part)
        else:
            out = gdown.download(id=fid, output=str(part) + "/", quiet=False, resume=True)
    except BaseException as e:
        if not any(p.name != ".listing.json" for p in part.rglob("*")):    # nothing downloaded: leave nothing
            shutil.rmtree(part, ignore_errors=True)
            _rmdir_if_empty(part.parent)
        # a file without access → FileURLRetrievalError; a folder → "status code 401"
        if isinstance(e, FileURLRetrievalError) or (isinstance(e, DownloadError)
                                                    and re.search(r"status code 40[13]", str(e))):
            raise RuntimeError(NO_ACCESS) from e
        raise
    if kind == "folder":
        (part / ".listing.json").unlink(missing_ok=True)
        kids = [p for p in part.iterdir() if p.is_dir()]        # gdown puts the files in <part>/<folder name>/
        name = kids[0].name if len(kids) == 1 else dataset_name(src)
        part.rename(dest)
        _rmdir_if_empty(part.parent)
        return name
    if not out:
        raise RuntimeError(NO_ACCESS)
    dest.mkdir(parents=True)
    if out.endswith(".zip"):
        extract_zip(out, dest)
        os.remove(out)
    else:
        shutil.move(out, dest / Path(out).name)
    shutil.rmtree(part, ignore_errors=True)
    _rmdir_if_empty(part.parent)
    return dataset_name(out)


def fetch_cloud(url: str, dest: Path) -> None:
    """s3://bucket/prefix (also R2 / MinIO via AWS_ENDPOINT_URL), gs://bucket/prefix, az://container/prefix."""
    try:
        import fsspec
    except ImportError as e:
        raise RuntimeError("Cloud buckets need fsspec + s3fs / gcsfs / adlfs: pip install -e '.[cloud]'") from e
    proto = url.split("://", 1)[0]
    opts: dict = {}
    if proto == "s3" and os.environ.get("AWS_ENDPOINT_URL"):
        opts["client_kwargs"] = {"endpoint_url": os.environ["AWS_ENDPOINT_URL"]}
    if proto in ("gs", "gcs") and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        opts["token"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    if proto in ("az", "abfs") and os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        opts["connection_string"] = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    try:
        fs = fsspec.filesystem(proto, **opts)
        path = url.split("://", 1)[1].rstrip("/")
        if path.endswith(".zip"):
            dest.mkdir(parents=True)
            tmp = dest / "_download.zip"
            fs.get(path, str(tmp))
            extract_zip(tmp, dest)
            tmp.unlink()
        else:
            fs.get(path + "/", str(dest), recursive=True)
    finally:
        drop_console_key()


def drop_console_key() -> None:
    """A GCS key uploaded in the console is used once: remove it (also when the fetch failed earlier)."""
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if key and Path(key).parent == KEY_DIR:
        Path(key).unlink(missing_ok=True)


NEEDS = {"hf": ["huggingface_hub"], "gdrive": ["gdown"], "cloud": ["fsspec"]}
CLOUD_NEEDS = {"s3": "s3fs", "gs": "gcsfs", "az": "adlfs"}


def missing_packages(kind: str, url: str = "") -> list[str]:
    """Python packages this kind of fetch needs that aren't installed here."""
    from importlib.util import find_spec
    need = list(NEEDS.get(kind, []))
    if kind == "cloud":
        need.append(CLOUD_NEEDS.get(url.split("://", 1)[0], ""))
    return [m for m in need if m and find_spec(m) is None]


def fetch(kind: str, src: str, name: str | None = None, revision: str | None = None, subdir: str = "",
          copy: bool = False, base: Path = REPO_ROOT) -> list[Path]:
    """Fetch src into data/incoming/<name> and return the dataset folders found there.
    A folder already under data/incoming or data/raw is used in place (nothing is copied or linked); anything
    else gets a data/incoming/<name> path, so demo IDs match on every machine."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if kind == "local":
        p = local_path(src)              # not resolved: data/raw/<name> is usually a symlink to the real files
        homes = [Path(os.path.abspath(base / d)) for d in SEARCH_DIRS]
        if p.is_dir() and any(p == h or h in p.parents for h in homes):
            return _check(p)
    auto_name = kind == "gdrive" and not name           # the link has no name in it: use the one on Drive
    dest = _dest(name or (f"gdrive-{gdrive_id(src)[1][:10]}" if auto_name else dataset_name(src)), base)
    try:
        got = {"local": lambda: fetch_local(src, dest, copy),
               "hf": lambda: fetch_hf(src, dest, revision, subdir),
               "github": lambda: fetch_github(src, dest, revision),
               "gdrive": lambda: fetch_gdrive(src, dest, base),
               "cloud": lambda: fetch_cloud(src, dest)}[kind]()
        roots = _check(dest)
        up = local_path(src) if kind == "local" else None
        if up is not None and up.suffix == ".zip" and up.parent.name == "_uploads":
            up.unlink()                     # uploaded from the console: the checked, extracted copy is enough
        if auto_name and got and got != dest.name and not (dest.parent / got).exists():
            final = _dest(got, base)
            dest.rename(final)
            roots = _check(final)
        return roots
    except BaseException:
        if dest.is_symlink():
            dest.unlink()
        else:
            shutil.rmtree(dest, ignore_errors=True)       # never leave a half-fetched folder behind
        raise
