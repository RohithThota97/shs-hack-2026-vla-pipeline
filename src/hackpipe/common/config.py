"""Load YAML configs from the repo's configs/ folder."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_dir() -> Path:
    """configs/, or HACKPIPE_CONFIG_DIR (tests use synthetic-data configs)."""
    return Path(os.environ["HACKPIPE_CONFIG_DIR"]) if os.environ.get("HACKPIPE_CONFIG_DIR") else CONFIG_DIR


def project_config() -> dict[str, Any]:
    """configs/project.yaml; HACKPIPE_OUTPUTS_DIR / HACKPIPE_REGISTRY override paths (used by tests)."""
    cfg = load_yaml(config_dir() / "project.yaml")
    if os.environ.get("HACKPIPE_OUTPUTS_DIR"):
        cfg["project"]["outputs_dir"] = os.environ["HACKPIPE_OUTPUTS_DIR"]
    if os.environ.get("HACKPIPE_REGISTRY"):
        cfg["project"]["registry_path"] = os.environ["HACKPIPE_REGISTRY"]
    return cfg


def dataset_config() -> dict[str, Any]:
    return load_yaml(config_dir() / "dataset.yaml")


def gates_config() -> dict[str, Any]:
    return load_yaml(config_dir() / "gates.yaml")


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def source_key(root: str | Path) -> str:
    """Stable name for a dataset folder: repo-relative inside the repo (same on laptop and Colab), else absolute.
    Symlinks are not followed, so data/raw/<name> keeps its name wherever the files really live."""
    p = Path(os.path.normpath(os.path.abspath(resolve(root))))
    try:
        return p.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()
