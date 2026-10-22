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


def project_config() -> dict[str, Any]:
    """configs/project.yaml; HACKPIPE_OUTPUTS_DIR / HACKPIPE_REGISTRY override paths (used by tests)."""
    cfg = load_yaml(CONFIG_DIR / "project.yaml")
    if os.environ.get("HACKPIPE_OUTPUTS_DIR"):
        cfg["project"]["outputs_dir"] = os.environ["HACKPIPE_OUTPUTS_DIR"]
    if os.environ.get("HACKPIPE_REGISTRY"):
        cfg["project"]["registry_path"] = os.environ["HACKPIPE_REGISTRY"]
    return cfg


def dataset_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "dataset.yaml")


def gates_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "gates.yaml")


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
