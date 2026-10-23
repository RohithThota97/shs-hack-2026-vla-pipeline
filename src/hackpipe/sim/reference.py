"""The reference file the Isaac Sim scripts read (team's `eval_isaac.py`, our `scripts/isaac/record_demos.py`).

Same fields as the team's `export_eval_assets.py` (task, initial state, one demo's 30 fps actions + states, hold
cut, gripper contract and drive, camera parameters, `val_lighting`), built from an approved dataset version instead
of their own split.json, with two differences:
- `val_lighting` (the lighting of each evaluation episode) comes from the **held-out test source** (lighting10),
  never from episodes the model was trained or validated on. The team's file used their val episodes, which are in
  our train split.
- `record_episodes` lists the version's train / val episodes of the main source with their own lighting and seed,
  for re-rendering them in the evaluation scene (`record_demos.py --mode render|dart|snapshot`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hackpipe.common.config import dataset_config, project_config, resolve, source_key
from hackpipe.common.ids import demo_id, now_iso
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.dataset.versioning import load_manifest


def _light(extra: dict[str, Any]) -> dict[str, Any]:
    return (extra.get("randomization") or {}).get("applied", {}).get("light") or {}


def build_reference(dataset_version: str, suite: dict[str, Any], registry: Registry | None = None,
                    demo_episode: int | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    m = load_manifest(dataset_version, reg)
    if m["status"] != "approved":
        raise RuntimeError(f"{dataset_version} is '{m['status']}': build the sim reference from an approved version")
    main = source_key(resolve(project_config()["dataset"]["root"]))
    cols = dataset_config()["columns"]
    ds = DatasetFolder(resolve(main))
    extra = ds.episode_extras()
    conv = json.loads((ds.root / "meta" / "fr3_conversion.json").read_text())
    split_of = {(s, int(i)): k for k in ("train", "val") for s, i in m["episodes"][k]}
    mine = sorted(i for (s, i) in split_of if s == main)
    if not mine:
        raise RuntimeError(f"{dataset_version} has no train/val episodes of the main source {main}")
    train_eps = sorted(i for (s, i), k in split_of.items() if s == main and k == "train")
    demo = int(demo_episode if demo_episode is not None else train_eps[0])
    if split_of.get((main, demo)) is None:
        raise ValueError(f"demo episode {demo} is not a train/val episode of {main} in {dataset_version}")
    df = ds.frames()
    ep = df[df[cols["episode"]] == demo].sort_values(cols["frame"])
    S, A = np.stack(ep[cols["state"]]), np.stack(ep[cols["action"]])

    sim = suite.get("sim") or {}
    test_src = sim.get("lighting_source")
    if not test_src:
        raise ValueError("suite has no sim.lighting_source (the held-out test source whose lighting the eval uses)")
    test_key = source_key(resolve(test_src))
    if test_key not in (m.get("test_sources") or []):
        raise ValueError(f"{test_key} is not a held-out test source of {dataset_version} "
                         f"({m.get('test_sources')}): the eval lighting must come from unseen episodes")
    t_extra = DatasetFolder(resolve(test_key)).episode_extras()
    test_eps = sorted(int(i) for s, i in m["episodes"]["test"] if s == test_key)
    return {
        "task": conv["dataset_cfg"]["task"],
        "fps": int(ds.fps),
        "state_names": ds.features[cols["state"]].get("names"),
        "initial_state": S[0].round(6).tolist(),
        "demo_episode": demo,
        "demo_actions": A.round(6).tolist(),
        "demo_states": S.round(6).tolist(),
        "hold_cut_frame": int(extra[demo]["hold_cut"]["dropped_frames"][0]),
        "demo_final_box_pose": extra[demo]["final_box_pose"],
        "gripper_contract": conv["dataset_cfg"]["gripper_contract"],
        "gripper_drive_recorded": extra[demo]["gripper_drive"],
        "camera_params": conv["recorder"]["camera_params"],
        "val_lighting": [{"episode_index": i, **_light(t_extra.get(i, {}))} for i in test_eps],
        "record_episodes": [{"source": main, "episode_index": i, "split": split_of[(main, i)],
                             "demo_id": demo_id(main, i), "seed": extra.get(i, {}).get("seed"),
                             "name": extra.get(i, {}).get("name"), "light": _light(extra.get(i, {}))}
                            for i in mine],
        "hackpipe": {"dataset_version": dataset_version, "fingerprint": m["fingerprint"],
                     "suite": suite.get("version"), "lighting_source": test_key, "main_source": main,
                     "created_at": now_iso(), "note": "val_lighting = held-out test source (unseen lighting)"},
    }


def write_reference(dataset_version: str, suite_path: str | Path, out: str | Path,
                    registry: Registry | None = None, demo_episode: int | None = None) -> Path:
    from hackpipe.common.config import load_yaml
    ref = build_reference(dataset_version, load_yaml(suite_path), registry, demo_episode)
    p = resolve(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ref))
    return p
