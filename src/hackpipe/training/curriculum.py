"""Curriculum training of a VLA (SmolVLA / pi0 / pi0.5) on sub-task labels (ported from the team's
train_curriculum.py). Started by `hackpipe train` for a config with `trainer: curriculum`, which writes the run
record and registers the model; run directly only through that.

Teaches simple skills first, then harder ones, then the complete tasks:

  stage 1  reach_grasp       pick up an object with one arm          (handover sub-task 1, cube sub-tasks 1+3)
  stage 2  transport_place   carry / put an object somewhere          (handover 2+5, cube 2+4)
  stage 3  bimanual          two-arm coordination: regrasp + release  (handover 3+4)
  stage 4  full_task         the complete instruction, whole episode  (both datasets)

Each stage draws `new_skill_weight` (70%) of its samples from the skills it adds and the rest from earlier ones
(rehearsal); within each part every (dataset, skill) group gets an equal share, so 76 handover episodes aren't
drowned by 441 cube episodes. `no_curriculum: true` mixes all skills from the start (the baseline).

Differences to the team script, so runs stay reproducible and fit Colab + a 15 GB Drive:
- data and split come from the approved dataset version (its train / val demos, never the test split) and its
  sub-task labels; flagged episodes are left out; nothing is read from splits.json
- normalisation stats are pooled over the train rows of every source (equal weight)
- checkpoints every `save_freq` steps and at each stage end in LeRobot's layout (checkpoints/<step>/
  pretrained_model + `last`), with optimizer / scheduler state, so `hackpipe train --resume` continues the run;
  `keep_checkpoints` (outside this module) deletes older ones
- the sample order and the instruction rewording come from seeded generators: a resumed run draws the same
  samples it would have drawn without the interruption
- `grad_accum` micro-batches per optimizer step (effective batch = batch_size × grad_accum) and LoRA
  (`lora_rank` > 0, SmolVLA only, training/lora.py) from the team script, for GPUs that don't fit the full run
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SKILLS = ["reach_grasp", "transport_place", "bimanual", "full_task"]
DEFAULT_PLAN = {
    "stages": [
        {"name": "stage1_reach_grasp", "frac": 0.15, "skills": ["reach_grasp"]},
        {"name": "stage2_transport_place", "frac": 0.20, "skills": ["reach_grasp", "transport_place"]},
        {"name": "stage3_bimanual", "frac": 0.20, "skills": ["reach_grasp", "transport_place", "bimanual"]},
        {"name": "stage4_full_task", "frac": 0.45, "skills": SKILLS},
    ],
    "new_skill_weight": 0.7,
}
DEFAULTS = {"chunk": 20, "rate_hz": 6, "batch_size": 8, "steps": 20000, "save_freq": 1000, "log_freq": 50,
            "val_every": 1000, "val_batches": 4, "num_workers": 2, "subtask_tail": "hold", "paraphrase_prob": 0.3,
            "gripper_lag_fix": True, "bf16": False, "no_curriculum": False, "keep_stage_checkpoints": False,
            "stage_end_eval": True,      # validate + checkpoint at each stage end (off for timing runs)
            "train_sources": None,       # folder names to train on (e.g. [fr3_duo_handover_84]); None = all labelled
            "image_aug": False,          # colour jitter + small shifts of the camera images (train samples only)
            "state_noise_std": 0.0,      # Gaussian noise on the arm joints of the input state (rad)
            "grad_accum": 1,             # micro-batches per optimizer step: effective batch = batch_size × this
            "lora_rank": 0,              # > 0: LoRA adapters on SmolVLA's VLM (training/lora.py); saved merged
            "lora_alpha": None, "lora_dropout": 0.0, "lora_targets": "vlm",   # alpha default 2 × rank; vlm | all
            "lr": None, "policy_overrides": {}}
BASES = {"smolvla": "lerobot/smolvla_base", "pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_base"}


# ------------------------------------------------------------------ plan
def source_names(v: Any) -> list[str] | None:
    """train_sources as a list of folder names: `--set train_sources=a` gives a str, `[a, b]` a list."""
    if v is None:
        return None
    return [Path(x).name for x in ([v] if isinstance(v, str) else v)]


def settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train-config `args` with defaults filled in."""
    s = {**DEFAULTS, **(cfg.get("args") or {})}
    s["train_sources"] = source_names(s["train_sources"])
    s["grad_accum"] = int(s["grad_accum"])
    if s["grad_accum"] < 1 or int(s["lora_rank"]) < 0 or s["lora_targets"] not in ("vlm", "all"):
        raise ValueError("grad_accum must be ≥ 1, lora_rank ≥ 0 and lora_targets vlm or all")
    if s["lora_rank"] and cfg.get("policy", "smolvla") != "smolvla":
        raise ValueError("lora_rank is implemented for SmolVLA only")
    return s


def plan_of(cfg: dict[str, Any]) -> dict[str, Any]:
    if settings(cfg)["no_curriculum"]:
        return {"stages": [{"name": "all_skills_mixed", "frac": 1.0, "skills": SKILLS}], "new_skill_weight": 1.0}
    plan = cfg.get("curriculum") or DEFAULT_PLAN
    for s in plan["stages"]:
        bad = [k for k in s["skills"] if k not in SKILLS]
        if bad or not s["skills"] or s["frac"] <= 0:
            raise ValueError(f"curriculum stage {s.get('name')}: needs known skills (got {s['skills']}) and frac > 0")
    return plan


def stage_steps(plan: dict[str, Any], steps: int) -> list[int]:
    fr = np.array([s["frac"] for s in plan["stages"]], float)
    out = np.floor(fr / fr.sum() * steps).astype(int)
    out[-1] = steps - out[:-1].sum()
    return out.tolist()


def stage_at(step: int, steps_per_stage: list[int]) -> tuple[int, int]:
    """(stage index, steps already done in it) for a run that has done `step` steps."""
    done = 0
    for i, n in enumerate(steps_per_stage):
        if step < done + n:
            return i, step - done
        done += n
    return len(steps_per_stage) - 1, steps_per_stage[-1]


def weights(groups: np.ndarray, skills: list[str], new_skills: list[str], new_weight: float) -> np.ndarray:
    """Per-item sampling weights for a stage: the new skills share new_weight, earlier skills the rest, and each
    (dataset, skill) group gets an equal share of its part. groups: "<dataset>:<skill>" per item."""
    old = [s for s in skills if s not in new_skills]
    parts = [(new_skills, new_weight), (old, 1.0 - new_weight)] if new_skills and old else [(skills, 1.0)]
    w = np.zeros(len(groups))
    present = np.unique(groups)
    for sel, part in parts:
        gs = [g for g in present if g.split(":", 1)[1] in sel]
        for g in gs:
            m = groups == g
            w[m] = part / len(gs) / m.sum()
    return w


def group_mix(groups: list[str], plan: dict[str, Any]) -> tuple[list[dict[str, float]], list[str]]:
    """Share of each (dataset, skill) group per stage — weights() summed per group, without loading any data — and
    warnings. Raises if a stage has nothing to draw from (no train source has any of its skills)."""
    g = np.array(groups)
    have = {x.split(":", 1)[1] for x in groups}
    out, warn, seen = [], [], []
    for s in plan["stages"]:
        new = [k for k in s["skills"] if k not in seen]
        seen += new
        if not set(s["skills"]) & have:
            raise ValueError(f"stage {s['name']}: no train source has any of its skills {s['skills']}")
        missing = [k for k in new if k not in have]
        if missing:
            warn.append(f"stage {s['name']}: no train source has {missing} — it only rehearses earlier skills")
        w = weights(g, s["skills"], new, plan.get("new_skill_weight", 0.7))
        out.append({x: round(float(w[g == x].sum() / w.sum()), 3) for x in groups if w[g == x].sum() > 0})
    return out, warn


def stage_draws(w: np.ndarray, n: int, seed: int, stage: int, paraphrase_prob: float, skip: int = 0) -> np.ndarray:
    """The stage's sample codes (item * 3 + rewording: 0 none, 1 / 2 the first / second alternative), drawn from a
    generator seeded by (seed, stage). skip: samples already used before a resume."""
    rng = np.random.default_rng([seed, stage])
    items = rng.choice(len(w), size=n, p=w / w.sum())
    alt = np.where(rng.random(n) < paraphrase_prob, 1 + rng.integers(0, 2, n), 0)
    return (items * 3 + alt)[skip:]


def describe(cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """What a curriculum run will do, from the config and the version alone (no data loaded, no torch): stages,
    steps per stage, sample mix per stage, trained sources, flagged episodes left out. For dry runs and the UI."""
    from hackpipe.dataset.subtasks import SKILLS as SOURCE_SKILLS
    sub = manifest.get("subtasks")
    if not sub:
        raise RuntimeError(f"{manifest['version']} has no sub-task labels — the curriculum trainer needs them: "
                           f"hackpipe subtasks label --base {manifest['version']} --create (then approve the draft)")
    s, plan = settings(cfg), plan_of(cfg)
    train_eps = split_episodes(manifest, "train", s["train_sources"])
    groups = []
    for src in sorted(train_eps):
        info = sub["sources"].get(src)
        if info is None:
            raise RuntimeError(f"{src} is a train source of {manifest['version']} but has no sub-task labels")
        name = Path(src).name
        groups += [f"{name}:{k}" for k in dict.fromkeys(SOURCE_SKILLS[info["kind"]])] + [f"{name}:full_task"]
    steps = stage_steps(plan, int(s["steps"]))
    mix, warn = group_mix(groups, plan)
    empty = [x for x, e in train_eps.items() if not set(e) - set(sub["sources"][x]["flagged"])]
    if empty:
        raise RuntimeError(f"every train episode of {empty} is flagged: nothing to train on from it")
    return {"plan": plan, "stage_steps": steps, "sample_mix": mix, "warnings": warn, "rate_hz": s["rate_hz"],
            "chunk": s["chunk"],
            "sources": sorted(train_eps), "no_curriculum": bool(s["no_curriculum"]),
            "train_episodes": {x: len(e) - len(set(e) & set(sub["sources"][x]["flagged"]))
                               for x, e in train_eps.items()},
            "flagged_left_out": {x: sorted(set(e) & set(sub["sources"][x]["flagged"])) for x, e in train_eps.items()}}


# ------------------------------------------------------------------ instruction rewording
PARAPHRASES = [  # (regex on the original text, two alternatives using the regex groups)
    (r"^Pick up the box with the left gripper\.$", ["Grasp the box with the left hand.", "Use the left gripper to pick up the box."]),
    (r"^Bring the box to the handover position\.$", ["Carry the box to the handover point.", "Move the box to the middle for the handover."]),
    (r"^Grasp the box with the right gripper\.$", ["Take the box with the right hand.", "Close the right gripper on the box."]),
    (r"^Release the box from the left gripper\.$", ["Let go of the box with the left hand.", "Open the left gripper and release the box."]),
    (r"^Place the box on the serving tray with the right gripper\.$", ["Put the box on the tray with the right hand.", "Set the box down on the serving tray."]),
    (r"^Hand the box from the left gripper to the right gripper and place it on the serving tray\.$",
     ["Pass the box from the left hand to the right hand, then put it on the tray.",
      "Hand over the box from the left to the right gripper and set it on the serving tray."]),
    (r"^Pick up the (\w+) cube with the (left|right) gripper\.$", ["Grab the {0} cube using the {1} hand.", "Use the {1} gripper to pick up the {0} cube."]),
    (r"^Put the (\w+) cube on (.+)\.$", ["Place the {0} cube on top of {1}.", "Set the {0} cube down on {1}."]),
    (r"^Stack the (\w+) cube on top of the (\w+) cube\.$", ["Put the {0} cube on the {1} cube.", "Place the {0} cube on top of the {1} cube."]),
    (r"^Stack the (\w+) cube on (.+), then put the (\w+) cube on top\.$",
     ["Build a tower: the {0} cube on {1}, then the {2} cube on top.", "Put the {0} cube on {1} and finish with the {2} cube on top."]),
]


def reword(text: str, alt: int) -> str:
    """alt 0: unchanged; 1 / 2: the first / second alternative wording (unchanged if none matches)."""
    if alt:
        for pat, alts in PARAPHRASES:
            m = re.match(pat, text)
            if m:
                return alts[alt - 1].format(*m.groups())
    return text


# ------------------------------------------------------------------ data
def fix_gripper_lag(S: np.ndarray, A: np.ndarray, grip: list[int]) -> tuple[np.ndarray, list[int]]:
    """Move each gripper command flip to the frame where the measured gripper starts moving (the cube recordings
    log the command 1-4 frames late). Returns corrected actions and the shifts (frames)."""
    A = A.copy()
    shifts = []
    for g in grip:
        cmd = A[:, g]
        for f in np.flatnonzero(np.diff(cmd) != 0) + 1:
            new_val = cmd[f]
            direction = 1.0 if new_val > cmd[f - 1] else -1.0
            t = f
            while t > max(0, f - 8) and (S[t, g] - S[t - 1, g]) * direction > 0.02:   # slower drift = settling
                t -= 1
            if t < f:
                A[t:f, g] = new_val
                shifts.append(int(f - t))
    return A, shifts


def split_episodes(manifest: dict[str, Any], split: str, sources: list[str] | None = None) -> dict[str, list[int]]:
    """Episodes per source of a split, from the version (train / val only — the test split is never read).
    sources: only these folder names (train_sources); None = every source."""
    if split not in ("train", "val"):
        raise ValueError("training reads the train and val splits only")
    out: dict[str, list[int]] = {}
    for s, i in manifest["episodes"][split]:
        if sources is None or Path(s).name in sources:
            out.setdefault(s, []).append(int(i))
    if sources is not None:
        unknown = set(sources) - {Path(s).name for s, _ in manifest["episodes"]["train"]}
        if unknown:
            raise ValueError(f"train_sources {sorted(unknown)} are not train sources of {manifest.get('version')}")
    return out


def build_table(source: str, episodes: list[int], labels: dict[str, Any], s: dict[str, Any],
                grip: list[int]) -> dict[str, Any]:
    """Frames of one source as arrays + the sample rows (on the rate_hz grid, split episodes, not flagged)."""
    from hackpipe.common.config import dataset_config, resolve
    from hackpipe.common.lerobot_io import DatasetFolder
    cols = dataset_config()["columns"]
    ds = DatasetFolder(resolve(source))
    stride = ds.fps / s["rate_hz"]
    if abs(stride - round(stride)) > 1e-6:
        raise ValueError(f"{source}: {ds.fps:g} fps is not a multiple of rate_hz {s['rate_hz']}")
    stride = int(round(stride))
    df = ds.frames()[["index", cols["episode"], cols["frame"], cols["state"], cols["action"]]]
    df = df[df[cols["episode"]].isin(episodes)].sort_values("index").reset_index(drop=True)
    seg, eps_tab = labels["segments"], labels["episodes"].set_index("episode_index")
    flagged = set(labels["flagged"])
    S, A = np.stack(df[cols["state"]]).astype(np.float32), np.stack(df[cols["action"]]).astype(np.float32)
    ep, fi = df[cols["episode"]].to_numpy(), df[cols["frame"]].to_numpy()
    n = len(df)
    ep_end = np.zeros(n, np.int64)
    sub_end = np.zeros(n, np.int64)
    sub_text = np.empty(n, object)
    skill = np.empty(n, object)
    full_text = np.empty(n, object)
    shifts: list[int] = []
    for e, g in df.groupby(cols["episode"]).indices.items():
        g = np.sort(g)
        ep_end[g] = g[-1] + 1
        full_text[g] = eps_tab.loc[e, "full_text"] if e in eps_tab.index else ""
        for r in seg[seg.episode_index == e].itertuples():
            m = (fi[g] >= r.start_frame) & (fi[g] < r.end_frame)
            sub_end[g[m]] = g[0] + r.end_frame                 # row position just after the sub-task
            sub_text[g[m]], skill[g[m]] = r.text, r.skill
        if labels["kind"] == "cubes" and s["gripper_lag_fix"]:
            A[g], sh = fix_gripper_lag(S[g], A[g], grip)
            shifts += sh
    keep = np.isin(ep, list(set(episodes) - flagged)) & (fi % stride == 0) & (sub_text != None)  # noqa: E711
    return {"source": source, "name": Path(source).name, "kind": labels["kind"], "stride": stride, "S": S, "A": A,
            "abs_index": df["index"].to_numpy(), "ep": ep, "ep_end": ep_end, "sub_end": sub_end, "sub_text": sub_text,
            "full_text": full_text, "skill": skill, "rows": np.flatnonzero(keep),
            "flagged": sorted(flagged & set(episodes)), "gripper_shifts": shifts}


def subset(t: dict[str, Any], episodes: list[int]) -> dict[str, Any]:
    """The same table with only the sample rows of these episodes (train or val)."""
    rows = t["rows"][np.isin(t["ep"][t["rows"]], episodes)]
    return {**t, "rows": rows, "flagged": sorted(set(t["flagged"]) & set(episodes))}


def item_groups(tables: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Every sample row twice: (table, row, 0 = sub-task instruction) and (…, 1 = complete instruction)."""
    items, groups = [], []
    for ti, t in enumerate(tables):
        for r in t["rows"]:
            items += [(ti, r, 0), (ti, r, 1)]
            groups += [f"{t['name']}:{t['skill'][r]}", f"{t['name']}:full_task"]
    return np.array(items, np.int64).reshape(-1, 3), np.array(groups)


def chunk_for(t: dict[str, Any], r: int, mode: int, chunk: int, tail: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Action rows and padding of one sample. Sub-task samples hold the end pose after the sub-task (tail
    "hold": the policy learns to stop when the step is done) or mask it out ("mask"); full-task samples pad
    after the episode end."""
    idx = r + t["stride"] * np.arange(chunk)
    if mode == 0:
        last = t["sub_end"][r] - 1
        pad = np.zeros(chunk, bool) if tail == "hold" else idx > last
        text = t["sub_text"][r]
    else:
        last = t["ep_end"][r] - 1
        pad = idx > last
        text = t["full_text"][r]
    return np.minimum(idx, last), pad, text


def norm_stats(tables: list[dict[str, Any]], seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Equal-weight pooled stats over the train rows of each source (mean/std/min/max + quantiles for pi0.5)."""
    rng = np.random.default_rng(seed)
    out = {}
    empty = [t["name"] for t in tables if not len(t["rows"])]
    if empty:
        raise ValueError(f"no train samples from {empty} (all flagged or unlabelled): stats would be empty")
    n = min(len(t["rows"]) for t in tables)
    for key, arr in (("observation.state", "S"), ("action", "A")):
        pooled = np.concatenate([t[arr][rng.choice(t["rows"], n, replace=False)] for t in tables]).astype(np.float64)
        st = {"mean": pooled.mean(0), "std": np.maximum(pooled.std(0), 1e-6), "min": pooled.min(0), "max": pooled.max(0)}
        for q in (1, 10, 50, 90, 99):
            st[f"q{q:02d}"] = np.percentile(pooled, q, axis=0)
        out[key] = st
    return out


# ------------------------------------------------------------------ torch parts (Colab / GPU machine)
def augment(img, max_shift: int = 8, jitter: float = 0.2):
    """Colour jitter (brightness, contrast, saturation ± jitter) and a random shift of up to max_shift pixels
    (edge-padded) of one CxHxW image in 0..1. Uses torch's RNG, seeded per DataLoader worker."""
    import torch
    b, c, sat = (1 + (torch.rand(3) * 2 - 1) * jitter).tolist()
    x = img * b
    mean = x.mean()
    x = (x - mean) * c + mean
    gray = x.mean(0, keepdim=True)
    x = (x - gray) * sat + gray
    if max_shift:
        h, w = x.shape[-2:]
        dy, dx = torch.randint(-max_shift, max_shift + 1, (2,)).tolist()
        x = torch.nn.functional.pad(x[None], (max_shift,) * 4, mode="replicate")[0]
        x = x[:, max_shift + dy:max_shift + dy + h, max_shift + dx:max_shift + dx + w]
    return x.clamp(0, 1)


class CurriculumData:
    """torch Dataset over item codes from stage_draws (item * 3 + rewording)."""

    def __init__(self, tables: list[dict[str, Any]], items: np.ndarray, cams: list[str], s: dict[str, Any]):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from hackpipe.common.config import resolve
        self.tables, self.items, self.cams, self.s = tables, items, cams, s
        self.arm = [*range(7), *range(8, 15)]
        self.lr = [LeRobotDataset(f"local/{t['name']}", root=resolve(t["source"]), video_backend="pyav")
                   for t in tables]

    def __len__(self) -> int:
        return len(self.items) * 3

    def __getitem__(self, code: int) -> dict[str, Any]:
        import torch
        i, alt = divmod(int(code), 3)
        ti, r, mode = self.items[i]
        t = self.tables[ti]
        item = self.lr[ti][int(t["abs_index"][r])]
        idx, pad, text = chunk_for(t, r, mode, self.s["chunk"], self.s["subtask_tail"])
        out = {c: augment(item[c]) if self.s["image_aug"] else item[c] for c in self.cams}
        state = torch.from_numpy(t["S"][r].copy())
        if self.s["state_noise_std"]:
            state[self.arm] += torch.randn(len(self.arm)) * float(self.s["state_noise_std"])
        out["observation.state"] = state
        out["action"] = torch.from_numpy(t["A"][idx])
        out["action_is_pad"] = torch.from_numpy(pad)
        out["task"] = reword(text, alt)
        return out


def build_policy(cfg: dict[str, Any], s: dict[str, Any], stats: dict[str, Any], cams: list[str], state_dim: int,
                 action_dim: int, checkpoint: Path | None = None):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    base = s.get("policy.path") or BASES[cfg.get("policy", "smolvla")]
    if checkpoint and Path(base).is_absolute() and not Path(base).exists():
        base = str(checkpoint)          # fine-tune start point from another machine: resume has its own weights
    pc = PreTrainedConfig.from_pretrained(base)
    pc.input_features = {**{c: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for c in cams},
                         "observation.state": PolicyFeature(FeatureType.STATE, (state_dim,))}
    pc.output_features = {"action": PolicyFeature(FeatureType.ACTION, (action_dim,))}
    pc.chunk_size, pc.n_action_steps = s["chunk"], s["chunk"]
    pc.device, pc.push_to_hub = s.get("policy.device", "cuda"), False
    pc.scheduler_warmup_steps, pc.scheduler_decay_steps = max(1, s["steps"] // 20), s["steps"]
    if s["lr"]:
        pc.optimizer_lr = s["lr"]
    for k, v in (s["policy_overrides"] or {}).items():     # e.g. gradient_checkpointing: true for pi0
        setattr(pc, k, v)
    if s["lora_rank"]:                 # the VLM must stay trainable for its adapters; add_lora freezes its base
        pc.train_expert_only, pc.freeze_vision_encoder = False, False
    t_stats = {k: {kk: torch.tensor(vv, dtype=torch.float32) for kk, vv in v.items()} for k, v in stats.items()}
    for c in cams:
        t_stats[c] = {"mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                      "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)}
    policy = get_policy_class(pc.type).from_pretrained(str(checkpoint or base), config=pc).to(pc.device)
    if s["lora_rank"]:
        from hackpipe.training import lora
        n = lora.add_lora(lora.scopes(policy, s["lora_targets"]), int(s["lora_rank"]), s["lora_alpha"],
                          float(s["lora_dropout"]))
        _log(f"LoRA rank {s['lora_rank']} on {s['lora_targets']}: {n} adapters, "
             f"{sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6:.0f}M trainable parameters")
    pre, post = make_pre_post_processors(pc, dataset_stats=t_stats)
    optim = pc.get_optimizer_preset().build(policy.get_optim_params())
    preset = pc.get_scheduler_preset()
    sched = preset.build(optim, num_training_steps=s["steps"]) if preset else None
    return policy, pre, post, optim, sched


def validate(policy, pre, post, val_sets: dict[str, list[dict[str, Any]]], arm: list[int]) -> dict[str, Any]:
    """Error of the chunk predicted at inference time, per dataset x mode: masked L1 in normalized units and mean
    arm-joint error in rad (through the policy's own unnormalizer)."""
    import torch
    policy.eval()
    res = {}
    with torch.no_grad():
        for name, batches in val_sets.items():
            e_norm, e_rad = [], []
            for b in batches:
                raw = b["action"].clone()
                bb = pre({k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()})
                pred = policy.predict_action_chunk(bb)
                valid = ~bb["action_is_pad"]
                e_norm.append(((pred - bb["action"]).abs() * valid.unsqueeze(-1)).sum().item()
                              / max(1, valid.sum().item() * pred.shape[-1]))
                pr = post(pred).float().cpu()
                e_rad.append((pr - raw).abs()[..., arm][valid.cpu()].mean().item())
            res[name] = {"l1_norm": round(float(np.mean(e_norm)), 4), "arm_rad": round(float(np.mean(e_rad)), 4)}
    policy.train()
    return res


def _log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def save_checkpoint(out: Path, step: int, policy, pre, post, optim, sched, log: dict[str, Any]) -> Path:
    """checkpoints/<step>/pretrained_model (+ training_state.pt) and the `last` link, like lerobot-train."""
    import torch
    ck = out / "checkpoints" / f"{step:06d}"
    pm = ck / "pretrained_model"
    from hackpipe.training import lora
    if lora.adapters(policy):          # merged: a plain checkpoint; the adapters themselves for --resume
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(lora.trainable_state(policy), ck / "trainable_params.pt")
        with lora.merged(policy):
            policy.save_pretrained(pm)
    else:
        policy.save_pretrained(pm)
    pre.save_pretrained(pm)
    post.save_pretrained(pm)
    torch.save({"step": step, "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict() if sched is not None else None}, ck / "training_state.pt")
    last = out / "checkpoints" / "last"
    tmp = out / "checkpoints" / ".last.tmp"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(ck.name, target_is_directory=True)
    tmp.replace(last)
    (out / "curriculum_log.json").write_text(json.dumps(log, indent=1))
    return ck


def run(dataset_version: str, cfg: dict[str, Any], out: Path, resume: bool = False) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    from hackpipe.common.registry import Registry
    from hackpipe.dataset.subtasks import Joints, load_labels, subtasks_config
    from hackpipe.dataset.versioning import load_manifest
    s, plan = settings(cfg), plan_of(cfg)
    seed = int(cfg.get("seed", 1000))
    torch.manual_seed(seed)
    m = load_manifest(dataset_version, Registry())
    if not m.get("subtasks"):
        raise RuntimeError(f"{dataset_version} has no sub-task labels: hackpipe subtasks label --base … --create")
    lab = load_labels(m["subtasks"]["dir"])
    j = Joints(subtasks_config())
    grip, arm = [j.l_g, j.r_g], [*range(j.l_arm.start, j.l_arm.stop), *range(j.r_arm.start, j.r_arm.stop)]
    schema = m["preprocessing"]["schema"]
    cams = sorted(schema["cameras"])
    for src, sc in (m["preprocessing"].get("per_source_schema") or {}).items():
        if sorted(sc["cameras"]) != cams:
            raise RuntimeError(f"{src} has other cameras than {cams}: one policy needs the same cameras")

    _log("loading tables …")
    tr_eps, va_eps = split_episodes(m, "train", s["train_sources"]), split_episodes(m, "val", s["train_sources"])
    train_t, val_t = [], []
    for src in sorted(set(tr_eps) | set(va_eps)):           # each source read once, its train + val episodes only
        t = build_table(src, tr_eps.get(src, []) + va_eps.get(src, []),
                        {**lab[src], "kind": lab["index"]["sources"][src]["kind"],
                         "flagged": lab["index"]["sources"][src]["flagged"]}, s, grip)
        for eps, dest in ((tr_eps.get(src), train_t), (va_eps.get(src), val_t)):
            if eps:
                dest.append(subset(t, eps))
    for si, st_ in enumerate(plan["stages"]):                # fail before loading the policy, not mid-run
        group_mix([g for t in train_t for g in {f"{t['name']}:{k}" for k in t["skill"][t["rows"]]}]
                  + [f"{t['name']}:full_task" for t in train_t], {**plan, "stages": plan["stages"][:si + 1]})
    for t in train_t:
        _log(f"  {t['name']}: {len(t['rows'])} train samples @ {s['rate_hz']} Hz, flagged left out {t['flagged']}"
             + (f", gripper flips moved earlier: {len(t['gripper_shifts'])}" if t["gripper_shifts"] else ""))
    items, groups = item_groups(train_t)
    v_items, _ = item_groups(val_t)
    stats_file = out / "norm_stats.npz"
    if resume:
        z = np.load(stats_file)                   # the run's own stats, so the resumed model sees the same inputs
        stats: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            k, kk = key.split("|")
            stats.setdefault(k, {})[kk] = z[key]
    else:
        stats = norm_stats(train_t, seed)
        np.savez(stats_file, **{f"{k}|{kk}": vv for k, v in stats.items() for kk, vv in v.items()})
    last = out / "checkpoints" / "last"
    start = 0
    ckpt = None
    if resume:
        ckpt = last.resolve() / "pretrained_model"
        start = int(torch.load(last / "training_state.pt", weights_only=False)["step"])
    sd = int(schema["state"]["shape"][0])
    ad = int(schema["action"]["shape"][0])
    # LoRA: the saved model is merged, so a resumed run starts from the base again and loads its adapters. If the base
    # isn't on this machine (a fine-tune start point from another one), it continues from the merged model with fresh
    # adapters instead: the same model, only the adapters' optimizer moments start over.
    base = s.get("policy.path") or BASES[cfg.get("policy", "smolvla")]
    lora_resume = bool(resume and s["lora_rank"]) and not (Path(base).is_absolute() and not Path(base).exists())
    if resume and s["lora_rank"] and not lora_resume:
        _log(f"LoRA resume: {base} is not on this machine, continuing from the merged checkpoint with fresh adapters")
    policy, pre, post, optim, sched = build_policy(cfg, s, stats, cams, sd, ad, None if lora_resume else ckpt)
    if lora_resume:
        bad = policy.load_state_dict(torch.load(last / "trainable_params.pt", weights_only=False),
                                     strict=False).unexpected_keys
        if bad:
            raise RuntimeError(f"trainable_params.pt does not match the model: {bad[:5]}")
    if resume:
        state = torch.load(last / "training_state.pt", weights_only=False)
        if lora_resume or not s["lora_rank"]:
            optim.load_state_dict(state["optimizer"])
        if sched is not None and state["scheduler"]:
            sched.load_state_dict(state["scheduler"])
        _log(f"resumed at step {start}")
    data = CurriculumData(train_t, items, cams, s)
    vdata = CurriculumData(val_t, v_items, cams, {**s, "paraphrase_prob": 0.0, "image_aug": False,
                                                  "state_noise_std": 0.0})
    rng = np.random.default_rng([seed, 99])
    vsets = {}
    for ti, t in enumerate(val_t):
        for mode, label in ((0, "sub"), (1, "full")):
            idx = np.flatnonzero((v_items[:, 0] == ti) & (v_items[:, 2] == mode))
            pick = rng.choice(idx, min(len(idx), s["batch_size"] * s["val_batches"]), replace=False) * 3
            vsets[f"{t['name']}/{label}"] = list(DataLoader(Subset(vdata, pick.tolist()),
                                                            batch_size=s["batch_size"], num_workers=0))
    per_stage = stage_steps(plan, s["steps"])
    log_path = out / "curriculum_log.json"
    log = json.loads(log_path.read_text()) if resume and log_path.exists() else \
        {"dataset_version": dataset_version, "plan": plan, "stage_steps": per_stage, "stages": [], "val": []}
    first_stage, done_in_stage = stage_at(start, per_stage)
    step, t0 = start, time.time()
    seen = [k for st in plan["stages"][:first_stage] for k in st["skills"]]
    for si in range(first_stage, len(plan["stages"])):
        stage, n_steps = plan["stages"][si], per_stage[si]
        new = [k for k in stage["skills"] if k not in seen]
        seen += new
        w = weights(groups, stage["skills"], new, plan.get("new_skill_weight", 0.7))
        skip = done_in_stage if si == first_stage else 0
        per_step = s["batch_size"] * s["grad_accum"]
        codes = stage_draws(w, n_steps * per_step, seed, si, s["paraphrase_prob"], skip * per_step)
        mix = {g: round(float(w[groups == g].sum()), 3) for g in np.unique(groups) if w[groups == g].sum() > 0}
        _log(f"=== {stage['name']}: {n_steps} steps (from {skip}), new skills {new}, sample mix {mix}")
        if not any(x["name"] == stage["name"] for x in log["stages"]):
            log["stages"].append({"name": stage["name"], "steps": n_steps, "new_skills": new, "sample_mix": mix})
        dl = DataLoader(data, batch_size=s["batch_size"], sampler=codes.tolist(), num_workers=s["num_workers"],
                        pin_memory=True, drop_last=True)
        policy.train()
        optim.zero_grad(set_to_none=True)
        t_data = t_upd = time.time()
        micro_loss = 0.0
        for micro, batch in enumerate(dl, 1):
            data_s = time.time() - t_data
            batch = pre(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(s["bf16"])):
                loss, _ = policy.forward(batch)
            (loss / s["grad_accum"]).backward()
            micro_loss += loss.item() / s["grad_accum"]      # the step's loss: mean over its micro-batches
            if micro % s["grad_accum"]:
                t_data = time.time()
                continue
            step += 1
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()
            if step % s["log_freq"] == 0:       # same fields as lerobot-train, so the Train page reads it
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                _log(f"INFO stage:{si + 1} step:{step} loss:{micro_loss:.4f} grdn:{float(gn):.3f} "
                     f"lr:{optim.param_groups[0]['lr']:.1e} updt_s:{time.time() - t_upd:.3f} data_s:{data_s:.3f} "
                     f"mem_gb:{mem:.2f}")
            stage_end = step == sum(per_stage[:si + 1])
            if step % s["val_every"] == 0 or stage_end and s["stage_end_eval"]:
                v = validate(policy, pre, post, vsets, arm)
                log["val"].append({"step": step, "stage": stage["name"], "end_of_stage": stage_end, **v})
                _log(("VAL end of " + stage["name"] + ": " if stage_end else "VAL ")
                     + " | ".join(f"{k} l1={x['l1_norm']} arm={x['arm_rad']}rad" for k, x in v.items()))
            if step % s["save_freq"] == 0 or stage_end and (s["stage_end_eval"] or step == s["steps"]):
                ck = save_checkpoint(out, step, policy, pre, post, optim, sched, log)
                if stage_end and s["keep_stage_checkpoints"]:
                    import shutil
                    shutil.copytree(ck / "pretrained_model", out / "stages" / stage["name"], dirs_exist_ok=True)
            t_data = t_upd = time.time()
            micro_loss = 0.0
    _log(f"done: {step} steps in {(time.time() - t0) / 60:.1f} min")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="curriculum training (started by `hackpipe train`)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True, help="resolved train config (JSON) written by hackpipe train")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "train_config.json"
    if a.resume:
        cfg = json.loads(saved.read_text())        # the run's own config, never a new one
    else:
        cfg = json.loads(Path(a.config).read_text())
        saved.write_text(json.dumps(cfg, indent=2))
    run(a.dataset, cfg, out, resume=a.resume)


if __name__ == "__main__":
    main()
