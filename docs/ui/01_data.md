# 01 · Data page (R1–R3) — first to build

## Goal
Adding demonstrations is one action. Import, quality check (**new demos only**) and the next dataset version
then run by themselves, with the expected time and machine shown. A human reviews and approves. ds-v2 keeps
ds-v1's validated demos; nothing is checked twice; demos that already exist are recognised.

## Tabs

### 1 · Add demonstrations
- **Pick a folder:** a select box of folders under `data/incoming/` and `data/raw/`, or a typed path. The folder
  must be a LeRobotDataset v3 (`meta/info.json`).
- **Preview before running** (read-only, instant):
  - episodes, frames, fps, cameras, task, robot type, from `info.json`;
  - how many episodes are **new**, **already registered** (same demo ID and unchanged files) or **duplicates** of
    a demo from another folder (same content and file checksums);
  - per step: CPU/GPU and expected time (import = bytes to hash; QC = frames × cameras to decode;
    version = demos).
- **Run pipeline** starts one job, `hackpipe auto --root <folder>`: import → QC of new/changed demos → draft
  version `ds-vN+1` based on the latest approved version. It ends with "ds-vN+1 is waiting for approval".
  Disabled when this machine isn't the registry writer.

### 2 · Demos
- Table: demo ID (short), source, episode, status (VALID / REVIEW / INVALID / DUPLICATE), reasons, seed,
  lighting colour temperature, outcome, frames, which version(s) contain it.
- Filters: source, status, version.
- **Exclude / Include** a demo, with a required reason and name. This is stored on the demo and applied when the
  next version is created; the card lists it under "excluded".

### 3 · Episode viewer (LeRobot visualiser style)
- Pick source + episode. Camera picker (external, left wrist, right wrist): the mp4 starts at the episode's
  `from_timestamp`.
- Plotly line charts of 16 joints, state vs action, grouped left arm / right arm / grippers.
- Side panel: the demo record (metadata, QC result, file checksums).

### 4 · Quality
- Per source: the latest QC report (`outputs/qc/<source>/qc_report.json`), with status counts, reasons table,
  redundancy, video decode summary, flagged episodes.
- "Last checked" per demo. Unchanged demos keep their earlier result; the report says how many were checked now
  vs reused.

### 5 · Versions
- A list of `ds-vN`: state, fingerprint, created, base version, counts (train / val / test), notes.
- **Diff against the base:** inherited / added / excluded counts with episode lists, test split unchanged ✓ or
  ✗ (frozen guard), leakage guard ✓.
- The dataset card (markdown).
- **Approve** (drafts only): name required, writes `approved_by` + time. Shown only to a human; agents never
  click it.

## Backend contract (built in `ui/data`)
| Function / command | Does |
|---|---|
| `hackpipe auto --root F` | import → QC (new/changed) → `dataset create --base <latest approved>`; prints a JSON summary |
| `ingest` incremental | skips registered demos with unchanged files; a file-hash cache (path, size, mtime) avoids rehashing |
| duplicate detection | same content hash + same file checksums as a demo from another source → status `DUPLICATE`, `duplicate_of` |
| `qc` incremental | checks only new/changed demos; decodes only the video files they live in; merges with earlier results |
| `dataset create --base ds-vN` | keeps the base's train/val assignment; new VALID demos are split by the fixed seed; test frozen; excluded demos removed; manifest gets `base`, `inherited`, `added`, `excluded` |
| `dataset materialize ds-vN` | multi-source versions → one merged LeRobot folder via `lerobot-edit-dataset --operation.type merge`, with the episode mapping in the manifest (runs where LeRobot is installed) |
| `dataset approve ds-vN --by NAME` | records who approved |

## Empty / error states
- No demos: the Add tab is open, with a one-line explanation.
- Folder not LeRobot v3: a clear error, nothing is written.
- Writer held elsewhere: Run, Exclude and Approve are disabled, with the reason.
- QC job failed: the job box shows failed + log; demos stay "imported"; the next step is "re-run".

## Done when
Adding a new synthetic folder imports and QCs only its demos. Re-running the same folder is a no-op. A copy of an
existing folder is flagged DUPLICATE. ds-v2 shows inherited vs added with the test set unchanged. Approve records
the name. Expected times appear after the first run. The real data reproduces ds-v1 fingerprint
`01a77eac94728a96`.
