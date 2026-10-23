# 01 · Data page (R1–R3) — first to build

## Goal
Adding demonstrations is one action. Import, quality check (**new demos only**) and the next dataset version
then run by themselves, with the expected time and machine shown. A human reviews and approves. ds-v2 keeps
ds-v1's validated demos; nothing is checked twice; demos that already exist are recognised.

## Tabs

### 1 · Add demonstrations
- **New datasets found automatically:** every LeRobot v3 folder at any depth under `data/incoming/` and
  `data/raw/` with demos the registry hasn't seen, **plus** datasets in the watch folders
  (`dataset.watch_dirs` in `configs/project.yaml`: the organisers' download `data/C2_Teach-the-robot_VLA`,
  `/content/drive/MyDrive/shs-hack-2026/data` on Colab, Google Drive for desktop on macOS) that aren't linked yet.
  Copies are recognised by `meta/info.json` + `meta/stats.json`, so an already linked dataset or the zip of a folder
  that is there is listed once. **Run pipeline on all new datasets** = `hackpipe auto --all`: links new ones as
  `data/raw/<name>` (zips unzipped to `data/cache/<name>` first; names from `dataset.pinned_names`, else the folder
  name — same as the Colab notebook), then import → QC → one draft version. `hackpipe watch` lists what's waiting.
- **Check one dataset:** a watched dataset shows episodes / frames / fps and **Link as data/raw/<name> and run
  pipeline** (`hackpipe auto --link PATH --name NAME`).
- **Bring in a dataset from elsewhere** (`hackpipe fetch --from KIND SRC --name N --auto`, a background job) into
  `data/incoming/<name>`, then the pipeline:
  | From | Source | Keys (password fields, passed to the job as env vars only — never saved or logged) |
  |---|---|---|
  | This machine | folder (linked, or copied) or `.zip`; a mounted Drive path works here | — |
  | Upload a zip | browser upload, up to 5 GB (Streamlit holds it in memory; on Colab prefer the Drive link / this machine) | — |
  | Hugging Face | dataset repo ID, revision, subfolder | `HF_TOKEN` |
  | GitHub | `owner/repo` or URL, branch; Git LFS pulled if installed | `GITHUB_TOKEN` (sent as a header) |
  | Google Drive link | link shared as "Anyone with the link" to a zip or folder (`gdown`); named after the file / folder on Drive. Google limits downloads of shared links, so a folder may stop partway: the finished files stay in `data/incoming/_partial/` and the same fetch later gets only the missing ones. A zip (one file) avoids most of it. Private Drive → Drive for desktop + "This machine" | — |
  | Cloud bucket | `s3://` (also R2 / MinIO), `gs://`, `az://`, folder or `.zip` | AWS keys + endpoint, GCS service-account JSON, Azure connection string |
  Empty key fields fall back to this machine's environment. Extras: `pip install -e ".[sources]"` (HF, Drive),
  `".[cloud]"` (buckets).
- **Check one folder (a typed path):** a select box of folders under `data/incoming/` and `data/raw/`, or a typed path. The folder
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

## Sources with different fps (subtask data)
`versions.allow_mixed_schemas: true` (configs/dataset.yaml) lets a version hold sources whose fps / schema differ —
here the 6 fps cube-stacking subtask `fr3_duo_rand500` next to the 30 fps handover data. The manifest gets
`mixed_schemas: true` and a per-source schema (also in the card).
- **Training uses the sources with the main dataset's schema** (`dataset.root` in project.yaml = handover); the
  others are **data only** (`trainable_sources()`): not trained, not merged, not counted in the step budget. The
  Versions tab, the card and the Train page say which. LeRobot merges only equal fps, so this is the only way a
  mixed version trains.
- Later versions (based on an approved mixed one) keep training on the main-schema sources; new main-task data
  is merged with `dataset materialize` as before, the subtask stays out.
- Seeds of a data-only source aren't compared with the test split's seeds (another task, own numbering).
- The flag is outside `splits`/`preprocessing`, so single-source fingerprints (ds-v1 `01a77eac94728a96`) don't change.

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

## Running the Data page on the laptop against the shared registry
The Colab notebook keeps the registry and outputs on Google Drive. To use the same ones from the laptop, install
Google Drive for desktop, then start the console with the Drive paths:

```bash
export HACKPIPE_REGISTRY="$HOME/Google Drive/My Drive/shs-hack-2026/state/registry/registry.db"
export HACKPIPE_OUTPUTS_DIR="$HOME/Google Drive/My Drive/shs-hack-2026/state/outputs"
hackpipe console
```

Datasets must be at the same `data/raw/<name>` paths on both machines (demo IDs depend on them). Only one machine
writes at a time (registry writer, shown top right).

## Measured on the real data (laptop, 23 Oct)
- First run on night84: 76 s (566,244 frames decoded). ds-v1 fingerprint `01a77eac94728a96`, 76 / 8 / 10.
- Re-run on the same folder: 0.7 s, all 84 unchanged, no file rehashed, no new version.
- A copy of lighting10 added as "new": 10 × DUPLICATE in 0.4 s, nothing checked, no new version.
