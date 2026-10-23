# FR3 Duo data preparation

> **Current trainer handoff:** use [`HANDOFF.md`](HANDOFF.md) and regenerate the source-bound artifacts with `python prepare_handoff.py`. This README records an earlier preparation run; its snapshot paths, split assignments, and runtime details are historical and are not the current handoff. The current handover manifests intentionally have no train/validation/test episodes because exact numeric trajectory checks found only one group in each handover source.

Prepared for the Robotics & AI Hackathon on 22 October 2026. Scope: R1 ingestion, R2 quality checks, R3 versioning and splits. The source folders were not modified. No data was uploaded and no training job was launched.

The subsequently supplied `isaac_sim_hackathon_environment.md` has also been inspected. See [the environment evidence review](reports/ENVIRONMENT_REVIEW.md). It supports the handover control convention but describes a different ten-episode dataset and contains no numerical joint limits or stacking-specific mapping. The handover order/units/absolute-target convention was cross-checked against the actual 84-episode converter metadata; no episode status changed. Reattach these report annotations after rebuilding with `python review_environment_document.py`.

## Current result

| Source | Recorded task | Episodes | Frames | FPS | Candidate train / val / test |
|---|---|---:|---:|---:|---|
| handover | Box handover and placement on serving tray | 84 | 188,748 | 30 | 0 / 0 / 84 |
| stacking (`rand500`) | Stack three cubes into one tower | 490 | 172,526 | 6 | 342 / 74 / 74 |

**Status: candidates prepared; release blocked pending review. Zero episodes are approved.** No file failed the performed technical checks. All 246 MP4 files were fully decoded, covering 1,083,822 camera frames. Decoding success does not establish physical safety or task success.

The 84 handover episodes have byte-identical numeric state/action trajectories (images may differ). The conservative leakage guard assigns their one shared trajectory group wholly to the test reservation. Consequently, this source cannot support independent train/validation/test sets. Additional independent demonstrations are needed; no handover training statistics or usable training configuration can be produced from this split. Do not use the reserved test episodes for training and subsequently report them as held out.

The handover metadata contains Isaac Sim prim paths, rendering parameters, simulation seeds and simulation metrics. This conflicts with its description as real-camera data. The stacking source has a different task, FPS and unnamed feature dimensions. The datasets remain separate, without invented labels or control mappings. The stacking source's original `info.json` train range (`0:41`) was inconsistent with its actual 490 episodes; generated manifests explicitly select episode IDs.

## Files

- `datasets/`: self-contained native LeRobot v3 snapshots, each with a fingerprint in its directory name, source task text, Parquet data, videos and metadata.
- `datasets/*/preparation/`: preprocessing contract, frozen candidate split membership, ACT/SmolVLA manifests, original checksums and final payload checksums.
- `reports/<source>/episodes.json`: full per-demo registry, UUID, provenance, checks, source checksums, reasons and feature statistics.
- `reports/<source>/episodes.csv`: compact review queue.
- `reports/<source>/aggregate_statistics.json`: full-dataset descriptive statistics; never use these for model normalization.
- `reports/<source>/file_inventory.json`: SHA-256 and complete video decode results, including frame presentation timestamps.
- `reports/<source>/frozen_test.json`: frozen test trajectory groups. Do not edit an existing test reservation to improve results; make a new named benchmark version if the protocol changes.
- `reports/<source>/*_batch.npz`: actual camera/state/action batches, including action padding masks. These are candidate data smoke tests, not approved training samples.
- `reports/<source>/*_train_argv.json`: LeRobot CLI argument templates, with explicit episode IDs and 16-dimensional feature overrides.
- `verification.json`: snapshot checksum, normalization and batch verification results.

## Quality interpretation

`INVALID`: required file/data missing, decode failure, bad dimensions, nonfinite values, index gaps or violation of explicitly configured physical limits. `NEEDS_REVIEW`: unresolved bounds, provenance, semantic, synchronization, completion or outcome questions. `VALID`: all applicable checks passed with no unresolved review reason.

Unknown robot limits are reported as unverified; no nominal Franka limits were substituted for the actual scene/robot contract. Independent raw stream synchronization is not recoverable from these already-converted files. Recorded success booleans are source assertions. Handover includes final simulation metrics; stacking lacks final pose/operator evidence. There are no reliable layout identifiers for a verified held-out-layout test.

The reports store unavailable recording dates, operator and layout values as null/unknown. Relative timestamps are not wall-clock dates. Stable UUIDs depend on episode content and referenced file hashes; replacing a shared source shard creates new identities for affected episodes.

## Model preprocessing

State/action dimensions are preserved at 16. Native images remain RGB 224x224, with all three cameras and original frame rates. No resampling, interpolated actions, inferred instructions, automatic trimming or cross-task merging was applied. Chunk size is 50: 1.67 seconds at 30 Hz, 8.33 seconds at 6 Hz. Suggested execution horizon is one action before replanning; tune this on the training host.

Stacking `meta/stats.json` contains mean/std/min/max and quantiles calculated **only on the 342 candidate training episodes**. Camera normalization uses fixed ImageNet constants for ACT; SmolVLA uses identity visual normalization and its own image preprocessing. Full-source `norm_stats.json` copies were removed from candidate snapshots to prevent accidental use. The native per-episode descriptive statistics are retained as provenance; LeRobot 0.6.1 reads normalization from `meta/stats.json`.

ACT uses images and state, without language conditioning. SmolVLA receives the original task text, maps the three camera keys to `camera1/2/3`, and pads state/actions internally to 32 dimensions. ALOHA-specific coordinate conversion and delta-action conversion are disabled. These settings do not resolve the missing stacking control specification. The CLI templates explicitly override the base model's six-dimensional feature declarations to 16 dimensions.

No automatic `eval_split` is enabled: the explicit 74-episode validation and 74-episode test manifests are authoritative. Validation and test must reuse training normalization statistics. Do not ask the trainer to randomly repartition the training episodes or use the full snapshot without an episode filter.

## Run and reproduce

Use Python 3.12. The current preparation runtime is:

```powershell
$prepPython = 'C:\Users\ndnqu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$prepRoot = 'C:\Users\ndnqu\OneDrive\Hackathon-Oct2026\training_preparation'
& $prepPython "$prepRoot\pipeline.py"
& $prepPython "$prepRoot\verify_and_package.py"
```

The bundled `vendor/` contains pyarrow 26.0.0 and av 19.0.2. NumPy 2.3.5 comes from the runtime. On another machine install `requirements-data.txt`. `config.json` controls source paths, quality tolerances, proposed duration bounds, split seed and optional robot limits. The bounds are pipeline defaults, not organizer standards.

To poll the configured native v3 recording folders:

```powershell
& $prepPython "$prepRoot\pipeline.py" --watch --interval 30
```

The watcher imports after two stable polls, caches unchanged file checks, rejects source changes during ingestion and retries failures. It is supplied but is not running as a background service. It only supports the native v3 source format actually present. No live recording API, HDF5 source or Hub endpoint was supplied. To connect a real recorder, point each configured path at its finalized native dataset folder and run the watcher; recorder-side finalization is still needed. Stable polling alone cannot prove a paused writer has finalized an episode.

Copied snapshot payload checksums allow verification after transfer. The dataset fingerprint covers sorted source file hashes, preprocessing/quality configuration and candidate split membership; it is a local version, not a published Hugging Face revision.

## Before training release

1. Obtain the actual robot/scene limits and confirm state/action names, units, gripper convention and timing. Supply limits per feature in `config.json` as `limits[feature].min` and `.max` arrays.
2. Resolve the real-vs-simulation provenance conflict. Obtain the intended stacking task success evidence and criteria.
3. Review episodes and record explicit approvals. The optional local `approvals.json` is keyed by each episode's `content_hash`, with `{"decision":"APPROVE","reviewed":true}`. This is an explicit human assertion, not automatic approval. Never approve without resolving or explicitly accepting every listed reason. `approved_splits.json` remains separate from candidate manifests.
4. For handover, collect independent trajectories before creating a trainable dataset with credible holdouts. For both sources, provide layout/variation grouping to establish the requested unseen-layout benchmark.
5. Run `framework_smoke_test.py <path-to-preparation/act_train.json>` and the SmolVLA equivalent on the training host with `lerobot==0.6.1`, PyTorch and the relevant policy extras installed. Data-level NumPy/PyAV batches were tested locally; a native LeRobot DataLoader/model forward pass has **not** been run here.
6. Build a new approved release using only approved episodes and recompute normalization on its train subset. Do not substitute a reduced approved episode list while retaining candidate normalization. Review the supplied CLI templates against that release and GPU setup before training. Model weight loading, tokenization, GPU memory and optimization remain training-host checks.

## Primary compatibility references

- [LeRobot v3 dataset](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobot 0.6.1 dataset config](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/configs/default.py)
- [ACT configuration](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py)
- [SmolVLA configuration](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/configuration_smolvla.py)
- [SmolVLA fine-tuning](https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/smolvla.mdx)

The local Hackathon_Pipeline_Plan.pdf was read as a requirements reference; speculative robots, tasks, training time estimates and unrelated R4-R11 components were not adopted as facts about these data.
