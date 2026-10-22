# Decision log

Newest first. One line for the decision, one for the reason.

| Date | Decision | Reason |
|---|---|---|
| 23 Oct | `hackpipe auto`: import → QC of new/changed demos only → next draft based on the latest approved version; a human approves | New demonstrations flow through without manual steps; nothing is checked twice; approval stays human |
| 23 Oct | Duplicate = same episode index + content hash + file checksums as a demo from another folder | In LeRobot v3 episodes share files, and here all trajectories are identical, so content alone can't tell episodes apart |
| 23 Oct | ds-vN+1 inherits the base's train/val assignment; only added demos are split; test stays frozen | Versions stay comparable and v1 demos are not re-validated or reshuffled |
| 23 Oct | Multi-source versions are merged with `lerobot-edit-dataset` and the mapping is verified by content hash before training | The merge's episode ordering is a LeRobot internal; verifying beats assuming |
| 23 Oct | Web console in Streamlit (`hackpipe console`), one spec per page in `docs/ui/`, built page by page (Data first) | Python only, already a dependency, fastest to build; specs reviewed before code |
| 23 Oct | Look: Siemens Healthineers-like orange/grey palette, Inter font, no logo or Siemens wording | No official template files; avoid posing as an official product |
| 23 Oct | Pipeline steps from the console run as background jobs (registry `jobs` table); expected time only from measured earlier runs | Survive page reloads; show CPU/GPU, machine and time without guessing |
| 23 Oct | One machine writes the registry at a time (writer lock next to registry.db, no timeout) | Laptop and Colab share the registry on Google Drive; concurrent SQLite writes over sync can corrupt it |
| 22 Oct | Train on free Colab (T4); data comes from our Google Drive, registry + outputs on Drive | No GPU box confirmed; Drive state survives disconnects, checkpoints allow resume |
| 22 Oct | The dataset is never uploaded to the Hub; only model + model card, private repo (`hackpipe publish`) | Organiser data is restricted / team-specific |
| 22 Oct | All 94 handover demos have bit-identical state/action trajectories; only lighting (images) differs | Measured by R2 redundancy check; reported in QC report + dataset card. The policy will not learn pose variation from this data |
| 22 Oct | Joint limits in QC = Franka FR3 datasheet values, gripper 0..1 | Not in the organisers' files; data has ≥ 0.19 rad headroom (left q7). Confirm against the USD robot |
| 22 Oct | Train episodes selected with `--dataset.episodes` on the source folder, not a copy made with `lerobot-edit-dataset` | Episode indices stay those of the source (traceable); no 2 GB copy. `lerobot-edit-dataset` kept for merging sources and as AV1 → h264 fallback |
| 22 Oct | Test split = whole held-out dataset `v9_lighting10`, frozen from ds-v1; leakage guard on seed | Other seeds and lighting; a frozen test makes model versions comparable |
| 22 Oct | Main track = organisers' FR3 Duo handover data; LIBERO-Object dropped | It is the provided data and matches the provided Isaac Sim scene; DoD asks for our own policy on it |
| 22 Oct | ds-v1 from `fr3_duo_handover_v9_night_84` (84 ep); `v9_lighting10` (10 ep) = held-out test set | lighting10 has other seeds and lighting → honest robustness test; never trained on |
| 22 Oct | `fr3_duo_rand500` (cube stacking, 6 fps) not used for now | Different task, no matching scene in the environment zip |
| 22 Oct | Main policy: SmolVLA fine-tuned from `lerobot/smolvla_base` | Fits 24 GB, native LeRobot v3 support, language-conditioned |
| 22 Oct | Fine-tune pretrained VLA (SmolVLA) as the main model; ACT from scratch as baseline | Few demos per variation, language input, measured gain vs scratch on the same test suite |
| 22 Oct | π0.5 only as LoRA stretch; no GR00T fine-tuning | One RTX 4090 (24 GB): π0.5 LoRA is tight, GR00T fine-tuning needs 40 GB+ |
| 22 Oct | Optional modules A (success judge) + B (self-improving pipeline) | Both extend the core we build anyway; C/D only if time remains |
| 22 Oct | SQLite registry + Streamlit dashboard | No server needed; works offline |
