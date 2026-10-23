# Dataset handoff for training

Run `python training_preparation/prepare_handoff.py` to generate episode manifests and train-only state/action statistics under `training_preparation/handoff_data/`. The script reads the original LeRobot v3 datasets at the paths in `handoff_sources.json`; it does not copy or edit them and does not start training. If the `D:` paths are not mounted on the training machine, update only the source config to the corresponding existing dataset locations, then regenerate the artifacts.

Choose one dataset ID and use its `dataset_manifest.json`, `splits.json`, and, when present, `train_state_action_stats.json`. The manifest's `dataset_root` is the exact root for the trainer. Pass only the listed `train` episode IDs to optimization, use `validation` for model selection, and reserve `test` for final evaluation. Do not use the source dataset's existing `meta/stats.json` for normalization; it was not produced from this prepared training split.

## Current data disposition

- `cube_stack490` is the only dataset with enough distinct exact numeric state/action trajectories for a train/validation/test split. It remains a separate cube-stacking task. Its 16-element state/action features have no declared names, so confirm the control semantics and units with the dataset owner before interpreting outputs for robot execution. Its source success labels are all true and are not independently verified by this preparation step.
- `handover84` and `handover10` are kept as separate selectable sources, but each contains one exact numeric state/action trajectory group. Those same numeric trajectories occur across both datasets. No train/validation/test split or normalization stats are emitted for either handover source. Identical numeric trajectories do not prove identical images or demonstrations; provenance and independence remain unresolved.

All sources already use LeRobot v3 with three 224x224 camera streams, 16-element float32 state/actions, and the source FPS. No resampling, frame trimming, image re-encoding, task mixing, or source-data edits are performed. Video decoding and image loading occur later in the trainer's data loader; this handoff does not claim a model forward pass or successful robot behavior.

The generated artifacts are split/configuration sidecars, not copies of the datasets. The training machine needs read access to the original dataset root. The policy trainer must support explicit episode filtering and accept the provided train-only statistics (or recompute equivalent statistics from the train IDs only).
