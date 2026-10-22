# Decision log

Newest first. One line for the decision, one for the reason.

| Date | Decision | Reason |
|---|---|---|
| 22 Oct | Main track = organisers' FR3 Duo handover data; LIBERO-Object dropped | It is the provided data and matches the provided Isaac Sim scene; DoD asks for our own policy on it |
| 22 Oct | ds-v1 from `fr3_duo_handover_v9_night_84` (84 ep); `v9_lighting10` (10 ep) = held-out test set | lighting10 has other seeds and lighting → honest robustness test; never trained on |
| 22 Oct | `fr3_duo_rand500` (cube stacking, 6 fps) not used for now | Different task, no matching scene in the environment zip |
| 22 Oct | Main policy: SmolVLA fine-tuned from `lerobot/smolvla_base` | Fits 24 GB, native LeRobot v3 support, language-conditioned |
| 22 Oct | Fine-tune pretrained VLA (SmolVLA) as the main model; ACT from scratch as baseline | Few demos per variation, language input, measured gain vs scratch on the same test suite |
| 22 Oct | π0.5 only as LoRA stretch; no GR00T fine-tuning | One RTX 4090 (24 GB): π0.5 LoRA is tight, GR00T fine-tuning needs 40 GB+ |
| 22 Oct | Optional modules A (success judge) + B (self-improving pipeline) | Both extend the core we build anyway; C/D only if time remains |
| 22 Oct | SQLite registry + Streamlit dashboard | No server needed; works offline |
