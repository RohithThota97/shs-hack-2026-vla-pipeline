# Decision log

Newest first. One line for the decision, one for the reason.

| Date | Decision | Reason |
|---|---|---|
| 22 Oct | Fine-tune pretrained VLA (SmolVLA) as the main model; ACT from scratch as baseline | Few demos per variation, language input, measured gain vs scratch on the same test suite |
| 22 Oct | π0.5 only as LoRA stretch; no GR00T fine-tuning | One RTX 4090 (24 GB): π0.5 LoRA is tight, GR00T fine-tuning needs 40 GB+ |
| 22 Oct | Optional modules A (success judge) + B (self-improving pipeline) | Both extend the core we build anyway; C/D only if time remains |
| 22 Oct | SQLite registry + Streamlit dashboard | No server needed; works offline |
