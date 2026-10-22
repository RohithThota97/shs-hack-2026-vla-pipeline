# P6 · Pitch, demo and jury Q&A

> **Not yet updated for the FR3 Duo handover track (decided 22 Oct).** This brief still describes the old
> LIBERO-Object plan (per-object suite, `lerobot-eval`, MuJoCo). Rewrite it before evaluation starts —
> see `agents/README.md`. Don't run the LIBERO commands below.

**Goal:** every number in the pitch is traceable to a logged run; the live demo has a script and a backup;
every `[bracket]` in `docs/jury_qa.md` is filled.
**Owner:** pitch lead (all help). **Where:** code session. **No GPU.**
**Read first:** `docs/jury_qa.md`, STATUS.md, the quality statements, QC reports, dataset cards, `gates.json`
files, `docs/decisions.md`. Ask the human for the pitch format and length.
**Needs:** both evaluations done (Day 3). Start the structure earlier.

## 1 · Numbers sheet

`docs/pitch_numbers.md`: every number that will be said or shown.

| Number | Value | Source file | Command that produced it |
|---|---|---|---|

Sources: `model_card.md`, `metrics.json`, `qc_report.md`, `dataset_card.md`, `run_record.json`, `gates.json`.
A number without a source file is removed.

## 2 · Fill the jury Q&A

Fill each `[bracket]` in `docs/jury_qa.md` from the numbers sheet. If something wasn't run, write
"not run — <reason>". Use "within noise" wherever the card says so. Add anything that went wrong on site to
"Limitations".

## 3 · Demo script

`docs/demo_script.md`, timed to the pitch length:
1. Problem in one line (clinical staff shortage → robots must learn from demonstrations).
2. Control centre live: Overview → Data (QC, ds-v1 / ds-v2) → Evaluate (per-object CIs, a rollout video) →
   Quality (gates, statement, improve command) → deploy / rollback.
3. Result: model_v1 vs model_v2, per object, with the assessment column.
4. Limits and next steps.

Backup: a screen recording of the full demo plus the demo clips from P3, playable offline.
Fallback commands if the dashboard fails: `hackpipe status`, `hackpipe card --eval <id>`.

## 4 · Slides

Content only from the numbers sheet. Ask the human which format to use. Reference-checkpoint results are
never shown as ours; the held-back pool is described as simulated targeted data collection.

## Done when

- [ ] `docs/pitch_numbers.md` with a source for every number
- [ ] No `[bracket]` left in `docs/jury_qa.md`
- [ ] `docs/demo_script.md` rehearsed once with a timer
- [ ] Backup video recorded and playable without network

## Hand-off (STATUS.md)

Paths of the numbers sheet, demo script, backup video; open risks for the live demo.

## STOP and ask

A number is needed that no logged run produced · pitch format unknown.
