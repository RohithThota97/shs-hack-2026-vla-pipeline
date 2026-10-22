# 00 · Shell, theme and Overview (global page)

## Goal
One composite layout for every page, and a global page that explains the challenge and shows where the pipeline
stands, so a juror or teammate understands the state in 10 seconds.

## Layout (every page)
```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ FR3 Duo · Demo-to-Deployment          [Dataset ds-v1 (approved)] [Model —] [Deployed —]   │
│ SHS Hackathon 2026 · Challenge 2      [Machine laptop · CPU only] [Writer laptop]         │
├────────────┬─────────────────────────────────────────────────────────┬───────────────────┤
│ Pipeline   │                                                         │ Jobs              │
│  Overview  │                 page content                            │  job-0003 QC  ▶   │
│ Stages     │                                                         │  expected 1m 20s  │
│  Data      │                                                         │  elapsed 0m 31s   │
│  Train     │                                                         │  CPU · laptop     │
│  Evaluate  │                                                         │  [log] [Cancel]   │
│  Deploy    │                                                         │ Registry writer   │
│  Improve   │                                                         │  laptop · Release │
└────────────┴─────────────────────────────────────────────────────────┴───────────────────┘
```
- **Header chips:** latest dataset version and state, latest model, deployed model, machine (name + GPU or "CPU
  only"), registry writer.
- **Left nav:** Overview · Data · Train · Evaluate · Deploy · Improve.
- **Right rail (refreshes every 5 s):** the last 8 jobs. Each shows label, status badge, job ID, CPU/GPU, machine,
  expected time and its basis, elapsed time, and a progress bar *by expected time* (labelled as such, not as
  measured progress). It also has a log tail and Cancel. Below the jobs, the writer box: Release, or Take over
  (only after ticking "I'm sure the other machine has stopped writing").

## Theme
Siemens Healthineers-like approximation; no logo, no "Siemens" wording, no official template.

| Token | Value | Use |
|---|---|---|
| accent | `#EC6602` | header rule, IDs, primary buttons, "next" box |
| ink / muted | `#1A1A1A` / `#6B6B6B` | text |
| line / panel | `#E3E3E3` / `#F6F6F6` | card borders, sidebar |
| ok / warn / bad / info | `#2E7D32` / `#B26A00` / `#C62828` / `#00677F` | status badges |

The font is Inter (Google Fonts, free). The tokens live in `src/hackpipe/console/theme.py` and
`.streamlit/config.toml`.

## Overview content
1. The challenge's leading question (from the organiser slides).
2. **Next box:** one sentence from `hackpipe.status.next_steps`, for example "Review and approve ds-v2:
   Data → Versions".
3. **Five stage cards** with live status, all from `hackpipe.status.overview`:

| Card | Lines | Status rule |
|---|---|---|
| 01 Ingest & Transform | demos, sources | done if any demo |
| 02 Dataset Intelligence | valid/review/invalid/duplicate; latest version + state | done if an approved version; waiting if only drafts |
| 03 Fine-Tune | models; latest + state | active if training; done if a trained model |
| 04 Evaluate | evaluations | done if any |
| 05 Version & Serve | deployed model | done if deployed |

4. Option cards **A** (success judgement) and **B** (self-improving pipeline).
5. **Known gaps** of the latest approved dataset (from the dataset card logic), e.g. "94 demos share only 1
   distinct trajectory".

## Machine & expected time
- `hackpipe.jobs.machine.info()`: machine ID ("colab" on any Colab session), CPUs, RAM, free disk, GPUs from
  `nvidia-smi`.
- `hackpipe.jobs.estimates.estimate(kind, work_units)` = work ÷ median throughput of earlier finished jobs of the
  same kind, preferring the same machine. The first run shows "no estimate yet (first run)".

## Registry writer
The SQLite registry may sit on Google Drive shared by the laptop and Colab. A machine claims the writer role on
its first write (CLI or console) and keeps it until **Release** or a deliberate **Take over**. There's no timeout,
because a training run can last hours. CLI: `hackpipe writer [--release | --take-over]`.

## Running on Colab (GPU pages)
A notebook cell starts `hackpipe console` and a `cloudflared` quick tunnel (free, no account) and prints the URL.
The machine chip then shows the T4. Built with `ui/train`.

## Done when
`hackpipe console` opens Overview with the correct cards on an empty and on a filled registry. Jobs started from
the CLI-based job API appear in the rail with expected time, machine and log. Writer Release and Take over work.
Every page loads in `AppTest`.
