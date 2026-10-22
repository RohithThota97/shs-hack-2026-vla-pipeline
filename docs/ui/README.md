# UI specs · pipeline console

One file per page. Review a spec before its branch is built; change the spec first, then the code.

| Spec | Page | Branch | State |
|---|---|---|---|
| [00_shell_and_overview.md](00_shell_and_overview.md) | Layout, theme, global page, jobs, machine, writer | `ui/specs-shell` | built |
| [01_data.md](01_data.md) | Data (R1–R3) | `ui/data` | built |
| [02_train.md](02_train.md) | Train (R4) | `ui/train` | spec for review |
| [03_evaluate.md](03_evaluate.md) | Evaluate (R5, R8, option A) | `ui/evaluate` | spec for review |
| [04_deploy.md](04_deploy.md) | Deploy (R6, R7, R10, Hub) | `ui/deploy` | spec for review |
| [05_improve.md](05_improve.md) | Improve (R11, option B) | `ui/improve` | spec for review |

Start: `pip install -e ".[console]"` then `hackpipe console` (http://localhost:8501).

Rules for every page
- Numbers only from the registry, manifests, reports and logs. Unmeasured = "n/a", never 0 or a guess.
- Every action that runs a pipeline step starts a **job**: it shows CPU/GPU, machine, expected time (from measured
  earlier runs, else "no estimate yet") and a log.
- Every write needs the **registry writer** role (one machine at a time). Read-only pages always work.
- Human decisions (approve a dataset, deploy) are buttons that record who decided. Agents never press them.
- Free tools only: Streamlit, Plotly, SQLite, cloudflared (Colab tunnel).
