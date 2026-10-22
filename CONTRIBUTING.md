# Contributing

## Branches and pull requests

- `main` is always runnable. Do not push to it directly.
- Branch names: `<person>/<task>`, e.g. `p1/qc-video-check`, `p4/judge-vlm`.
- Open a PR, get **one** review, squash-merge.
- PR description: what changed, how you tested it, which interface (if any) changed.

## Commits

Short imperative messages: `Add timestamp gap check`, `Fix leakage guard for groups`.

## Before you open a PR

```bash
pytest
```

## Data, weights and secrets

- Never commit datasets, videos, checkpoints, `registry/registry.db` or anything under `outputs/`.
- Never commit links, QR keys or tokens from the organisers' file share. Put local values in `.env`
  (copy `.env.example`).

## Interfaces

Schemas in `docs/interfaces.md` are contracts between people. Changing one needs a PR that updates the doc and
tells the affected owner.

## Numbers

Only numbers from logged runs go into reports, the dashboard or the pitch. The dummy simulator adapter exists
for tests and produces fake results.
