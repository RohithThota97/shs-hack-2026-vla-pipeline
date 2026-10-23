"""Command line: `hackpipe <command>` (or `python -m hackpipe <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hackpipe.pipeline.sources import KINDS as SOURCE_KINDS   # stdlib-only module, cheap to import



def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _exit_for(out: dict) -> None:
    """auto over several folders: fail the job only if no folder went through; otherwise warn in the log."""
    errs = out.get("errors") or []
    if not errs:
        return
    for e in errs:
        print(f"WARNING: skipped {e['source']}: {e['error']}", file=sys.stderr)
    if not out.get("sources"):
        sys.exit(1)


def _writes(args) -> bool:
    """Commands that change the registry claim the writer role first (see common/writelock.py)."""
    if args.cmd in ("ingest", "qc", "auto", "demo", "eval", "gates", "deploy", "rollback"):
        return True
    if args.cmd == "dataset":
        return args.action in ("create", "approve") or (args.action == "materialize" and not args.plan)
    if args.cmd in ("train", "publish"):
        return not args.dry_run
    if args.cmd == "fetch":
        return args.auto
    return False


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hackpipe", description="Demo-to-deployment pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="R1 import episodes as demo records")
    s.add_argument("--root", required=True)
    s.add_argument("--watch", action="store_true", help="keep polling for new episodes")
    s.add_argument("--interval", type=int, default=60)

    s = sub.add_parser("qc", help="R2 data quality checks + report (new/changed demos only)")
    s.add_argument("--root", required=True)
    s.add_argument("--full", action="store_true", help="re-check every episode, not only new/changed ones")
    s = sub.add_parser("auto", help="new demos → import → QC (new only) → next draft version, automatically")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", action="append", help="dataset folder (repeat for several → one version)")
    g.add_argument("--all", action="store_true", help="every dataset under data/incoming and data/raw, after "
                   "linking new ones from dataset.watch_dirs as data/raw/<name>")
    g.add_argument("--link", metavar="PATH", help="link this folder / zip as data/raw/<name>, then run on it")
    s.add_argument("--name", help="with --link: name under data/raw (default: pinned name or folder name)")
    sub.add_parser("watch", help="datasets in dataset.watch_dirs that aren't linked into data/raw yet")
    s.add_argument("--notes", default="")
    s = sub.add_parser("fetch", help="bring a dataset into data/incoming/<name> from a folder/zip, Hugging Face, "
                                     "GitHub, Google Drive or a cloud bucket; keys from the environment")
    s.add_argument("--from", dest="kind", required=True, choices=SOURCE_KINDS,
                   help="local = folder or zip on this machine (also a mounted Drive)")
    s.add_argument("src", help="path / zip, HF dataset repo ID, owner/repo or URL, Drive link, s3:// gs:// az://")
    s.add_argument("--name", help="folder name under data/incoming (default: from src)")
    s.add_argument("--revision", help="branch / tag / commit (hf, github)")
    s.add_argument("--subdir", default="", help="only this folder of the HF repo")
    s.add_argument("--copy", action="store_true", help="local: copy instead of linking the folder")
    s.add_argument("--auto", action="store_true", help="then run the pipeline on what was fetched")
    s = sub.add_parser("demo", help="exclude / include a demo for future dataset versions")
    s.add_argument("action", choices=["exclude", "include"])
    s.add_argument("demo_id")
    s.add_argument("--reason", default="")
    s.add_argument("--by", default="")

    s = sub.add_parser("dataset", help="R3 versioned datasets")
    ds_sub = s.add_subparsers(dest="action", required=True)
    c = ds_sub.add_parser("create")
    c.add_argument("--notes", default="")
    c.add_argument("--base", help="approved version to build on, or 'latest'")
    a = ds_sub.add_parser("approve")
    a.add_argument("version")
    a.add_argument("--by", default="", help="who approves (recorded)")
    mz = ds_sub.add_parser("materialize", help="merge a multi-source version into one training folder")
    mz.add_argument("version")
    mz.add_argument("--plan", action="store_true", help="print the merge command and mapping only")
    ds_sub.add_parser("list")

    s = sub.add_parser("train", help="R4 reproducible training")
    s.add_argument("--config")
    s.add_argument("--dataset", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--resume", metavar="MODEL_VERSION")
    s.add_argument("--register-only", action="store_true", help="register an externally trained checkpoint")
    s.add_argument("--checkpoint")
    s.add_argument("--policy-name")
    s.add_argument("--purpose", choices=["model", "timing"], default="model",
                   help="'timing' = short speed measurement, labelled 'timing run, not a result'")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a train-config value, e.g. --set steps=200 (repeatable; stored in the run record)")

    s = sub.add_parser("eval", help="R5 run the test suite")
    s.add_argument("--model", required=True)
    s.add_argument("--suite", default="configs/eval/test_suite_v1.yaml")
    s.add_argument("--adapter", choices=["organizer", "dummy"], default="organizer",
                   help="'dummy' gives FAKE results for pipeline testing only")

    for name, helptext in (("gates", "R6 quality gates"), ("card", "R7 quality statement"),
                           ("improve", "R11/B recommendations")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--eval", required=True, dest="eval_id")

    s = sub.add_parser("deploy", help="R10 gated deployment")
    s.add_argument("--model", required=True)
    s.add_argument("--allow-supervised", action="store_true")
    sub.add_parser("rollback", help="R10 roll back to the previous deployed model")
    s = sub.add_parser("publish", help="push a model + model card to the Hugging Face Hub (never the dataset)")
    s.add_argument("--model", required=True)
    s.add_argument("--repo", help="<hf-user>/<name>; default hub.model_repo_id in configs/project.yaml")
    s.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    s.add_argument("--dry-run", action="store_true", help="write the card and list files, upload nothing")
    sub.add_parser("status", help="show registry contents")
    s = sub.add_parser("console", help="start the web console (needs the [console] extra)")
    s.add_argument("--port", type=int, default=8501)
    s = sub.add_parser("jobs", help="list background jobs started from the console")
    s.add_argument("--cancel", metavar="JOB_ID")
    s = sub.add_parser("writer", help="show / release / take over the registry writer role")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--release", action="store_true")
    g.add_argument("--take-over", action="store_true", help="only when the other machine has stopped writing")

    args = p.parse_args(argv)

    if args.cmd == "console":
        import subprocess
        from hackpipe.common.config import REPO_ROOT
        app = Path(__file__).parent / "console" / "app.py"
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                                  "--server.port", str(args.port), "--server.maxUploadSize", "5120"],
                                 cwd=REPO_ROOT))
    if args.cmd == "writer":
        from hackpipe.common import writelock
        if args.release:
            print("released" if writelock.release() else "this machine is not the writer")
        elif args.take_over:
            _print(writelock.acquire(force=True))
        else:
            _print({"writer": writelock.holder(), "this_machine": writelock.machine_id()})
        return
    if _writes(args):
        from hackpipe.common import writelock
        try:
            writelock.acquire()
        except writelock.WriterLockError as e:
            sys.exit(str(e))

    from hackpipe.common.registry import Registry
    reg = Registry()

    if args.cmd == "ingest":
        from hackpipe.ingest.importer import ingest, watch
        if args.watch:
            watch(args.root, args.interval, reg)
        else:
            from hackpipe.ingest.importer import import_demos
            r = import_demos(args.root, reg)
            _print({k: (len(v) if isinstance(v, list) else v) for k, v in r.items()})
    elif args.cmd == "qc":
        from hackpipe.quality.report import run_qc
        r = run_qc(args.root, reg, full=args.full)
        _print({k: r[k] for k in ("n_episodes", "checked_now", "reused", "status_counts", "reason_counts",
                                  "dataset_issues", "paths")})
    elif args.cmd == "auto":
        from hackpipe.common.config import source_key
        from hackpipe.pipeline.auto import run_auto, run_auto_many
        if args.all:
            from hackpipe.pipeline.preview import candidate_folders
            from hackpipe.pipeline.watch import link, unlinked
            link_errors = []
            for d in unlinked():
                try:
                    print(f"linked {source_key(link(d['path'], d['name']))} ← {d['path']}", flush=True)
                except (OSError, ValueError) as e:
                    link_errors.append({"source": d["path"], "error": f"{type(e).__name__}: {e}"})
            out = run_auto_many(candidate_folders(), reg, args.notes)
            out["errors"] = link_errors + out.get("errors", [])
        elif args.link:
            from hackpipe.pipeline.watch import link
            dest = link(args.link, args.name)
            print(f"linked {source_key(dest)} ← {args.link}", flush=True)
            out = run_auto(str(dest), reg, args.notes)
        elif len(args.root) == 1:
            out = run_auto(args.root[0], reg, args.notes)
        else:
            out = run_auto_many(args.root, reg, args.notes)
        _print(out)
        _exit_for(out)
    elif args.cmd == "watch":
        from hackpipe.pipeline.watch import unlinked, watch_dirs
        _print({"watch_dirs": [str(d) for d in watch_dirs()], "not_linked": unlinked()})
    elif args.cmd == "fetch":
        import signal
        from hackpipe.common.config import source_key
        from hackpipe.pipeline.sources import drop_console_key, fetch
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # cancel from the console → finally runs
        try:
            roots = fetch(args.kind, args.src, args.name, args.revision, args.subdir, args.copy)
        finally:
            drop_console_key()
        for r in roots:
            print(f"found dataset: {source_key(r)}")
        if args.auto:
            from hackpipe.pipeline.auto import run_auto_many
            out = run_auto_many([str(r) for r in roots], reg)
            _print(out)
            _exit_for(out)
    elif args.cmd == "demo":
        from hackpipe.dataset.versioning import set_excluded
        set_excluded(args.demo_id, args.action == "exclude", args.reason, args.by, reg)
        print(f"{args.demo_id} {args.action}d")
    elif args.cmd == "dataset":
        from hackpipe.dataset import versioning as v
        if args.action == "create":
            base = v.latest_approved(reg) if args.base == "latest" else args.base
            m = v.create_dataset_version(reg, notes=args.notes, base=base)
            _print({"version": m["version"], "fingerprint": m["fingerprint"], "base": base,
                    "counts": {k: len(x) for k, x in m["demo_ids"].items()}, "status": "draft"})
        elif args.action == "approve":
            v.approve(args.version, reg, by=args.by)
            print(f"{args.version} approved" + (f" by {args.by}" if args.by else ""))
        elif args.action == "materialize":
            from hackpipe.dataset.materialize import materialize
            _print(materialize(args.version, reg, run=not args.plan))
        else:
            _print([{k: d[k] for k in ("version", "fingerprint", "status", "created_at")} for d in reg.datasets()])
    elif args.cmd == "train":
        from hackpipe.training import train as t
        if args.register_only:
            if not (args.checkpoint and args.policy_name):
                sys.exit("--register-only needs --checkpoint and --policy-name")
            print(t.register_external(args.checkpoint, args.dataset, args.policy_name, reg))
        else:
            if not args.config and not args.resume:
                sys.exit("--config is required")
            rec = t.train(args.config, args.dataset, args.dry_run, reg, resume_model=args.resume,
                          overrides=t.parse_overrides(args.set), purpose=args.purpose)
            _print({k: rec.get(k) for k in ("model_version", "dataset_version", "output_dir", "exit_code")})
    elif args.cmd == "eval":
        from hackpipe.evaluation.runner import run_suite
        from hackpipe.sim.adapter import DummySimAdapter, OrganizerSimAdapter
        adapter = DummySimAdapter() if args.adapter == "dummy" else OrganizerSimAdapter()
        r = run_suite(args.model, args.suite, adapter, reg)
        m = r["metrics"]
        _print({"eval_id": r["eval_id"], "results_dir": r["results_dir"], "task_success": m["task_success"],
                "fake_results": m["fake_results"]})
    elif args.cmd == "gates":
        from hackpipe.gates.gates import evaluate_gates
        _print(evaluate_gates(args.eval_id, reg))
    elif args.cmd == "card":
        from hackpipe.reporting.model_card import build_card
        print(build_card(args.eval_id, reg))
    elif args.cmd == "improve":
        from hackpipe.improve.gaps import recommend
        _print(recommend(args.eval_id, reg))
    elif args.cmd == "deploy":
        from hackpipe.deploy.deploy import deploy
        r = deploy(args.model, reg, args.allow_supervised)
        _print({k: r[k] for k in ("dep_id", "status", "smoke_test", "launch")} |
               {"recommendation": r["gates"]["recommendation"]})
    elif args.cmd == "rollback":
        from hackpipe.deploy.deploy import rollback
        _print(rollback(reg))
    elif args.cmd == "publish":
        from hackpipe.hub.publish import publish
        _print(publish(args.model, args.repo, private=not args.public, dry_run=args.dry_run, registry=reg))
    elif args.cmd == "jobs":
        from hackpipe.jobs import runner
        runner.reconcile(reg)
        if args.cancel:
            print("cancelled" if runner.cancel(args.cancel, reg) else "not running")
        else:
            _print([{k: j[k] for k in ("job_id", "kind", "status", "needs", "machine", "estimate_s", "started_at",
                                       "ended_at", "exit_code")} for j in reg.jobs(limit=20)])
    elif args.cmd == "status":
        _print({
            "demos": {s: len(reg.demos(status=s)) for s in ("imported", "VALID", "REVIEW", "INVALID", "DUPLICATE")},
            "datasets": [{k: d[k] for k in ("version", "status")} for d in reg.datasets()],
            "models": [{k: m[k] for k in ("version", "policy_name", "dataset_version", "status")} for m in reg.models()],
            "evals": [{"eval_id": e["eval_id"], "success": e["metrics"]["task_success"]["rate"],
                       "fake": e["metrics"].get("fake_results")} for e in reg.evals()],
            "deployed": reg.current_deployed_model(),
        })


if __name__ == "__main__":
    main()
