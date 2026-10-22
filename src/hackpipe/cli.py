"""Command line: `hackpipe <command>` (or `python -m hackpipe <command>`)."""
from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hackpipe", description="Demo-to-deployment pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="R1 import episodes as demo records")
    s.add_argument("--root", required=True)
    s.add_argument("--watch", action="store_true", help="keep polling for new episodes")
    s.add_argument("--interval", type=int, default=60)

    s = sub.add_parser("qc", help="R2 data quality checks + report")
    s.add_argument("--root", required=True)

    s = sub.add_parser("dataset", help="R3 versioned datasets")
    ds_sub = s.add_subparsers(dest="action", required=True)
    c = ds_sub.add_parser("create")
    c.add_argument("--notes", default="")
    a = ds_sub.add_parser("approve")
    a.add_argument("version")
    ds_sub.add_parser("list")

    s = sub.add_parser("train", help="R4 reproducible training")
    s.add_argument("--config")
    s.add_argument("--dataset", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--resume", metavar="MODEL_VERSION")
    s.add_argument("--register-only", action="store_true", help="register an externally trained checkpoint")
    s.add_argument("--checkpoint")
    s.add_argument("--policy-name")

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
    sub.add_parser("status", help="show registry contents")

    args = p.parse_args(argv)

    from hackpipe.common.registry import Registry
    reg = Registry()

    if args.cmd == "ingest":
        from hackpipe.ingest.importer import ingest, watch
        if args.watch:
            watch(args.root, args.interval, reg)
        else:
            print(f"Imported {ingest(args.root, reg)} demos")
    elif args.cmd == "qc":
        from hackpipe.quality.report import run_qc
        r = run_qc(args.root, reg)
        _print({k: r[k] for k in ("n_episodes", "status_counts", "reason_counts", "dataset_issues", "paths")})
    elif args.cmd == "dataset":
        from hackpipe.dataset import versioning as v
        if args.action == "create":
            m = v.create_dataset_version(reg, notes=args.notes)
            _print({"version": m["version"], "fingerprint": m["fingerprint"],
                    "counts": {k: len(x) for k, x in m["splits"].items()}, "status": "draft"})
        elif args.action == "approve":
            v.approve(args.version, reg)
            print(f"{args.version} approved")
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
            rec = t.train(args.config, args.dataset, args.dry_run, reg, resume_model=args.resume)
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
    elif args.cmd == "status":
        _print({
            "demos": {s: len(reg.demos(status=s)) for s in ("imported", "VALID", "REVIEW", "INVALID")},
            "datasets": [{k: d[k] for k in ("version", "status")} for d in reg.datasets()],
            "models": [{k: m[k] for k in ("version", "policy_name", "dataset_version", "status")} for m in reg.models()],
            "evals": [{"eval_id": e["eval_id"], "success": e["metrics"]["task_success"]["rate"],
                       "fake": e["metrics"].get("fake_results")} for e in reg.evals()],
            "deployed": reg.current_deployed_model(),
        })


if __name__ == "__main__":
    main()
