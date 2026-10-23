"""Policy server processes: start (detached, survives the console and the job), stop, status, health check.

State per port in <outputs>/deploy/serving_<port>.json (pid, model, settings); log in server_<port>.log. Only
servers started here are stopped — a port held by something else is reported, never killed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.ids import now_iso


def settings() -> dict[str, Any]:
    return {"host": "127.0.0.1", "port": 8090, "stage_port": 8091, "probe_source": None, "n_action_steps": None,
            "start_timeout_s": 240, **(project_config().get("deploy") or {})}


def _dir() -> Path:
    d = resolve(project_config()["project"]["outputs_dir"]) / "deploy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_file(port: int) -> Path:
    return _dir() / f"serving_{port}.json"


def http_json(url: str, data: bytes | None = None, timeout: float = 5) -> dict[str, Any]:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _url(port: int, path: str) -> str:
    host = settings()["host"]
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}{path}"


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:                                  # reap it if it is our own child (else a stopped server stays a zombie)
        if os.waitpid(int(pid), os.WNOHANG)[0]:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def info(port: int, timeout: float = 2) -> dict[str, Any] | None:
    try:
        return http_json(_url(port, "/info"), timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def status(port: int | None = None) -> dict[str, Any]:
    port = port or settings()["port"]
    st = json.loads(_state_file(port).read_text()) if _state_file(port).exists() else None
    ours = bool(st and _alive(st.get("pid")))
    live = info(port)
    return {"port": port, "live": live is not None, "info": live, "ours": ours, "state": st if ours else None,
            "foreign": live is not None and not ours}


def start(port: int, model: str | None = None, checkpoint: str | None = None, dummy: bool = False,
          n_action_steps: int | None = None, grasp_extend: float = 0.0, wait: bool = True) -> dict[str, Any]:
    """Start `hackpipe serve` on this port in its own session; wait until /info answers."""
    cfg = settings()
    st = status(port)
    if st["live"]:
        raise RuntimeError(f"port {port} is already serving {(st['info'] or {}).get('model') or 'something'}"
                           + ("" if st["ours"] else " (not started by hackpipe — stop it yourself)"))
    cmd = [sys.executable, "-m", "hackpipe.cli", "serve", "--port", str(port), "--host", cfg["host"]]
    cmd += ["--dummy"] if dummy else (["--model", model] if model else ["--checkpoint", str(checkpoint)])
    if n_action_steps:
        cmd += ["--n-action-steps", str(int(n_action_steps))]
    if grasp_extend:
        cmd += ["--grasp-extend", str(float(grasp_extend))]
    if cfg.get("probe_source"):
        cmd += ["--probe-source", str(cfg["probe_source"])]
    log = _dir() / f"server_{port}.log"
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                cwd=str(resolve(".")))
    state = {"pid": proc.pid, "port": port, "model": model, "checkpoint": checkpoint, "dummy": dummy,
             "n_action_steps": n_action_steps, "grasp_extend": grasp_extend, "started_at": now_iso(),
             "log": str(log), "command": cmd}
    _state_file(port).write_text(json.dumps(state, indent=1))
    if not wait:
        return state
    t0 = time.time()
    while time.time() - t0 < float(cfg["start_timeout_s"]):
        if proc.poll() is not None:
            _state_file(port).unlink(missing_ok=True)
            raise RuntimeError(f"the server exited ({proc.returncode}): {log.read_text(errors='replace')[-600:]}")
        if info(port):
            return {**state, "info": info(port)}
        time.sleep(1)
    stop(port)
    raise RuntimeError(f"the server did not answer within {cfg['start_timeout_s']} s: "
                       f"{log.read_text(errors='replace')[-600:]}")


def stop(port: int | None = None, timeout: float = 20) -> bool:
    """Stop the server hackpipe started on this port (its whole process group). False if nothing of ours runs."""
    port = port or settings()["port"]
    f = _state_file(port)
    st = json.loads(f.read_text()) if f.exists() else None
    if not st or not _alive(st.get("pid")):
        f.unlink(missing_ok=True)
        return False
    try:
        os.killpg(int(st["pid"]), signal.SIGTERM)
    except OSError:
        os.kill(int(st["pid"]), signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < timeout and (_alive(st["pid"]) or info(port, 0.5)):
        time.sleep(0.5)
    if _alive(st["pid"]):
        try:
            os.killpg(int(st["pid"]), signal.SIGKILL)
        except OSError:
            pass
    f.unlink(missing_ok=True)
    return True


def health_check(port: int, allow_fake: bool = False) -> dict[str, Any]:
    """/info answers, /reset works, one recorded frame (/probe) gives 16 finite values; not FAKE (unless allowed)."""
    out: dict[str, Any] = {"port": port, "ok": False}
    i = info(port, 5)
    if i is None:
        return {**out, "error": "/info does not answer"}
    out.update(model=i.get("model"), policy_type=i.get("policy_type"), fake=i.get("fake"))
    if i.get("fake") and not allow_fake:
        return {**out, "error": "FAKE policy (dummy): never deployed"}
    try:
        http_json(_url(port, "/reset"), b"", timeout=30)
        t0 = time.time()
        p = http_json(_url(port, "/probe"), b"", timeout=300)
    except urllib.error.HTTPError as e:
        return {**out, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {**out, "error": f"{type(e).__name__}: {e}"}
    out.update(probe=p, first_call_ms=round((time.time() - t0) * 1000))
    out["ok"] = bool(p.get("ok")) and (allow_fake or not i.get("fake"))
    if not out["ok"]:
        out["error"] = "FAKE policy" if i.get("fake") and not allow_fake else f"probe failed: {p}"
    return out
