"""Demo pages (hackpipe/demo/pages.py) on a small synthetic handover recording, and the seekable static server."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")
READY = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])


def _recording(root, n=300):
    """Left reaches and grasps at frame 100, right grasps at 220 (30 fps), like the grid tests' demo."""
    A = np.tile(np.r_[READY, 0.48, READY, 0.48], (n, 1))
    A[:, 1] += np.linspace(0, 0.3, n)
    A[100:, 7] = 0.0
    A[220:, 15] = 0.0
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "features": {}}))
    (root / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame({"episode_index": 0, "frame_index": range(n), "timestamp": np.arange(n) / 30,
                  "observation.state": list(A.astype(np.float32)), "action": list(A.astype(np.float32))}
                 ).to_parquet(root / "data/chunk-000/file-000.parquet")
    return A


def test_tcp_and_retarget_pages(tmp_path):
    from hackpipe.demo import pages
    from hackpipe.sim import scene
    src = tmp_path / "rec"
    A = _recording(src)
    r = pages.build_tcp(tmp_path / "tcp.html", str(src), start_s=2.0, seconds=2.0)
    html = (tmp_path / "tcp.html").read_text()
    assert r["frames"] == 61 and "/*DATA*/" not in html and "Plotly" in html             # self-contained
    out = tmp_path / "retarget"
    st = pages.build_retarget(out, 2.0, 1.0, str(src))
    assert st["ok"] and st["max_pos_err_mm"] < 1 and {"index.html", "plotly.min.js", "data.json"} <= {
        f.name for f in out.iterdir()}
    d = json.loads((out / "data.json").read_text())
    g = int(round(d["grasp_t"] * 30))
    assert np.allclose(np.subtract(d["new"][g], d["orig"][g]), [0.02, 0, 0.01], atol=1e-3)  # room X / Z shift
    assert np.allclose(d["orig"], scene.tcp_world("left", A[:, :7]), atol=1e-4)


def test_model_page_recovers_the_joints(tmp_path):
    from hackpipe.demo import pages
    q = np.tile(READY, (20, 1)) + np.linspace(0, 0.2, 20)[:, None] * [1, 1, 0, 1, 0, 1, 1]
    QB = pages.recover_joints(q, fix7=True)
    assert np.abs(QB - q).max() < 1e-4                          # joint 7 given: the one solution is the recording
    from hackpipe.evaluation.kinematics import fk_T
    QA = pages.recover_joints(q, fix7=False)
    assert np.abs(fk_T(QA)[:, :3, 3] - fk_T(q)[:, :3, 3]).max() < 1e-4   # same gripper pose either way
    from hackpipe.sim import scene
    pts = scene.link_points("left", q[0])
    assert pts.shape == (9, 3) and np.allclose(pts[0], scene.arm_base("left")[:3, 3])


def test_labels_for_the_newest_version(tmp_path, monkeypatch):
    from hackpipe.demo import pages
    for v, when, text in (("ds-v2", "2026-10-22", "old"), ("ds-v3", "2026-10-23", "Pick up the box.")):
        d = tmp_path / "outputs/datasets" / v / "subtasks"
        d.mkdir(parents=True)
        (d / "labels.json").write_text(json.dumps({"created_at": when, "sources": {"data/raw/rec": {
            "segments": "seg.csv", "episodes": 1, "flagged": []}}}))
        pd.DataFrame({"episode_index": [0, 0], "start_frame": [0, 3], "end_frame": [3, 5],
                      "text": [text, "Place it."]}).to_csv(d / "seg.csv", index=False)
    monkeypatch.setattr(pages, "resolve", lambda p: tmp_path / p)
    assert pages.frame_subtasks("data/raw/rec", 0, 6) == ["Pick up the box."] * 3 + ["Place it."] * 2 + [""]
    assert pages.frame_subtasks("data/raw/other", 0, 2) == ["", ""]


def test_static_server_serves_byte_ranges(tmp_path):
    from hackpipe.demo import pages
    (tmp_path / "video.mp4").write_bytes(bytes(range(256)) * 4)
    srv = pages.make_server(tmp_path, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/video.mp4"
        req = urllib.request.Request(url, headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and r.headers["Content-Range"] == "bytes 10-19/1024" and r.read() == bytes(range(10, 20))
        with urllib.request.urlopen(url) as r:
            assert r.status == 200 and len(r.read()) == 1024
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=-6"})) as r:   # last 6
            assert r.headers["Content-Range"] == "bytes 1018-1023/1024" and r.read() == bytes(range(250, 256))
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=20-10"}))
        assert e.value.code == 416
    finally:
        srv.shutdown()
