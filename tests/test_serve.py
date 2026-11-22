"""Serving path: checkpoint -> ANN -> JSON over real HTTP (no mocks: a socket, a request, a response)."""
import json
import threading
import urllib.error
import urllib.request

import torch
from config import CFG
from model import XRec
from serve import make_server

from data import build_world, make_split


def _ckpt(tmp_path) -> tuple[str, CFG, dict]:
    cfg = CFG()
    cfg.n_items, cfg.n_anchor, cfg.n_modes = 300, 8, 4
    cfg.seq_len, cfg.d_e, cfg.d_m, cfg.n_head, cfg.n_layer = 8, 24, 48, 4, 2
    cfg.K, cfg.steps = 4, 2
    torch.manual_seed(0)
    world = build_world(cfg, seed=0)
    model = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    path = str(tmp_path / "ckpt.pt")
    torch.save(dict(cfg=cfg.to_dict(), arch="xrec", state_dict=model.state_dict(),
                    item_emb=world["emb"], anchor_of_item=world["anchor_of_item"]), path)
    return path, cfg, world


def _post(port: int, payload: dict, path: str = "/recommend"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _server(tmp_path):
    path, cfg, world = _ckpt(tmp_path)
    srv, app = make_server(path, bind="127.0.0.1", port=0, warmup=False)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, app, cfg, world


def test_health_and_recommend_end_to_end(tmp_path):
    srv, app, cfg, world = _server(tmp_path)
    port = srv.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as r:
            health = json.loads(r.read())
        assert health["ok"] and health["n_items"] == cfg.n_items and health["arch"] == "xrec"

        ctx = make_split(cfg, world, n=2, seed=5)["ctx"].tolist()
        status, out = _post(port, {"items": ctx, "K": 4, "R": 2, "nprobe": 4})
        assert status == 200
        assert len(out["items"]) == 2 and all(len(row) == 4 * 2 for row in out["items"])
        flat = [i for row in out["items"] for i in row]
        assert min(flat) >= 0 and max(flat) < cfg.n_items
        assert len(out["scores"]) == 2 and out["latency_ms"] >= 0

        status, dedup = _post(port, {"items": ctx[0], "K": 4, "R": 2, "dedup": True})
        assert status == 200 and len(dedup["items"]) == 1
        assert len(set(dedup["items"][0])) == len(dedup["items"][0])

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
            m = json.loads(r.read())
        assert m["requests"] >= 2 and m["triggers"] > 0 and m["p50_ms"] >= 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_bad_requests_are_rejected_without_killing_the_server(tmp_path):
    srv, app, cfg, world = _server(tmp_path)
    port = srv.server_address[1]
    try:
        assert _post(port, {})[0] == 400                                        # no items
        assert _post(port, {"items": [0, 1, 10 ** 9]})[0] == 400                # out of range
        assert _post(port, {"items": [0, 1], "K": 0})[0] == 400                 # K out of bounds
        req = urllib.request.Request(f"http://127.0.0.1:{port}/recommend", data=b"{not json",
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert app.rec.counters["errors"] >= 4
        assert _post(port, {"items": [0, 1, 2]})[0] == 200                      # still alive
    finally:
        srv.shutdown()
        srv.server_close()
