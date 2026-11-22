"""Serving path: checkpoint -> triggers -> ANN -> ranked item ids, behind a stdlib HTTP server.

    python xrec_ref/serve.py --ckpt runs/xrec.pt --port 8731 --nlist 256

    GET  /health                 model/index/metrics summary
    GET  /metrics                counters + latency percentiles
    POST /recommend              {"items": [...], "actions": [...], "durations": [...],
                                  "K": 20, "R": 1, "nprobe": 8}

Response shape (one entry per request in the batch):

    {"items": [[id, ...], ...], "scores": [[f, ...], ...], "latency_ms": 12.3,
     "n_triggers": 20, "nprobe": 8, "dedup": false}

Design notes
  * One model, one frozen item table: the ANN index is built once at startup and is read-only, so the
    only lock is the one around `model.sample` (torch module state, not the index).
  * Candidates are returned per trigger, un-deduped by default, because Recall@(K x R) is defined on the
    union of per-trigger lists (eq.33); "dedup": true collapses duplicates for UI display.
  * No FastAPI/uvicorn dependency: the repo stays torch+numpy+sklearn, so `pip install -r requirements.txt`
    is still the whole story.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from config import CFG
from index import IVFIndex
from model import U2I, XRec

MAX_BODY = 8 << 20


def load_artifact(path: str, device: torch.device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in blob or "cfg" not in blob:
        raise ValueError(f"{path} is not an xrec_ref checkpoint (missing cfg/state_dict)")
    fields = set(CFG.__dataclass_fields__)
    cfg = CFG(**{k: v for k, v in blob["cfg"].items() if k in fields})
    emb = blob.get("item_emb")
    anchor = blob.get("anchor_of_item")
    if emb is None or anchor is None:
        raise ValueError("checkpoint has no item table; retrain with --ckpt-out (it stores item_emb)")
    emb = emb.float()
    arch = blob.get("arch", cfg.arch)
    cfg.n_items = int(emb.shape[0])
    if arch == "xrec":
        model = XRec(cfg, emb, anchor)
    elif arch in ("u2i", "u2i_k"):
        model = U2I(cfg, emb, anchor, n_interest=cfg.n_interest if arch == "u2i_k" else 1)
    else:
        raise ValueError(f"arch={arch} has no servable artifact (sid_ar needs its quantiser refit)")
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), cfg, emb, arch


class Recommender:
    def __init__(self, ckpt: str, nlist: int | None = None, device: str = "auto", warmup: bool = True):
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.model, self.cfg, self.emb, self.arch = load_artifact(ckpt, self.device)
        n = self.emb.shape[0]
        self.nlist = min(nlist or 256, n)
        t0 = time.perf_counter()
        self.index = IVFIndex(d=self.emb.shape[1], nlist=self.nlist, seed=0).train(self.emb).add(self.emb)
        self.index_ms = (time.perf_counter() - t0) * 1e3
        self.lock = threading.Lock()
        self.chunk = max(1, int(self.cfg.n_eval_chunk))
        self.lat: deque[float] = deque(maxlen=2048)
        self.counters = dict(requests=0, errors=0, triggers=0, items_served=0)
        self.warmup_ms = None
        if warmup:
            t0 = time.perf_counter()
            self.recommend([[0] * min(8, self.cfg.seq_len)], K=self.cfg.K, R=1,
                           nprobe=min(4, self.nlist))
            self.warmup_ms = (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def recommend(self, items: list[list[int]], actions=None, durations=None,
                  K: int = 20, R: int = 1, nprobe: int = 1, dedup: bool = False) -> dict:
        ctx = torch.tensor(items, dtype=torch.long, device=self.device)
        n, _ = ctx.shape
        act = torch.zeros_like(ctx) if actions is None else \
            torch.tensor(actions, dtype=torch.long, device=self.device)
        dur = torch.zeros_like(ctx) if durations is None else \
            torch.tensor(durations, dtype=torch.long, device=self.device)
        parts = []
        for i in range(0, n, self.chunk):                    # cache is replicated K x per request
            with self.lock:
                parts.append(self.model.sample(ctx[i:i + self.chunk], act[i:i + self.chunk],
                                               dur[i:i + self.chunk], K=K))
        trig = torch.cat(parts)                                          # (n, K, d_e)
        scores, ids = self.index.search(trig.reshape(n * K, -1), k=R, nprobe=nprobe)
        scores, ids = scores.reshape(n, K * R), ids.reshape(n, K * R)
        out_items, out_scores = [], []
        for b in range(n):
            pair = list(zip(scores[b].tolist(), ids[b].tolist(), strict=False))
            if dedup:
                seen, keep = set(), []
                for s, i in pair:
                    if i not in seen:
                        seen.add(i)
                        keep.append((s, i))
                pair = keep
            out_scores.append([s for s, _ in pair])
            out_items.append([i for _, i in pair])
        self.counters["triggers"] += n * K
        self.counters["items_served"] += sum(len(x) for x in out_items)
        return dict(items=out_items, scores=out_scores, n_triggers=K, nprobe=int(nprobe),
                    dedup=bool(dedup))

    # -------------------------------------------------------------- reporting
    def summary(self) -> dict:
        lat = sorted(self.lat)
        pct = (lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]) if lat else (lambda p: 0.0)
        return dict(ok=True, arch=self.arch, device=str(self.device), n_items=int(self.emb.shape[0]),
                    d_e=int(self.emb.shape[1]), nlist=self.nlist, index_ms=self.index_ms,
                    warmup_ms=self.warmup_ms, torch=torch.__version__, **self.counters,
                    p50_ms=pct(0.50), p95_ms=pct(0.95))


class App:
    """Routing + validation, kept separate from the socket so it is unit-testable."""

    def __init__(self, rec: Recommender, verbose: bool = False):
        self.rec, self.verbose = rec, verbose

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, dict]:
        t0 = time.perf_counter()
        try:
            if method == "GET" and path in ("/health", "/", "/metrics"):
                return 200, self.rec.summary()
            if method == "POST" and path == "/recommend":
                return 200, self._recommend(json.loads(body or b"{}"), t0)
            return 404, dict(error="not_found", path=path)
        except json.JSONDecodeError as e:
            self.rec.counters["errors"] += 1
            return 400, dict(error="bad_json", detail=str(e))
        except (ValueError, KeyError, TypeError, IndexError) as e:
            self.rec.counters["errors"] += 1
            return 400, dict(error="bad_request", detail=f"{type(e).__name__}: {e}")

    def _recommend(self, req: dict, t0: float) -> dict:
        items = req.get("items")
        if not items:
            raise ValueError("'items' (item ids, or a list of id lists) is required")
        if not isinstance(items[0], (list, tuple)):
            items = [items]
        n_items = int(self.rec.emb.shape[0])
        for seq in items:
            if not seq:
                raise ValueError("empty context")
            if min(seq) < 0 or max(seq) >= n_items:
                raise ValueError(f"item id out of range [0, {n_items})")
        max_len = self.rec.cfg.seq_len
        items = [list(seq)[-max_len:] for seq in items]

        def as_batch(v):
            if v is None:
                return None
            rows = v if isinstance(v[0], (list, tuple)) else [v]
            return [list(r)[-max_len:] for r in rows]

        act, dur = as_batch(req.get("actions")), as_batch(req.get("durations"))
        K, R = int(req.get("K", self.rec.cfg.K)), int(req.get("R", 1))
        nprobe = int(req.get("nprobe", min(8, self.rec.nlist)))
        if not (1 <= K <= 512 and 1 <= R <= 512 and 1 <= nprobe <= self.rec.nlist):
            raise ValueError("K/R/nprobe out of bounds")
        res = self.rec.recommend(items, act, dur, K=K, R=R, nprobe=nprobe,
                                 dedup=bool(req.get("dedup", False)))
        dt = (time.perf_counter() - t0) * 1e3
        self.rec.lat.append(dt)
        self.rec.counters["requests"] += 1
        if self.verbose:
            print(f"POST /recommend n={len(items)} K={K} R={R} nprobe={nprobe} {dt:.1f}ms", flush=True)
        return dict(**res, latency_ms=round(dt, 3), n_requests=len(items))


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "xrec_ref/0.2"

        def _send(self, code: int, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):                                            # noqa: N802
            self._send(*app.handle("GET", self.path.split("?")[0], b""))

        def do_POST(self):                                           # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self._send(413, dict(error="payload_too_large"))
                return
            self._send(*app.handle("POST", self.path.split("?")[0], self.rfile.read(n) if n else b""))

        def log_message(self, fmt, *args):
            if app.verbose:
                print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def make_server(ckpt: str, bind: str = "127.0.0.1", port: int = 8731, nlist: int | None = None,
                device: str = "auto", warmup: bool = True, verbose: bool = False):
    rec = Recommender(ckpt, nlist=nlist, device=device, warmup=warmup)
    app = App(rec, verbose=verbose)
    srv = ThreadingHTTPServer((bind, port), make_handler(app))
    srv.daemon_threads = True
    return srv, app


def self_test(ckpt: str, K: int = 8) -> int:
    """Bind an ephemeral port, issue one real HTTP request, print the round-trip. Exit 0 on success."""
    import urllib.request

    srv, app = make_server(ckpt, bind="127.0.0.1", port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as r:
            health = json.loads(r.read())
        ctx = list(range(min(8, app.rec.emb.shape[0])))
        body = json.dumps(dict(items=[ctx], K=K, R=1, nprobe=min(8, app.rec.nlist))).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/recommend", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        assert health["ok"] and len(out["items"][0]) == K
        print(f"self-test ok: arch={health['arch']} items={health['n_items']} nlist={health['nlist']} "
              f"K={K} -> {out['items'][0][:5]} ({out['latency_ms']} ms)")
        return 0
    finally:
        srv.shutdown()
        srv.server_close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--nlist", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="round-trip one request and exit")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(self_test(args.ckpt))
    srv, app = make_server(args.ckpt, args.bind, args.port, args.nlist, args.device,
                           warmup=not args.no_warmup, verbose=args.verbose)
    addr, port = srv.server_address[:2]
    s = app.rec.summary()
    print(f"listening on http://{addr}:{port}  arch={s['arch']} items={s['n_items']} "
          f"nlist={s['nlist']} index={s['index_ms']:.1f}ms warmup={s['warmup_ms']}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
