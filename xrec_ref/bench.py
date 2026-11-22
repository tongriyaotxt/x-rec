"""Efficiency analysis for X-Rec.

(A) model     measured: late-interaction DiT (1 layer per denoising step, condition KV cached once) vs
              full-layer denoising (re-run the whole stack every step). Same weights, same triggers.
(B) decode    measured: X-Rec vs the SID-AR control at identical shapes — serial decoding depth vs
              parallel denoising.
(C) ann       measured: IVF recall@k vs nprobe vs QPS, i.e. what "retrieval" costs once the catalog
              stops fitting in one matmul.
(D) analytic  cost model at the paper's production shapes (d=1024, L=28, N=199, K=20, T=30, 4 SID
              tokens, 20 beams). This is where the 3.46x comes from: autoregressive decoding is bound
              by the number of *serial* steps, while X-Rec advances all K triggers in parallel.

Run: python bench.py            (everything)
     python bench.py 12         (only the model section with L=12)
     python bench.py ann        (one section: model|decode|ann|analytic)
"""
from __future__ import annotations

import sys
import time

import torch
from config import CFG
from model import XRec
from rfm import normalize

from data import build_world, make_split


def timeit(fn, rep: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t0) / rep


# ---------------------------------------------------------------- (A) model
def measured(layers: int = None):
    torch.manual_seed(0)
    cfg = CFG()
    if layers:
        cfg.n_layer = layers
    world = build_world(cfg, seed=0)
    B, K, steps = 32, cfg.K, cfg.steps
    test = make_split(cfg, world, n=B, seed=4)
    items, act, dur = test["ctx"], test["act"], test["dur"]
    model = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    print(f"[A] CPU fixture: batch={B} N={cfg.seq_len} K={K} T={steps} "
          f"d_m={cfg.d_m} L={cfg.n_layer} d_e={cfg.d_e}")

    with torch.no_grad():
        t_late = timeit(lambda: model.sample(items, act, dur, K=K, steps=steps))

        def full_denoise():
            x = None
            for s in range(steps):
                cache, _h, f, N = model.prefill(items, act, dur)   # whole stack, every step
                if x is None:
                    x = normalize(torch.randn(B, K, cfg.d_e))
                idx = torch.arange(B).repeat_interleave(K)
                anc = torch.zeros(B * K, dtype=torch.long)
                v = model.velocity(x.reshape(B * K, -1), anc, torch.full((B * K,), s / steps),
                                   (cache[0].index_select(0, idx), cache[1].index_select(0, idx)),
                                   f.index_select(0, idx), N)
                from rfm import riemannian_euler_step
                x = riemannian_euler_step(x.reshape(B * K, -1), v, 1.0 / steps).reshape(B, K, cfg.d_e)
            return x

        t_full = timeit(full_denoise)

    print(f"{'method':<46}{'ms/request':>12}{'QPS':>10}{'vs late':>10}")
    for name, t in [("X-Rec late-interaction (1 layer/step)", t_late),
                    (f"X-Rec full-layer denoising ({cfg.n_layer} layers/step)", t_full)]:
        print(f"{name:<46}{1000*t/B:>12.2f}{B/t:>10.1f}{t/t_late:>9.2f}x")
    return t_late, t_full


# --------------------------------------------------------------- (B) decode
def decode_compare():
    from models_sid import SIDAR
    from sid import ResidualQuantizer

    torch.manual_seed(0)
    cfg = CFG()
    world = build_world(cfg, seed=0)
    test = make_split(cfg, world, n=32, seed=4)
    items, act, dur = test["ctx"], test["act"], test["dur"]

    x = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    q = ResidualQuantizer(n_codebooks=cfg.sid_codebooks, codebook_size=cfg.sid_size, seed=0)
    q.fit(world["emb"])
    s = SIDAR(cfg, world["emb"], world["anchor_of_item"], q, q.encode(world["emb"])).eval()

    with torch.no_grad():
        t_x = timeit(lambda: x.sample(items, act, dur, K=cfg.K, steps=cfg.steps))
        t_s = timeit(lambda: s.sample(items, act, dur, K=cfg.K))
        trig_s = s.sample(items, act, dur, K=cfg.K)

    print(f"\n[B] decode, same shapes: N={cfg.seq_len} K={cfg.K} d_m={cfg.d_m} L={cfg.n_layer} "
          f"RQ={cfg.sid_codebooks}x{cfg.sid_size} (recon cos={q.recon_cosine(world['emb']):.3f})")
    print(f"{'path':<34}{'ms/request':>12}{'serial steps':>14}{'triggers/req':>14}")
    print(f"{'X-Rec (T denoising steps)':<34}{1000*t_x/32:>12.2f}{cfg.steps:>14d}{cfg.K:>14d}")
    print(f"{'SID-AR (M tokens / level)':<34}{1000*t_s/32:>12.2f}{cfg.sid_codebooks:>14d}{cfg.K:>14d}")
    print("  note: beams decode in one batched top-k here, so serial depth is M not M x beams; the")
    print("        paper's 80-step figure needs 20 beams x 4 tokens, see section [D].")
    assert (trig_s.norm(dim=-1) - 1).abs().max().item() < 1e-4


# ------------------------------------------------------------------ (C) ann
def ann_compare(n_items: int = 50000, d: int = 128, nq: int = 512):
    from index import IVFIndex, sweep

    g = torch.Generator().manual_seed(0)
    # clustered catalog: 256 modes, Zipf-ish radii — the regime where IVF actually works
    centers = normalize(torch.randn(256, d, generator=g))
    mode = torch.randint(0, 256, (n_items,), generator=g)
    x = normalize(centers[mode] + 0.35 * normalize(torch.randn(n_items, d, generator=g)))
    q = normalize(x[torch.randint(0, n_items, (nq,), generator=g)]
                  + 0.30 * normalize(torch.randn(nq, d, generator=g)))

    t0 = time.perf_counter()
    ix = IVFIndex(d=d, nlist=256, seed=0).train(x).add(x)
    build_ms = (time.perf_counter() - t0) * 1e3
    print(f"\n[C] IVF over {n_items} items d={d} nlist=256 (build {build_ms:.0f} ms, "
          f"{ix.stats()['bytes']/1e6:.1f} MB fp32)")
    print(f"{'nprobe':>8}{'recall@10':>12}{'ms/query':>11}{'QPS':>10}")
    for r in sweep(ix, q, x, k=10, nprobes=(1, 2, 4, 8, 16, 32, 256), repeats=3):
        print(f"{r['nprobe']:>8}{r['recall']:>12.4f}{r['ms_per_query']:>11.3f}{r['qps']:>10.1f}")


# ------------------------------------------------------------- (D) analytic
def layer_macs(d: int, hid: int, t: int, kv: int) -> int:
    """MACs for one Transformer layer over t tokens attending to kv keys."""
    return 4 * d * d * t + 2 * t * kv * d + 3 * d * hid * t


def analytic():
    d, L, hid, N, K, T, M, beams = 1024, 28, 2734, 199, 20, 30, 4, 20
    enc = layer_macs(d, hid, N, N)
    prefill_all = (L - 1) * enc
    dit_step = layer_macs(d, hid, K, N + 1)
    xrec = prefill_all + T * dit_step

    ar_prefill = L * enc
    ar_token = layer_macs(d, hid, 1, N + M)          # one SID token for one beam
    ar_decode = M * beams * ar_token                 # 80 serial steps
    sid_ar = ar_prefill + ar_decode

    full_denoise = L * enc + T * L * layer_macs(d, hid, K, N + 1)   # every step re-runs all L layers
    print(f"\n[D] analytic @ paper shapes d={d} L={L} N={N} K={K} T={T} SID={M} beams={beams}")
    print(f"{'path':<46}{'GMAC/request':>14}{'serial steps':>14}")
    for name, macs, ser in [("X-Rec (late interaction)", xrec, T),
                            ("X-Rec full-layer denoising", full_denoise, T),
                            ("SID-AR (beam search)", sid_ar, M * beams)]:
        print(f"{name:<46}{macs/1e9:>14.1f}{ser:>14d}")
    print(f"\nX-Rec vs SID-AR: MAC ratio {sid_ar/xrec:.2f}x, serial-depth ratio "
          f"{M*beams/T:.2f}x  -> decode is latency-bound, which is what the paper's 3.46x reflects")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg.isdigit():
        measured(int(arg))
    elif arg in ("all", "model"):
        for L in (3, 12):
            measured(L)
            print()
    if arg in ("all", "decode"):
        decode_compare()
    if arg in ("all", "ann"):
        ann_compare()
    if arg in ("all", "analytic"):
        analytic()
