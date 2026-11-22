"""Train X-Rec (pretrain -> streaming SFT) and its controls; evaluate Recall@(K x R).

Usage
  python xrec_ref/train.py --out runs/xrec.json --sweep
  python xrec_ref/train.py --arch u2i --out runs/u2i.json
  python xrec_ref/train.py --arch sid_ar --out runs/sid_ar.json      # entity-level SID-AR control
  python xrec_ref/train.py --data data/ml-1m/ratings.dat --out runs/xrec_ml1m.json
  python xrec_ref/train.py --ckpt-in runs/xrec.pt --eval-only --out runs/xrec_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from config import CFG
from data import batches, build_world, make_split, mode_recall, pairwise_cosine, recall_at
from model import U2I, XRec
from models_sid import SIDAR
from sid import ResidualQuantizer


def build_model(cfg, world):
    """Returns (model, extras). `extras` carries the metrics that belong to the discretiser."""
    if cfg.arch == "xrec":
        return XRec(cfg, world["emb"], world["anchor_of_item"]), {}
    if cfg.arch == "u2i":
        return U2I(cfg, world["emb"], world["anchor_of_item"], n_interest=1), {}
    if cfg.arch == "u2i_k":
        return U2I(cfg, world["emb"], world["anchor_of_item"], n_interest=cfg.n_interest), {}
    if cfg.arch == "sid_ar":
        q = ResidualQuantizer(n_codebooks=cfg.sid_codebooks, codebook_size=cfg.sid_size, seed=cfg.seed)
        q.fit(world["emb"])
        codes = q.encode(world["emb"])
        # how many items share one SID: 1.0 means the discrete space is a unique identifier at this
        # catalog size, which is exactly the regime where an AR baseline stops being penalised
        uniq = len({tuple(c) for c in codes.tolist()}) / max(1, codes.shape[0])
        return SIDAR(cfg, world["emb"], world["anchor_of_item"], q, codes), dict(
            recon_cos=q.recon_cosine(world["emb"]), sid_levels=q.M, sid_vocab=q.K,
            sid_space=float(q.K) ** q.M, sid_unique_ratio=uniq)
    raise ValueError(cfg.arch)


def run_stage(model, split, cfg, steps, lr, tag, device, log_every=50, loss_fn=None):
    loss_fn = loss_fn or model.loss
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.wd)
    warm = max(1, min(200, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    model.train()
    t0, hist = time.time(), []
    for i, batch in enumerate(batches(split, cfg.batch, steps, seed=0)):
        batch = tuple(t.to(device) for t in batch)
        l_anchor, l_rfm = loss_fn(batch)
        loss = l_rfm + l_anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        if i % log_every == 0 or i == steps - 1:
            rec = dict(step=i, anchor=float(l_anchor.detach()), rfm=float(l_rfm.detach()),
                       total=float(loss.detach()))
            hist.append(rec)
            print(f"[{tag}] step {i:4d}  total={rec['total']:.4f}  anchor={rec['anchor']:.4f} "
                  f"rfm={rec['rfm']:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return hist


@torch.no_grad()
def evaluate(model, split, cfg, world, device, n=None, K=None, R=None, denoise=None, temp=None, seed=0):
    """Samples in chunks: the late-interaction cache is replicated K times per request, so a 4096-row
    eval batch would materialise ~2.5 GB of K/V per denoising step and thrash."""
    model.eval()
    n = min(n or cfg.n_eval, split["ctx"].shape[0])
    K = K or cfg.K
    idx = torch.arange(n, device=device)
    ctx = split["ctx"].to(device)[idx]
    act = split["act"].to(device)[idx]
    dur = split["dur"].to(device)[idx]
    tgt = split["tgt"].to(device)[idx]
    chunk = max(1, min(cfg.n_eval_chunk, n))
    gen = torch.Generator(device=device).manual_seed(seed)
    t0, parts = time.perf_counter(), []
    for i in range(0, n, chunk):
        parts.append(model.sample(ctx[i:i + chunk], act[i:i + chunk], dur[i:i + chunk],
                                  K=K, steps=denoise, anchor_temp=temp, gen=gen))   # (c,K,d_e)
    trig = torch.cat(parts)
    gen_ms = (time.perf_counter() - t0) * 1e3
    table = model.item_emb                                                              # frozen ANN source
    r = cfg.R if R is None else R
    return dict(
        recall_KxR=recall_at(trig, tgt, table, R=r),
        recall_1x20=recall_at(trig[:, :1], tgt, table, R=20),
        mode_recall_KxR=mode_recall(trig, split["active_modes"].to(device)[idx],
                                    world["mode"].to(device), table, R=r),
        pairwise_cos=pairwise_cosine(trig),
        norm_dev=float((trig.norm(dim=-1) - 1).abs().max()),
        gen_ms_per_request=gen_ms / max(1, n),
    ), trig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="xrec", choices=["xrec", "u2i", "u2i_k", "sid_ar"])
    ap.add_argument("--data", default=None,
                    help="ratings log (ml-1m/ml-100k/csv); omit for the synthetic fixture")
    ap.add_argument("--max-rows", type=int, default=None, help="cap parsed interactions (debug)")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--no-rfm", action="store_true")
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--pretrain-mask", default=None, choices=["all", "last"])
    ap.add_argument("--sid-codebooks", type=int, default=4)
    ap.add_argument("--sid-size", type=int, default=256)
    ap.add_argument("--steps-pre", type=int, default=None)
    ap.add_argument("--steps-sft", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--denoise", type=int, default=None)
    ap.add_argument("--anchor-temp", type=float, default=None)
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--zipf", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1")
    ap.add_argument("--ckpt-out", default=None)
    ap.add_argument("--ckpt-in", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default="runs/xrec.json")
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))

    cfg = CFG()
    cfg.arch = args.arch
    cfg.use_anchor = not args.no_anchor
    cfg.use_rfm = not args.no_rfm
    cfg.use_pretrain = not args.no_pretrain
    cfg.sid_codebooks, cfg.sid_size = args.sid_codebooks, args.sid_size
    for name, val in [("steps_pre", args.steps_pre), ("steps_sft", args.steps_sft), ("K", args.K),
                      ("R", args.R), ("steps", args.denoise), ("anchor_temp", args.anchor_temp),
                      ("n_eval", args.eval_n), ("pretrain_mask", args.pretrain_mask)]:
        if val is not None:
            setattr(cfg, name, val)

    t0 = time.time()
    ds_info = {}
    if args.data:
        from dataset import load_file
        world, splits, ds_info = load_file(cfg, args.data, seed=cfg.seed, max_rows=args.max_rows)
        pre, train, test = splits["train"], splits["train"], splits["test"]
        print(f"data={args.data} |I|={ds_info['n_items']} users={ds_info['n_users']} "
              f"interactions={ds_info['n_interactions']} anchors={ds_info['n_anchors']} "
              f"rows={ds_info['rows']}", flush=True)
    else:
        world = build_world(cfg, seed=cfg.seed, zipf_exp=args.zipf)
        pre = make_split(cfg, world, n=cfg.batch * 10, seed=3)
        train = make_split(cfg, world, n=cfg.batch * 12, seed=2)
        test = make_split(cfg, world, n=max(cfg.n_eval, 64), seed=4)
    acc = world["anchor_acc"]
    acc_s = "n/a" if acc is None else f"{acc:.3f}"
    print(f"device={device.type} world |X|={cfg.n_items} d_e={cfg.d_e} anchors={cfg.n_anchor} "
          f"modes={cfg.n_modes} anchor_acc={acc_s} "
          f"cos(same mode)={world['intra_mode_cos'][0]:.3f} cos(diff mode)={world['intra_mode_cos'][1]:.3f}",
          flush=True)

    model, extras = build_model(cfg, world)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"arch={cfg.arch} params={n_par/1e6:.2f}M anchor={cfg.use_anchor} rfm={cfg.use_rfm} "
          f"pretrain={cfg.use_pretrain}/{cfg.pretrain_mask} extras={extras}", flush=True)

    res = dict(config=cfg.to_dict(), arch=cfg.arch, params=n_par, device=str(device),
               anchor_acc=acc, intra_mode_cos=world["intra_mode_cos"], dataset=ds_info,
               extras=extras, hist={})

    if args.eval_only:
        if not args.ckpt_in:
            raise SystemExit("--eval-only requires --ckpt-in")
        model.load_state_dict(torch.load(args.ckpt_in, map_location=device)["state_dict"])
        print(f"loaded {args.ckpt_in}", flush=True)
    else:
        if cfg.arch == "xrec" and cfg.use_pretrain:
            loss_fn = model.loss_all_prefix if cfg.pretrain_mask == "all" else model.loss
            res["hist"]["pretrain"] = run_stage(model, pre, cfg, cfg.steps_pre, cfg.lr_pre, "pre",
                                                device, loss_fn=loss_fn)
        res["hist"]["sft"] = run_stage(model, train, cfg, cfg.steps_sft, cfg.lr_sft, "sft", device)
        if args.ckpt_out:
            os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)
            torch.save(dict(cfg=cfg.to_dict(), arch=cfg.arch, state_dict=model.state_dict(),
                            item_emb=world["emb"], anchor_of_item=world["anchor_of_item"],
                            dataset=ds_info), args.ckpt_out)
            print(f"saved {args.ckpt_out}", flush=True)
    res["train_seconds"] = time.time() - t0

    from baselines import popularity_recall, random_recall
    res["baselines"] = popularity_recall(train, test, world["mode"], cfg.n_items, K=cfg.K * (cfg.R or 1))
    res["baselines"]["uniform_KxR"] = random_recall(cfg.n_items, cfg.K * (cfg.R or 1))
    print(f"[baselines] popularity@{res['baselines']['K']}={res['baselines']['popularity_K']:.4f}  "
          f"oracle_mode_pop={res['baselines']['oracle_mode_pop_K']:.4f}  "
          f"uniform={res['baselines']['uniform_KxR']:.4f}", flush=True)

    for tag, kw in [("KxR", {}), ("1x20", dict(K=1, R=20)),
                    ("5x4", dict(K=5, R=4)), ("10x2", dict(K=10, R=2))]:
        m, _ = evaluate(model, test, cfg, world, device, seed=args.eval_seed, **kw)
        res[tag] = m
        print(f"[eval {tag}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                           for k, v in m.items()), flush=True)

    if args.sweep:
        res["sweep"] = {}
        for temp in (0.8, 1.0, 1.2, 1.4, 1.8):
            ms, _ = evaluate(model, test, cfg, world, device, temp=temp, seed=args.eval_seed)
            res["sweep"][f"T={temp}"] = ms
            print(f"[sweep anchor_temp={temp}] recall={ms['recall_KxR']:.4f} "
                  f"mode_recall={ms['mode_recall_KxR']:.4f} cos={ms['pairwise_cos']:.4f}", flush=True)

    res["total_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out} ({res['total_seconds']:.1f}s total)")


if __name__ == "__main__":
    main()
