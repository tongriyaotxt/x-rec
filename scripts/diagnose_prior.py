"""Is a variant winning on personalisation, or just reciting the popularity marginal?

Trains X-Rec and the SID-AR control with the repo's standard recipe, then profiles the *retrieved items*:

    mean_percentile   1.0 = every candidate is a head item, 0.5 = popularity-blind
    head20_share      fraction of candidates among the 20 most frequent training items
    distinct_items    how many distinct items the K triggers ever reach

Run: python scripts/diagnose_prior.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "xrec_ref"))

from baselines import item_frequency, popularity_recall, retrieval_profile  # noqa: E402
from config import CFG  # noqa: E402
from data import build_world, make_split  # noqa: E402
from train import build_model, evaluate, run_stage  # noqa: E402


def main():
    torch.manual_seed(0)
    cfg = CFG()
    world = build_world(cfg, seed=cfg.seed, zipf_exp=0.5)
    train = make_split(cfg, world, n=cfg.batch * 12, seed=2)
    test = make_split(cfg, world, n=4096, seed=4)
    freq = item_frequency(train, cfg.n_items)

    base = popularity_recall(train, test, world["mode"], cfg.n_items, K=cfg.K)
    print(f"fixture: |I|={cfg.n_items} popularity@20={base['popularity_K']:.4f} "
          f"oracle-mode@20={base['oracle_mode_pop_K']:.4f}")

    rows = [dict(variant="popularity@20", recall=base["popularity_K"], mean_percentile=1.0,
                 head20_share=1.0, distinct_items=20)]
    for arch in ("xrec", "sid_ar"):
        cfg.arch = arch
        model, extras = build_model(cfg, world)
        if cfg.use_pretrain and arch == "xrec":
            run_stage(model, train, cfg, cfg.steps_pre, cfg.lr_pre, f"{arch}-pre", torch.device("cpu"),
                      log_every=1000, loss_fn=model.loss_all_prefix)
        run_stage(model, train, cfg, cfg.steps_sft, cfg.lr_sft, f"{arch}-sft", torch.device("cpu"),
                  log_every=150)
        metrics, trig = evaluate(model, test, cfg, world, torch.device("cpu"), seed=0)
        prof = retrieval_profile(trig, model.item_emb, freq, R=1)
        rows.append(dict(variant=arch, recall=metrics["recall_KxR"], **prof, extras=extras))
        print(f"{arch:<12} recall@20x1={metrics['recall_KxR']:.4f} "
              f"mean_percentile={prof['mean_percentile']:.3f} head20_share={prof['head20_share']:.3f} "
              f"distinct={prof['distinct_items']} {extras}")

    out = os.path.join(ROOT, "runs", "prior_profile.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(baselines=base, rows=rows), f, indent=2, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
