"""Run the full comparison matrix (variants x seeds) and aggregate mean[min-max].

    python scripts/run_experiments.py --seeds 0 1 2 --eval-n 4096

Each cell is one `xrec_ref/train.py` run written to `runs/exp/<variant>_s<seed>.json`; the aggregate
goes to `runs/exp/summary.json` and is printed as a markdown table that the README quotes verbatim.
Nothing here is cached or invented: a variant that crashes leaves a hole that shows up as n<seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "exp")

VARIANTS: dict[str, list[str]] = {
    "xrec": [],
    "no_anchor": ["--no-anchor"],
    "no_rfm": ["--no-rfm"],
    "no_pretrain": ["--no-pretrain"],
    "no_all_prefix": ["--pretrain-mask", "last"],
    "u2i": ["--arch", "u2i"],
    "u2i_k": ["--arch", "u2i_k"],
    "sid_ar": ["--arch", "sid_ar"],
    "sid_ar_coarse": ["--arch", "sid_ar", "--sid-codebooks", "2", "--sid-size", "64"],
}
METRICS = ("recall_KxR", "recall_1x20", "mode_recall_KxR", "pairwise_cos", "gen_ms_per_request")


def run_cell(variant: str, seed: int, args) -> dict:
    out = os.path.join(OUT, f"{variant}_s{seed}.json")
    cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT, "xrec_ref", "train.py"),
           "--seed", str(seed), "--eval-seed", str(seed), "--eval-n", str(args.eval_n),
           "--out", out, *VARIANTS[variant]]
    if args.data:
        cmd += ["--data", args.data]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0:
        print(f"[{variant} s{seed}] FAILED rc={proc.returncode}\n{proc.stderr[-1200:]}", flush=True)
        return dict(variant=variant, seed=seed, ok=False, error=proc.stderr[-400:])
    blob = json.load(open(out, encoding="utf-8"))
    row = dict(variant=variant, seed=seed, ok=True, seconds=round(time.time() - t0, 1),
               params=blob.get("params"), extras=blob.get("extras"), **{
                   k: blob["KxR"][k] for k in METRICS})
    row["recall_KxR_1x20_cfg"] = blob["1x20"]["recall_1x20"]
    print(f"[{variant} s{seed}] recall@20x1={row['recall_KxR']:.4f} "
          f"recall@1x20={row['recall_1x20']:.4f} cos={row['pairwise_cos']:.3f} "
          f"({row['seconds']}s)", flush=True)
    return row


def aggregate(rows: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        a = agg.setdefault(r["variant"], {k: [] for k in METRICS})
        for k in METRICS:
            a[k].append(r[k])
    for d in agg.values():
        for k in METRICS:
            xs = d[k]
            d[k] = dict(mean=sum(xs) / len(xs), min=min(xs), max=max(xs), n=len(xs))
    return agg


def table(agg: dict) -> str:
    head = "| variant | Recall@20x1 | Recall@1x20 | mode recall | pairwise cos | ms/request | n |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for v, d in agg.items():
        m = d["recall_KxR"]
        one = d["recall_1x20"]
        mode = d["mode_recall_KxR"]
        cos = d["pairwise_cos"]
        ms = d["gen_ms_per_request"]
        lines.append(
            f"| {v} | {100*m['mean']:.2f}% [{100*m['min']:.2f}–{100*m['max']:.2f}] "
            f"| {100*one['mean']:.2f}% [{100*one['min']:.2f}–{100*one['max']:.2f}] "
            f"| {100*mode['mean']:.1f}% | {cos['mean']:.3f} | {ms['mean']:.2f} | {m['n']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--eval-n", type=int, default=4096)
    ap.add_argument("--data", default=None, help="ratings log; default = synthetic fixture")
    ap.add_argument("--out", default=os.path.join(OUT, "summary.json"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for seed in args.seeds:
        for v in args.variants:
            rows.append(run_cell(v, seed, args))
    agg = aggregate(rows)
    summary = dict(seeds=args.seeds, variants=args.variants, eval_n=args.eval_n, data=args.data,
                   cells=rows, aggregate=agg, table=table(agg))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + table(agg))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
