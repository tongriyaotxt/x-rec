"""Trivial baselines that any learned retriever must beat — the calibration this repo was missing.

Two numbers, both at the same candidate budget as Recall@(K x R):

  popularity@K          the K most frequent items in the training log, returned for every request
  oracle_mode_pop@K     the K most frequent items *within the user's true target mode* (upper bound
                        that still ignores everything else about the user)

If a model does not beat `popularity@K`, its personalisation is not paying for itself, and any claim of
the form "multi-trigger generation works" has to be restated as "it works relative to the other learned
models in this table".
"""
from __future__ import annotations

import torch


@torch.no_grad()
def item_frequency(train: dict, n_items: int) -> torch.Tensor:
    return torch.bincount(train["ctx"].flatten(), minlength=n_items).float()


@torch.no_grad()
def popularity_recall(train: dict, test: dict, item_mode: torch.Tensor, n_items: int,
                      K: int = 20) -> dict:
    freq = item_frequency(train, n_items)
    k = min(K, n_items)
    top = freq.topk(k).indices
    tgt = test["tgt"]
    global_hit = float(tgt[:, None].eq(top[None, :]).any(-1).float().mean())

    # oracle mode: condition on the *true* target mode but nothing else about the user
    hits, total = 0, tgt.numel()
    tgt_mode = test["tgt_mode"]
    for m in tgt_mode.unique().tolist():
        items_m = (item_mode == m).nonzero().flatten()
        if items_m.numel() == 0:
            continue
        top_m = items_m[freq[items_m].topk(min(k, items_m.numel())).indices]
        sel = tgt_mode == m
        hits += int(tgt[sel][:, None].eq(top_m[None, :]).any(-1).sum())
    return dict(K=int(k), popularity_K=global_hit, oracle_mode_pop_K=hits / max(1, total))


@torch.no_grad()
def random_recall(n_items: int, K: int = 20) -> float:
    """Expected recall of K distinct uniform draws: the floor any model must clear."""
    return min(1.0, K / max(1, n_items))


@torch.no_grad()
def retrieval_profile(triggers: torch.Tensor, item_emb: torch.Tensor, freq: torch.Tensor,
                      R: int = 1, chunk: int = 64) -> dict:
    """Where in the popularity distribution do the retrieved items sit?

    A recommender can score above its own weight simply by reciting the marginal. `mean_percentile`
    (~1.0 = always head items, 0.5 = popularity-blind) and `head20_share` (fraction of returned
    candidates that are among the 20 most frequent training items) separate that from real
    personalisation.
    """
    n_items = item_emb.shape[0]
    order = freq.argsort(descending=True)
    rank = torch.empty(n_items, dtype=torch.float)
    rank[order] = torch.arange(n_items, dtype=torch.float) / max(1, n_items - 1)
    percentile = 1.0 - rank                                  # 1.0 = most popular
    head20 = set(order[:20].tolist())

    perc, head_hits, total, distinct = 0.0, 0, 0, set()
    for i in range(0, triggers.shape[0], chunk):
        sim = torch.einsum("bkd,vd->bkv", triggers[i:i + chunk], item_emb)
        cand = sim.topk(R, dim=-1).indices.reshape(triggers[i:i + chunk].shape[0], -1)
        perc += float(percentile[cand].sum())
        head_hits += sum(1 for c in cand.flatten().tolist() if c in head20)
        total += cand.numel()
        distinct.update(cand.flatten().tolist())
    return dict(mean_percentile=perc / max(1, total), head20_share=head_hits / max(1, total),
                distinct_items=len(distinct), candidates=total)

