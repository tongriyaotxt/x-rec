"""Synthetic multi-modal stream mirroring the paper's setting.

Each user has two active interest modes (semantic clusters) in a 128-d unit-sphere item space; the
next positive item usually stays in the last mode but with p=0.25 jumps to the user's other mode.
That jump is what makes a single averaged U2I vector insufficient and K stochastic triggers useful.

Anchors = K-means centroids of the item embeddings: the analogue of the paper's coarse semantic
cluster index x^(a). `anchor_acc` reports how well an anchor identifies the generating mode.
"""
from __future__ import annotations

import torch

from rfm import normalize


def build_world(cfg, seed: int = 0, zipf_exp: float = 1.0):
    """Item embedding geometry deliberately encodes popularity: rank-r item in a mode sits at
    distance sigma(r) from the mode centre, sigma increasing with rank. Head items are therefore the
    nearest neighbours of a mode-centred trigger — the structure real video embeddings exhibit."""
    g = torch.Generator().manual_seed(seed)
    centers = normalize(torch.randn(cfg.n_modes, cfg.d_e, generator=g))
    mode = torch.randint(0, cfg.n_modes, (cfg.n_items,), generator=g)
    noise = normalize(torch.randn(cfg.n_items, cfg.d_e, generator=g))

    pools = []
    rank = torch.empty(cfg.n_items, dtype=torch.long)
    for m in range(cfg.n_modes):
        idx = torch.nonzero(mode == m).flatten()
        idx = idx[torch.randperm(idx.numel(), generator=g)]      # arbitrary popularity order
        pools.append(idx)
        rank[idx] = torch.arange(idx.numel())
    P = min(p.numel() for p in pools)
    pools = torch.stack([p[:P] for p in pools])                  # (n_modes, P), column 0 = head
    sigma = 0.12 + 0.65 * (rank.float() / rank.max()).pow(0.8)
    emb = normalize(centers[mode] + sigma[:, None] * noise)

    weights = 1.0 / (1.0 + torch.arange(P).float()).pow(zipf_exp)  # popularity ∝ 1/rank^s
    weights = weights / weights.sum()

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=cfg.n_anchor, n_init=3, random_state=seed).fit(emb.numpy())
    anchor_of_item = torch.from_numpy(km.labels_).long()
    # how pure each anchor is w.r.t. the generating mode
    maj = torch.zeros(cfg.n_anchor, dtype=torch.long)
    for a in range(cfg.n_anchor):
        sel = mode[anchor_of_item == a]
        if sel.numel():
            maj[a] = torch.bincount(sel, minlength=cfg.n_modes).argmax()
    anchor_acc = float((maj[anchor_of_item] == mode).float().mean())
    same_mode_cos = _intra_mode_cos(emb, mode)
    return dict(emb=emb, mode=mode, pools=pools, anchor_of_item=anchor_of_item, weights=weights,
                anchor_acc=anchor_acc, intra_mode_cos=same_mode_cos)


def _intra_mode_cos(emb, mode, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, emb.shape[0], (n,), generator=g)
    j = torch.randint(0, emb.shape[0], (n,), generator=g)
    same = mode[i] == mode[j]
    sim = (emb[i] * emb[j]).sum(-1)
    return float(sim[same].mean()), float(sim[~same].mean())


def make_split(cfg, world, n: int, seed: int = 1, p_stay: float = 0.85, p_same_mode: float = 0.75):
    """Vectorized over users: two-state Markov chain over the user's two interest modes."""
    g = torch.Generator().manual_seed(seed)
    N, M = cfg.seq_len, cfg.n_modes
    pools = world["pools"]
    P = pools.shape[1]

    m1 = torch.randint(0, M, (n,), generator=g)
    off = torch.randint(1, M, (n,), generator=g)
    m2 = (m1 + off) % M

    cur = m1.clone()
    items = torch.empty(n, N, dtype=torch.long)
    mode_seq = torch.empty(n, N, dtype=torch.long)
    w = world["weights"].expand(n, P)
    for pos in range(N):
        switch = torch.rand(n, generator=g) > p_stay
        cur = torch.where(switch, torch.where(cur == m1, m2, m1), cur)
        pick = torch.multinomial(w, 1, replacement=True, generator=g)[:, 0]
        items[:, pos] = pools[cur, pick]
        mode_seq[:, pos] = cur

    act = torch.randint(0, cfg.n_action, (n, N), generator=g)
    dur = torch.randint(0, cfg.n_duration, (n, N), generator=g)

    last = mode_seq[:, -1]
    stay = torch.rand(n, generator=g) < p_same_mode
    other = torch.where(last == m1, m2, m1)
    tgt_mode = torch.where(stay, last, other)
    tgt = pools[tgt_mode, torch.multinomial(w, 1, replacement=True, generator=g)[:, 0]]
    tgt_act = torch.randint(0, cfg.n_action, (n,), generator=g)
    tgt_dur = torch.randint(0, cfg.n_duration, (n,), generator=g)
    active = torch.stack([m1, m2], 1)          # (n,2) modes the user actually consumes
    return dict(ctx=items, act=act, dur=dur, tgt=tgt, tgt_act=tgt_act, tgt_dur=tgt_dur,
                tgt_mode=tgt_mode, active_modes=active)


def batches(split, batch: int, steps: int, seed: int = 0, shuffle: bool = True):
    n = split["ctx"].shape[0]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    for s in range(steps):
        start = (s * batch) % n
        idx = order[start:start + batch]
        if idx.numel() < batch:
            idx = torch.cat([idx, order[: batch - idx.numel()]])
        yield (split["ctx"][idx], split["act"][idx], split["dur"][idx],
               split["tgt"][idx], split["tgt_act"][idx], split["tgt_dur"][idx])


@torch.no_grad()
def recall_at(triggers: torch.Tensor, tgt: torch.Tensor, item_emb: torch.Tensor,
              R: int = 1, chunk: int = 64) -> float:
    """eq.33 — Recall@(K x R): target in the union of per-trigger top-R ANN results."""
    hits = 0
    for i in range(0, triggers.shape[0], chunk):
        sim = torch.einsum("bkd,vd->bkv", triggers[i:i + chunk], item_emb)
        cand = sim.topk(R, dim=-1).indices.reshape(triggers[i:i + chunk].shape[0], -1)
        hits += int((cand == tgt[i:i + chunk, None]).any(-1).sum())
    return hits / triggers.shape[0]


@torch.no_grad()
def mode_recall(triggers: torch.Tensor, active: torch.Tensor, item_mode: torch.Tensor,
                item_emb: torch.Tensor, R: int = 1, chunk: int = 64) -> float:
    """Diagnostic: did the retrieved candidates fall in an interest mode the user actually has?"""
    hits = 0
    for i in range(0, triggers.shape[0], chunk):
        sim = torch.einsum("bkd,vd->bkv", triggers[i:i + chunk], item_emb)
        cand = sim.topk(R, dim=-1).indices.reshape(triggers[i:i + chunk].shape[0], -1)
        m = item_mode[cand]                                  # (b, K*R)
        ok = (m[:, :, None] == active[i:i + chunk, None, :]).any(-1).any(-1)
        hits += int(ok.sum())
    return hits / triggers.shape[0]


@torch.no_grad()
def pairwise_cosine(triggers: torch.Tensor) -> float:
    k = triggers.shape[1]
    if k < 2:
        return 1.0
    x = normalize(triggers)
    sim = torch.einsum("bkd,bjd->bkj", x, x)
    return float(((sim.sum((-1, -2)) - k) / (k * (k - 1))).mean())


@torch.no_grad()
def dual_mode_hit(triggers: torch.Tensor, active: torch.Tensor, item_mode: torch.Tensor,
                  item_emb: torch.Tensor, R: int = 1, chunk: int = 64) -> float:
    """Does the candidate set touch *both* of the user's active interest modes?

    Recall@(K x R) only counts the hit item, so K diverse triggers and K copies of one trigger score the
    same. This is the metric that pays for diversity: with 20 candidates, a user with two interests should
    get both. It is the quantity the paper's "multi-trigger generation" claim is actually about.
    """
    hits = 0
    for i in range(0, triggers.shape[0], chunk):
        sim = torch.einsum("bkd,vd->bkv", triggers[i:i + chunk], item_emb)
        cand = sim.topk(R, dim=-1).indices.reshape(triggers[i:i + chunk].shape[0], -1)
        m = item_mode[cand]                                   # (b, K*R)
        act = active[i:i + chunk]                             # (b, 2)
        cov0 = (m == act[:, 0:1]).any(-1)
        cov1 = (m == act[:, 1:2]).any(-1)
        hits += int((cov0 & cov1).sum())
    return hits / triggers.shape[0]


@torch.no_grad()
def popularity_weights(train_ctx: torch.Tensor, n_items: int, alpha: float = 1.0,
                       gamma: float = 1.0) -> torch.Tensor:
    """Per-item training weight that puts probability mass back on the head.

    The fixture samples targets from a Zipf popularity; an unweighted loss treats a head item and a tail
    item as equally important, so a 1M-parameter model spends its capacity on tail items it will never be
    asked for. `alpha=0` reproduces the old uniform loss.
    """
    freq = torch.bincount(train_ctx.flatten(), minlength=n_items).float()
    logp = torch.log1p(freq)
    w = 1.0 + alpha * (logp / logp.max().clamp_min(1e-6)) ** gamma
    return w / w.mean()

