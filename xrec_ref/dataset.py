"""Real-interaction data path (paper §3.1 / §4.2) — the piece that separates a demo from a reproduction.

`data.py` invents a 6k-item world so the mechanism is observable; this module consumes an actual
user-item-timestamp log and rebuilds the same `world` / `split` dicts that `train.py` expects:

    ratings log ──► PPMI co-occurrence ──► truncated SVD ──► L2-normalised item vectors
                └─► k-means ──► anchors (= coarse semantic cluster index x^(a))
                └─► per-user time-ordered histories ──► leave-one-out splits

Supported inputs (auto-detected per line):
  * MovieLens-1M     `ratings.dat`    `user::item::rating::timestamp`
  * MovieLens-100K   `u.data`         `user\titem\trating\ttimestamp`
  * generic          `.csv`/`.tsv`    `user,item[,rating][,timestamp]` (header optional)

No network access is required: `write_fixture()` emits a synthetic log in the *same on-disk format*, so the
loader is exercised end-to-end in CI. Pointing `--data` at a real `ratings.dat` is a one-flag change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from index import kmeans
from rfm import normalize


# ------------------------------------------------------------------ container
@dataclass
class Ratings:
    user: torch.Tensor      # (n,) int64, 0-based contiguous
    item: torch.Tensor      # (n,) int64, 0-based contiguous
    ts: torch.Tensor        # (n,) int64, seconds
    rating: torch.Tensor    # (n,) float

    def __len__(self) -> int:
        return int(self.user.numel())

    @property
    def n_users(self) -> int:
        return int(self.user.max()) + 1 if len(self) else 0

    @property
    def n_items(self) -> int:
        return int(self.item.max()) + 1 if len(self) else 0


# ---------------------------------------------------------------------- input
def _split_line(line: str) -> list[str]:
    if "::" in line:
        return line.split("::")
    if "\t" in line:
        return line.split("\t")
    if "," in line:
        return line.split(",")
    return line.split()


def read_ratings(path: str, max_rows: int | None = None) -> Ratings:
    """Parse any of the three formats above; ids are remapped to 0..n-1 in first-seen order."""
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = _split_line(line)
            if not parts[0].replace(".", "", 1).isdigit():        # header
                continue
            if len(parts) < 2:
                continue
            rows.append(parts)
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"no interactions parsed from {path}")

    def col(j: int, default: float) -> list[float]:
        return [float(r[j]) if len(r) > j and r[j] not in ("", "NA") else default for r in rows]

    users_raw, items_raw = [r[0] for r in rows], [r[1] for r in rows]
    umap: dict[str, int] = {}
    imap: dict[str, int] = {}
    user = torch.tensor([umap.setdefault(u, len(umap)) for u in users_raw], dtype=torch.long)
    item = torch.tensor([imap.setdefault(i, len(imap)) for i in items_raw], dtype=torch.long)
    rating = torch.tensor(col(2, 1.0), dtype=torch.float)
    ts = torch.tensor([float(r[3]) if len(r) > 3 else float(k) for k, r in enumerate(rows)],
                      dtype=torch.float).long()
    return Ratings(user=user, item=item, ts=ts, rating=rating)


def write_fixture(path: str, n_users: int = 1500, n_items: int = 900, n_modes: int = 8,
                  seed: int = 0, fmt: str = "ml-1m", min_len: int = 8, max_len: int = 60) -> str:
    """Synthetic log in MovieLens format: 2 interest modes per user, Zipf popularity, monotone time.

    Deliberately *not* a benchmark — it exists so the real-data pipeline (parse -> PPMI/SVD -> anchors ->
    leave-one-out) is covered by tests without network access. Numbers from it are not comparable to
    numbers from a real log, and the docs say so.
    """
    g = torch.Generator().manual_seed(seed)
    mode_of_item = torch.randint(0, n_modes, (n_items,), generator=g)
    pools = [torch.nonzero(mode_of_item == m).flatten() for m in range(n_modes)]
    order = torch.randperm(n_items, generator=g)                       # popularity order
    rank = torch.empty(n_items, dtype=torch.float)
    rank[order] = torch.arange(n_items, dtype=torch.float)
    pop = (1.0 / (1.0 + rank)).pow(0.8)

    lines, ts = [], 1_000_000
    used: set[int] = set()
    first_ts: dict[int, int] = {}
    for u in range(n_users):
        m1 = int(torch.randint(0, n_modes, (1,), generator=g))
        m2 = int((m1 + 1 + torch.randint(0, n_modes - 1, (1,), generator=g)) % n_modes)
        n = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        cur = m1
        for _ in range(n):
            if float(torch.rand(1, generator=g)) > 0.85:
                cur = m2 if cur == m1 else m1
            pool = pools[cur]
            w = pop[pool]
            pick = int(pool[torch.multinomial(w, 1, generator=g)])
            ts += int(torch.randint(1, 4000, (1,), generator=g))
            first_ts.setdefault(u, ts)
            used.add(pick)
            rating = int(torch.randint(1, 6, (1,), generator=g))
            lines.append(f"{u + 1}::{pick + 1}::{rating}::{ts}")

    # every catalog item must appear, otherwise the loader's remapping shrinks |I| and downstream
    # shapes become data-dependent. Cold items enter as the user's first interaction (never a target).
    for it in sorted(set(range(n_items)) - used):
        u = it % n_users
        lines.append(f"{u + 1}::{it + 1}::3::{first_ts.get(u, 1_000_000) - 1}")
    path = path if path.endswith((".dat", ".csv", ".txt")) else path + ".dat"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ----------------------------------------------------------------- embeddings
def _ordered_indices(r: Ratings) -> dict[int, list[int]]:
    """Per-user interaction indices in time order; consecutive repeats of one item collapse to one.

    Single source of truth for "the user's history" — the embedding builder and the split builder both
    go through it, so a duplicated event cannot mean two different things in two places.
    """
    key = r.user * (int(r.ts.max()) + 1) + r.ts
    out: dict[int, list[int]] = {}
    for k in torch.argsort(key, stable=True).tolist():
        u, it = int(r.user[k]), int(r.item[k])
        seq = out.setdefault(u, [])
        if not seq or int(r.item[seq[-1]]) != it:
            seq.append(k)
    return out


def _histories(r: Ratings) -> list[list[int]]:
    return [[int(r.item[k]) for k in ks] for ks in _ordered_indices(r).values()]


def item_embeddings(r: Ratings, d: int = 128, window: int = 5, seed: int = 0) -> torch.Tensor:
    """PPMI co-occurrence over sliding windows -> truncated SVD -> unit vectors.

    This is the standard GRU4Rec/Item2Vec-style item representation, chosen because it needs no training
    and stays deterministic. The paper's items arrive pre-embedded from the content tower; swapping this
    function for "load the encoder's output" is the intended production path.
    """
    n_items = r.n_items
    hist = _histories(r)
    ii, jj = [], []
    for seq in hist:
        for a in range(len(seq)):
            for b in range(a + 1, min(a + 1 + window, len(seq))):
                ii.append(seq[a])
                jj.append(seq[b])
                ii.append(seq[b])                                        # symmetric
                jj.append(seq[a])
    if not ii:
        return normalize(torch.randn(n_items, d, generator=torch.Generator().manual_seed(seed)))
    C = torch.zeros(n_items, n_items)
    C.index_put_((torch.tensor(ii), torch.tensor(jj)), torch.ones(len(ii)), accumulate=True)

    total = C.sum().clamp_min(1.0)
    row, colm = C.sum(1, keepdim=True), C.sum(0, keepdim=True)
    with torch.no_grad():
        pmi = torch.log((C * total) / (row @ colm).clamp_min(1e-12) + 1e-12)
        ppmi = pmi.clamp_min(0.0)
    q = min(d + 16, min(ppmi.shape) - 1)
    torch.manual_seed(seed)
    u, s, _ = torch.svd_lowrank(ppmi, q=q, niter=4)
    emb = u[:, :d] * s[:d].sqrt()[None, :]
    pad = d - emb.shape[1]
    if pad > 0:                                                          # tiny catalog: pad with noise
        emb = torch.cat([emb, torch.zeros(emb.shape[0], pad)], dim=1)
    return normalize(emb)


# ---------------------------------------------------------------------- world
def build_world_from_ratings(cfg, r: Ratings, d: int | None = None, seed: int = 0) -> dict:
    d = d or cfg.d_e
    emb = item_embeddings(r, d=d, seed=seed)
    n_anchor = min(cfg.n_anchor, max(2, r.n_items // 4))
    cent, anchor_of_item = kmeans(emb, n_anchor, iters=15, seed=seed)
    anchor_of_item = anchor_of_item.long()
    same, diff = _anchor_cos(emb, anchor_of_item, seed=seed)
    return dict(emb=emb, anchor_of_item=anchor_of_item, mode=anchor_of_item.clone(),
                anchor_acc=None, intra_mode_cos=(same, diff), n_anchor=n_anchor,
                n_items=r.n_items, n_modes=n_anchor, source="interactions")


def _anchor_cos(emb: torch.Tensor, anchor: torch.Tensor, seed: int = 0, n: int = 4000):
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, emb.shape[0], (n,), generator=g)
    j = torch.randint(0, emb.shape[0], (n,), generator=g)
    sim = (emb[i] * emb[j]).sum(-1)
    same, diff = anchor[i] == anchor[j], anchor[i] != anchor[j]
    return (float(sim[same].mean()) if bool(same.any()) else 0.0,
            float(sim[diff].mean()) if bool(diff.any()) else 0.0)


# --------------------------------------------------------------------- splits
def _attr_buckets(r: Ratings, n_action: int, n_duration: int):
    """rating -> action bucket, inter-event gap -> duration bucket. Real logs carry no action/duration
    fields, so the paper's two sub-attributes are proxied by engagement strength and dwell gap."""
    act = (r.rating.clamp(1, 5) - 1).long().clamp_max(n_action - 1)
    gap = torch.zeros_like(r.ts, dtype=torch.float)
    order = torch.argsort(r.user * (int(r.ts.max()) + 1) + r.ts, stable=True)
    prev = torch.full((r.n_users,), -1, dtype=torch.long)
    for k in order.tolist():
        u = int(r.user[k])
        gap[k] = 0.0 if prev[u] < 0 else float(r.ts[k] - prev[u])
        prev[u] = int(r.ts[k])
    edges = torch.quantile(gap.float(), torch.linspace(0, 1, n_duration + 1)[1:-1])
    dur = torch.bucketize(gap, edges).clamp_max(n_duration - 1).long()
    return act, dur


def make_splits(cfg, r: Ratings, world: dict, seed: int = 0) -> dict[str, dict]:
    """leave-one-out: test = last interaction, valid = second-to-last, train = random cut per user."""
    act_all, dur_all = _attr_buckets(r, cfg.n_action, cfg.n_duration)
    ordered = _ordered_indices(r)
    g = torch.Generator().manual_seed(seed)
    L = cfg.seq_len

    def blank() -> dict[str, list]:
        return dict(ctx=[], act=[], dur=[], tgt=[], tgt_act=[], tgt_dur=[], tgt_mode=[],
                    active_modes=[], user=[])

    out = {"train": blank(), "valid": blank(), "test": blank()}
    anchor = world["anchor_of_item"]
    for u, idxs in ordered.items():
        if len(idxs) < L + 2:
            continue
        seq = [int(r.item[k]) for k in idxs]
        acts = [act_all[k] for k in idxs]
        durs = [dur_all[k] for k in idxs]
        hist_anchor = anchor[torch.tensor(seq)]
        top2 = torch.bincount(hist_anchor, minlength=anchor.max() + 1).topk(2).indices
        for name, cut in (("test", len(seq) - 1), ("valid", len(seq) - 2),
                          ("train", int(torch.randint(L + 1, len(seq), (1,), generator=g)))):
            rec = out[name]
            rec["ctx"].append(seq[cut - L:cut])
            rec["act"].append(acts[cut - L:cut])
            rec["dur"].append(durs[cut - L:cut])
            rec["tgt"].append(seq[cut])
            rec["tgt_act"].append(acts[cut])
            rec["tgt_dur"].append(durs[cut])
            rec["tgt_mode"].append(int(anchor[seq[cut]]))
            rec["active_modes"].append(top2.tolist())
            rec["user"].append(u)

    def pack(rec: dict) -> dict:
        if not rec["ctx"]:
            raise ValueError("no user has enough history; lower --seq-len or use a denser log")
        return dict(ctx=torch.tensor(rec["ctx"]), act=torch.tensor(rec["act"]),
                    dur=torch.tensor(rec["dur"]), tgt=torch.tensor(rec["tgt"]),
                    tgt_act=torch.tensor(rec["tgt_act"]), tgt_dur=torch.tensor(rec["tgt_dur"]),
                    tgt_mode=torch.tensor(rec["tgt_mode"]),
                    active_modes=torch.tensor(rec["active_modes"]), user=torch.tensor(rec["user"]))

    return {k: pack(v) for k, v in out.items()}


def load_file(cfg, path: str, seed: int = 0, max_rows: int | None = None):
    """One call from a path to everything train.py needs."""
    r = read_ratings(path, max_rows=max_rows)
    cfg.n_items = r.n_items
    world = build_world_from_ratings(cfg, r, seed=seed)
    cfg.n_anchor = world["n_anchor"]
    cfg.n_modes = world["n_anchor"]
    splits = make_splits(cfg, r, world, seed=seed)
    info = dict(path=path, n_interactions=len(r), n_users=r.n_users, n_items=r.n_items,
                n_anchors=world["n_anchor"], rows={k: int(v["ctx"].shape[0]) for k, v in splits.items()},
                intra_mode_cos=world["intra_mode_cos"], density=len(r) / (r.n_users * r.n_items))
    return world, splits, info


def stats_of(r: Ratings) -> dict:
    hist = _histories(r)
    lens = torch.tensor([len(h) for h in hist], dtype=torch.float)
    return dict(interactions=len(r), users=r.n_users, items=r.n_items,
                density=len(r) / (r.n_users * r.n_items),
                len_mean=float(lens.mean()), len_p50=float(lens.median()),
                len_min=int(lens.min()), len_max=int(lens.max()),
                rating_mean=float(r.rating.mean()), ts_span_days=float(r.ts.max() - r.ts.min()) / 86400)


if __name__ == "__main__":                                             # python xrec_ref/dataset.py
    import argparse
    import json
    import tempfile

    ap = argparse.ArgumentParser(description="profile a ratings log, or emit a synthetic one")
    ap.add_argument("path", nargs="?", default=None, help="ratings file to profile")
    ap.add_argument("--fixture", default=None, help="write a synthetic log in MovieLens format here")
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--items", type=int, default=900)
    ap.add_argument("--min-len", type=int, default=8, help="interactions per user, lower bound")
    ap.add_argument("--max-len", type=int, default=60, help="interactions per user, upper bound")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.fixture:
        print(write_fixture(a.fixture, n_users=a.users, n_items=a.items, seed=a.seed,
                            min_len=a.min_len, max_len=a.max_len))
    if a.path:
        r = read_ratings(a.path)
        print(json.dumps(stats_of(r), indent=2))
    elif not a.fixture:
        p = write_fixture(tempfile.mkdtemp() + "/fixture.dat")
        r = read_ratings(p)
        print(json.dumps(stats_of(r), indent=2))
        print(json.dumps({"ppmi_d": list(item_embeddings(r, d=32).shape)}, indent=2))
