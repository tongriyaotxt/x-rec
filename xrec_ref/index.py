"""IVF-flat ANN index, written against torch only.

`faiss` is not a dependency of this repo (it is not installable in the offline build environment either),
so the retrieval stage of X-Rec is served by an inverted-file index with the same contract:

    train(centroids) -> add(vectors) -> search(q, k, nprobe) -> (scores, ids)

Layout: vectors are reordered by coarse-list id, so each list is a contiguous slice and probing is
`x[offsets[p]:offsets[p+1]]`. With `nprobe == nlist` the index degenerates to exact brute-force search,
which is what the tests assert against.

What this is *not*: no product quantization, no HNSW, no SIMD. The point is to expose the recall/QPS
trade-off that the paper's 10.23 vs 10.76 comparison hides — an exact matmul over 6k synthetic items is
free, an ANN over a billion items is not.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import torch


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def kmeans(x: torch.Tensor, n_clusters: int, iters: int = 20, seed: int = 0,
           chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain Lloyd k-means; deterministic for a given seed. Returns (centroids, assignment)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    init = torch.randperm(n, generator=g)[:n_clusters]
    cent = x[init].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        assign = torch.empty(n, dtype=torch.long)
        for i in range(0, n, chunk):
            sim = x[i:i + chunk] @ cent.T
            assign[i:i + chunk] = sim.argmax(-1)
        sums = torch.zeros_like(cent)
        sums.index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=n_clusters).clamp_min(1)[:, None].to(x.dtype)
        new = sums / counts
        # keep empty clusters where they are instead of collapsing them to the origin
        empty = torch.bincount(assign, minlength=n_clusters) == 0
        if bool(empty.any()):
            new[empty] = cent[empty]
        if torch.allclose(new, cent, atol=1e-6):
            cent = new
            break
        cent = new
    return cent, assign


@dataclass
class IVFIndex:
    d: int
    nlist: int = 64
    metric: str = "ip"
    seed: int = 0
    iters: int = 20
    centroids: torch.Tensor | None = None
    vectors: torch.Tensor | None = None          # (N, d) reordered by list id
    ids: torch.Tensor | None = None              # (N,)   original item id per slot
    offsets: torch.Tensor | None = None          # (nlist + 1,)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @torch.no_grad()
    def train(self, x: torch.Tensor) -> IVFIndex:
        x = l2norm(x.float())
        self.d = x.shape[1]
        self.nlist = min(self.nlist, x.shape[0])
        self.centroids, _ = kmeans(x, self.nlist, self.iters, self.seed)
        self.centroids = l2norm(self.centroids)
        return self

    @torch.no_grad()
    def add(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> IVFIndex:
        if self.centroids is None:
            self.train(x)
        x = l2norm(x.float())
        ids = torch.arange(x.shape[0]) if ids is None else ids.long()
        assign = (x @ self.centroids.T).argmax(-1)
        order = torch.argsort(assign, stable=True)
        self.vectors = x[order].contiguous()
        self.ids = ids[order].contiguous()
        counts = torch.bincount(assign, minlength=self.nlist)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).long()
        self.meta.update(n=x.shape[0], nlist=self.nlist, d=self.d, metric=self.metric)
        return self

    # ----------------------------------------------------------------- search
    @torch.no_grad()
    def search(self, q: torch.Tensor, k: int = 1, nprobe: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores, ids) each (nq, k), on the caller's device. nprobe >= nlist == exact search."""
        if self.vectors is None:
            raise RuntimeError("index is empty: call train()/add() first")
        dev = self.vectors.device
        q = l2norm(q.float().to(dev))
        nprobe = max(1, min(int(nprobe), self.nlist))
        out_s = torch.empty(q.shape[0], k)
        out_i = torch.empty(q.shape[0], k, dtype=torch.long)
        lists = (q @ self.centroids.T).topk(nprobe, dim=-1).indices
        for j in range(q.shape[0]):
            parts = [self.vectors[self.offsets[p]:self.offsets[p + 1]] for p in lists[j].tolist()]
            ids = [self.ids[self.offsets[p]:self.offsets[p + 1]] for p in lists[j].tolist()]
            cand, cid = torch.cat(parts), torch.cat(ids)
            if cand.numel() == 0:
                out_s[j] = -1e4
                out_i[j] = -1
                continue
            top = (cand @ q[j]).topk(min(k, cand.shape[0]))
            out_s[j, :top.values.numel()] = top.values
            out_i[j, :top.indices.numel()] = cid[top.indices]
            if top.values.numel() < k:                      # list shorter than k
                out_s[j, top.values.numel():] = -1e4
                out_i[j, top.values.numel():] = -1
        return out_s.to(q.device), out_i.to(q.device)

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> str:
        torch.save(dict(d=self.d, nlist=self.nlist, metric=self.metric, seed=self.seed,
                        iters=self.iters, centroids=self.centroids, vectors=self.vectors,
                        ids=self.ids, offsets=self.offsets, meta=self.meta), path)
        return path

    @classmethod
    def load(cls, path: str) -> IVFIndex:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        ix = cls(d=blob["d"], nlist=blob["nlist"], metric=blob["metric"], seed=blob["seed"],
                 iters=blob["iters"])
        for f in ("centroids", "vectors", "ids", "offsets", "meta"):
            setattr(ix, f, blob[f])
        return ix

    def __len__(self) -> int:
        return 0 if self.vectors is None else self.vectors.shape[0]

    def stats(self) -> dict:
        sizes = (self.offsets[1:] - self.offsets[:-1]) if self.offsets is not None else None
        return dict(n=len(self), nlist=self.nlist, d=self.d,
                    list_min=int(sizes.min()) if sizes is not None else 0,
                    list_max=int(sizes.max()) if sizes is not None else 0,
                    bytes=int(len(self) * self.d * 4))


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def brute_force_topk(q: torch.Tensor, x: torch.Tensor, k: int = 1, chunk: int = 512):
    q, x = l2norm(q.float()), l2norm(x.float())
    scores, ids = [], []
    for i in range(0, q.shape[0], chunk):
        s, ix = (q[i:i + chunk] @ x.T).topk(k, dim=-1)
        scores.append(s)
        ids.append(ix)
    return torch.cat(scores), torch.cat(ids)


@torch.no_grad()
def recall_at_k(index: IVFIndex, q: torch.Tensor, x: torch.Tensor, k: int = 10, nprobe: int = 1,
                truth: torch.Tensor | None = None) -> float:
    _, got = index.search(q, k=k, nprobe=nprobe)
    if truth is None:
        _, truth = brute_force_topk(q, x, k=k)
    hit = (got[:, :, None] == truth[:, None, :]).any(-1).float().mean()
    return float(hit)


@torch.no_grad()
def sweep(index: IVFIndex, q: torch.Tensor, x: torch.Tensor, k: int = 10,
          nprobes: tuple[int, ...] = (1, 2, 4, 8, 16), repeats: int = 3) -> list[dict]:
    """Recall@k vs nprobe vs measured QPS (single process, one thread group)."""
    _, truth = brute_force_topk(q, x, k=k)
    rows = []
    for nprobe in nprobes:
        t0 = time.perf_counter()
        for _ in range(repeats):
            index.search(q, k=k, nprobe=nprobe)
        dt = (time.perf_counter() - t0) / repeats
        rows.append(dict(nprobe=nprobe, recall=recall_at_k(index, q, x, k, nprobe, truth=truth),
                         ms=dt * 1e3, ms_per_query=dt * 1e3 / max(1, q.shape[0]),
                         qps=q.shape[0] / dt))
    return rows


if __name__ == "__main__":                                   # python xrec_ref/index.py
    torch.manual_seed(0)
    n, d, nq = 12000, 128, 512
    x = l2norm(torch.randn(n, d))
    q = l2norm(x[torch.randint(0, n, (nq,))] + 0.35 * l2norm(torch.randn(nq, d)))
    ix = IVFIndex(d=d, nlist=128).train(x).add(x)
    print(json.dumps(ix.stats()))
    for r in sweep(ix, q, x, k=10, nprobes=(1, 2, 4, 8, 16, 32, 128), repeats=1):
        print(f"nprobe={r['nprobe']:4d}  recall@10={r['recall']:.4f}  {r['ms_per_query']:.3f} ms/q  "
              f"{r['qps']:8.1f} qps")
