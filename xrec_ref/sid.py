"""Residual (k-means) quantizer = the paper's semantic-ID construction, without a VAE.

item embedding x^(e) is encoded level by level:
    r_0 = x^(e);  c_m = argmin_k ||r_{m-1} - e_{m,k}||;  r_m = r_{m-1} - e_{m,c_m}
and decoded by summing the chosen codebook vectors. This is exact residual quantization, so the
reconstruction cosine tells us directly how much information discrete SIDs throw away (paper: 0.94).
"""
from __future__ import annotations

import torch


class ResidualQuantizer:
    def __init__(self, n_codebooks: int = 4, codebook_size: int = 256, iters: int = 20, seed: int = 0):
        self.M = n_codebooks
        self.K = codebook_size
        self.iters = iters
        self.seed = seed
        self.codebooks: list[torch.Tensor] = []

    def fit(self, emb: torch.Tensor) -> ResidualQuantizer:
        from sklearn.cluster import KMeans
        residual = emb.clone()
        self.codebooks = []
        for m in range(self.M):
            km = KMeans(n_clusters=self.K, n_init=3, max_iter=self.iters,
                        random_state=self.seed + m).fit(residual.numpy())
            cb = torch.from_numpy(km.cluster_centers_).float()
            self.codebooks.append(cb)
            residual = residual - cb[torch.from_numpy(km.labels_).long()]
        return self

    @torch.no_grad()
    def encode(self, emb: torch.Tensor) -> torch.Tensor:
        residual, codes = emb.clone(), []
        for cb in self.codebooks:
            # squared distance to every centroid, chunked to stay cache friendly
            d = (residual.pow(2).sum(-1, keepdim=True) - 2 * residual @ cb.T + cb.pow(2).sum(-1))
            idx = d.argmin(-1)
            codes.append(idx)
            residual = residual - cb[idx]
        return torch.stack(codes, dim=-1)                     # (N, M) int64

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(codes.shape[0], self.codebooks[0].shape[1])
        for m, cb in enumerate(self.codebooks):
            out = out + cb[codes[:, m]]
        return out

    @torch.no_grad()
    def recon_cosine(self, emb: torch.Tensor) -> float:
        rec = self.decode(self.encode(emb))
        return float(torch.nn.functional.cosine_similarity(rec, emb, dim=-1).mean())

    @torch.no_grad()
    def code_embeddings(self) -> torch.Tensor:
        """(M, K, d) codebook vectors, used by the AR model's input embedding."""
        return torch.stack(self.codebooks, dim=0)

    @property
    def vocab_size(self) -> int:
        return self.K

    @property
    def sid_meanings(self) -> str:
        return f"RQ {self.M}x{self.K} (levels=coarse->fine)"
