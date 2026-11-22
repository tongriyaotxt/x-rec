"""X-Rec model: anchor predictor + late-interaction DiT velocity field (arXiv:2609.29180 §3.3).

Backbone follows the paper's Qwen3-style recipe: RMSNorm, RoPE, QK-normalization, SwiGLU, causal mask.
Deliberate deviation: the late-interaction layer. Condition tokens are encoded by layers 1..L-1, their
last-layer K/V are cached once per request, and every denoising step runs exactly ONE Transformer layer
for the denoising token(s) — that is the source of the 3.46x throughput (§3.3.3, §4.3.3).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rfm import (
    conditional_velocity,
    euclidean_velocity,
    geodesic_interp,
    linear_interp,
    normalize,
    tangent_project,
)


# ---------------------------------------------------------------- primitives
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class RoPE(nn.Module):
    def __init__(self, d_head: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head)),
                             persistent=False)

    def cos_sin(self, pos: torch.Tensor, dtype):
        """pos (M,T) -> cos,sin of shape (M,1,T,d_head)."""
        f = torch.einsum("mt,k->mtk", pos.float(), self.inv_freq)
        c, s = f.cos().to(dtype), f.sin().to(dtype)
        return torch.cat([c, c], -1)[:, None], torch.cat([s, s], -1)[:, None]


class Attention(nn.Module):
    def __init__(self, d: int, n_head: int):
        super().__init__()
        self.h, self.dh = n_head, d // n_head
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.o = nn.Linear(d, d, bias=False)
        self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)

    def split(self, x):
        """(M,T,d) -> (M,H,T,dh)"""
        M, T, _ = x.shape
        return x.view(M, T, self.h, self.dh).transpose(1, 2)

    def merge(self, x):
        M, H, T, dh = x.shape
        return x.transpose(1, 2).reshape(M, T, H * dh)

    def keys_values(self, h, cos, sin):
        k = self.kn(self.split(self.k(h)))
        v = self.split(self.v(h))
        if cos is not None:
            k = k * cos + rotate_half(k) * sin
        return k, v

    def forward(self, x, cos=None, sin=None, kv=None, mask=None):
        q = self.qn(self.split(self.q(x)))
        if cos is not None:
            q = q * cos + rotate_half(q) * sin
        # the query token is rotated at its own position; cached keys already carry theirs
        k, v = self.keys_values(x, cos, sin)
        if kv is not None:
            k = torch.cat([kv[0], k], dim=-2)
            v = torch.cat([kv[1], v], dim=-2)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if mask is not None:
            s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        return self.o(self.merge(s.softmax(-1) @ v))


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 2.67):
        super().__init__()
        hid = (int(d * mult * 2 / 3) + 7) // 8 * 8
        self.w1, self.w3, self.w2 = (nn.Linear(d, hid, bias=False), nn.Linear(d, hid, bias=False),
                                     nn.Linear(hid, d, bias=False))

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncLayer(nn.Module):
    def __init__(self, d: int, n_head: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_head)
        self.ff = SwiGLU(d)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask=mask)
        return x + self.ff(self.n2(x))


class DiTLayer(nn.Module):
    """Final layer: AdaLN(t + anchor); condition K/V are injected as a cache (§3.3.3)."""

    def __init__(self, d: int, n_head: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_head)
        self.ff = SwiGLU(d)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    @torch.no_grad()
    def cache(self, h_cond, cos, sin):
        k, v = self.attn.keys_values(h_cond, cos, sin)
        return k.detach(), v.detach()

    def forward(self, x, cond, cache, cos, sin):
        s1, sc1, g1, s2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sc1[:, None]) + s1[:, None]
        x = x + g1[:, None] * self.attn(h, cos, sin, kv=cache)
        h = self.n2(x) * (1 + sc2[:, None]) + s2[:, None]
        return x + g2[:, None] * self.ff(h)


def timestep_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


# ------------------------------------------------------------------ X-Rec
class XRec(nn.Module):
    def __init__(self, cfg, item_emb: torch.Tensor, anchor_of_item: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("item_emb", item_emb, persistent=False)
        self.register_buffer("anchor_of_item", anchor_of_item, persistent=False)
        self.attr_a, self.attr_d = nn.Embedding(cfg.n_action, cfg.d_e), nn.Embedding(cfg.n_duration, cfg.d_e)
        self.W_f = nn.Linear(cfg.n_attr * cfg.d_e, cfg.d_m, bias=False)
        self.W_x = nn.Linear(cfg.d_e, cfg.d_m, bias=False)
        self.enc = nn.ModuleList([EncLayer(cfg.d_m, cfg.n_head) for _ in range(cfg.n_layer - 1)])
        self.dit = DiTLayer(cfg.d_m, cfg.n_head)
        self.anchor_emb = nn.Embedding(cfg.n_anchor + 1, cfg.d_m)
        self.anchor_head = nn.Linear(cfg.d_m, cfg.n_anchor, bias=False)
        self.v_head = nn.Linear(cfg.d_m, cfg.d_e, bias=False)
        self.t_mlp = nn.Sequential(nn.Linear(64, cfg.d_m), nn.SiLU(), nn.Linear(cfg.d_m, cfg.d_m))
        self.rope = RoPE(cfg.d_m // cfg.n_head)

    # ---- feature encoding (eq.25/26)
    def f_m(self, action, dur):
        return self.W_f(torch.cat([self.attr_a(action), self.attr_d(dur)], -1))

    def token(self, items, action, dur):
        return self.W_x(self.item_emb[items]) + self.f_m(action, dur)

    # ---- prefill + last-layer KV cache
    def prefill(self, items, action, dur):
        M, N = items.shape
        x = self.token(items, action, dur)
        pos = torch.arange(N, device=items.device)[None].expand(M, N)
        cos, sin = self.rope.cos_sin(pos, x.dtype)
        mask = torch.ones(N, N, dtype=torch.bool, device=items.device).tril()[None, None]
        for layer in self.enc:
            x = layer(x, cos, sin, mask)
        cache = self.dit.cache(x, cos, sin)
        return cache, x[:, -1], self.f_m(action[:, -1], dur[:, -1]), N

    # ---- all-prefix prefill: one encoder pass, N-1 next-item targets (paper Fig.3a)
    def prefill_all(self, items, action, dur):
        M, N = items.shape
        x = self.token(items, action, dur)
        pos = torch.arange(N, device=items.device)[None].expand(M, N)
        cos, sin = self.rope.cos_sin(pos, x.dtype)
        mask = torch.ones(N, N, dtype=torch.bool, device=items.device).tril()[None, None]
        for layer in self.enc:
            x = layer(x, cos, sin, mask)
        k, v = self.dit.attn.keys_values(x, cos, sin)
        return (k, v), x, self.f_m(action, dur), N

    def anchor_logits(self, h_last, f_m):                       # eq.29
        return self.anchor_head(h_last + f_m)

    def velocity(self, x_t, anc, t, cache, f_m, pos: int):       # eq.31
        cond = self.anchor_emb(anc) + self.t_mlp(timestep_embedding(t, 64))
        tok = (self.W_x(x_t) + f_m)[:, None, :]
        p = torch.full((x_t.shape[0], 1), pos, device=x_t.device)
        cos, sin = self.rope.cos_sin(p, x_t.dtype)
        out = self.dit(tok, cond, cache, cos, sin)[:, 0]
        return tangent_project(self.v_head(out), x_t)

    # ---------------------------------------------------------------- losses
    def loss(self, batch, gen=None):
        cfg = self.cfg
        items, act, dur, tgt, *_ = batch
        M = items.shape[0]
        cache, h_last, f, N = self.prefill(items, act, dur)

        anc_tgt = self.anchor_of_item[tgt]
        logits = self.anchor_logits(h_last, f)
        l_anchor = F.cross_entropy(logits, anc_tgt) if cfg.use_anchor else logits.new_zeros(())

        R = cfg.n_noise_sft
        tgt_emb = self.item_emb[tgt]
        eps = normalize(torch.randn(R, M, cfg.d_e, device=items.device))
        t = torch.rand(R, M, device=items.device)
        x1 = tgt_emb[None].expand(R, M, -1)
        if cfg.use_rfm:
            x_t, om = geodesic_interp(eps, x1, t)
            v_tgt = conditional_velocity(eps, x1, om, t)
        else:
            x_t, _ = linear_interp(eps, x1, t)
            v_tgt = euclidean_velocity(eps, x1)

        idx = torch.arange(M, device=items.device).repeat_interleave(R)
        x_flat = x_t.reshape(R * M, cfg.d_e)
        cache_flat = (cache[0].index_select(0, idx), cache[1].index_select(0, idx))
        anc = anc_tgt.index_select(0, idx) if cfg.use_anchor else \
            torch.full((R * M,), cfg.n_anchor, dtype=torch.long, device=items.device)

        drop = torch.rand(R * M, device=items.device) < cfg.ctx_drop   # CFG conditioning dropout
        if bool(drop.any()):
            cache_flat = (torch.where(drop[:, None, None, None], 0.0, cache_flat[0]),
                          torch.where(drop[:, None, None, None], 0.0, cache_flat[1]))
            anc = torch.where(drop, torch.full_like(anc, cfg.n_anchor), anc)

        v_pred = self.velocity(x_flat, anc, t.reshape(-1), cache_flat, f.index_select(0, idx), N)
        v_tgt = v_tgt.reshape(R * M, cfg.d_e)
        if cfg.use_rfm:
            v_tgt = tangent_project(v_tgt, x_flat)
        return cfg.alpha * l_anchor, F.mse_loss(v_pred, v_tgt)

    def loss_all_prefix(self, batch, gen=None):
        """Paper Fig.3(a): ONE encoder pass over the whole context, every position predicts its successor.

        The old path reran the encoder for a randomly cut prefix each step; this instead reuses a single
        causal encoder pass and only repeats the last-layer (DiT) work per position. That is the
        pretraining-time analogue of the late-interaction win at inference time, and it is what makes the
        model see N-1 targets per sequence instead of one.
        """
        cfg = self.cfg
        items, act, dur = batch[0], batch[1], batch[2]
        M, N = items.shape
        (k_all, v_all), h, f_all, _ = self.prefill_all(items, act, dur)
        dev = items.device
        R = max(1, cfg.n_noise_pre)
        idx = torch.arange(M, device=dev).repeat_interleave(R)
        l_anchor = h.new_zeros(())
        l_rfm = h.new_zeros(())
        n_pos = max(1, N - 1)
        for i in range(N - 1):
            tgt = items[:, i + 1]
            anc_tgt = self.anchor_of_item[tgt]
            if cfg.use_anchor:
                l_anchor = l_anchor + F.cross_entropy(self.anchor_logits(h[:, i], f_all[:, i]), anc_tgt)

            tgt_emb = self.item_emb[tgt]
            eps = normalize(torch.randn(R, M, cfg.d_e, device=dev, generator=gen))
            t = torch.rand(R, M, device=dev, generator=gen)
            x1 = tgt_emb[None].expand(R, M, -1)
            if cfg.use_rfm:
                x_t, om = geodesic_interp(eps, x1, t)
                v_tgt = conditional_velocity(eps, x1, om, t)
            else:
                x_t, _ = linear_interp(eps, x1, t)
                v_tgt = euclidean_velocity(eps, x1)

            cache_i = (k_all[:, :, : i + 1].index_select(0, idx),
                       v_all[:, :, : i + 1].index_select(0, idx))
            x_flat = x_t.reshape(R * M, cfg.d_e)
            f_flat = f_all[:, i].index_select(0, idx)
            anc = (anc_tgt.index_select(0, idx) if cfg.use_anchor
                   else torch.full((R * M,), cfg.n_anchor, dtype=torch.long, device=dev))
            drop = torch.rand(R * M, device=dev, generator=gen) < cfg.ctx_drop
            if bool(drop.any()):
                cache_i = (torch.where(drop[:, None, None, None], 0.0, cache_i[0]),
                           torch.where(drop[:, None, None, None], 0.0, cache_i[1]))
                anc = torch.where(drop, torch.full_like(anc, cfg.n_anchor), anc)

            v_pred = self.velocity(x_flat, anc, t.reshape(-1), cache_i, f_flat, i + 1)
            v_t = v_tgt.reshape(R * M, cfg.d_e)
            if cfg.use_rfm:
                v_t = tangent_project(v_t, x_flat)
            l_rfm = l_rfm + F.mse_loss(v_pred, v_t)
        l_anchor = l_anchor / n_pos if cfg.use_anchor else l_anchor
        return cfg.alpha * l_anchor, l_rfm / n_pos


    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, items, act, dur, K=None, steps=None, anchor_temp=None, cfg_scale=None, gen=None):
        cfg = self.cfg
        K = K or cfg.K
        steps = steps or cfg.steps
        T_anchor = cfg.anchor_temp if anchor_temp is None else anchor_temp
        scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
        B, dev = items.shape[0], items.device

        cache, h_last, f, N = self.prefill(items, act, dur)
        if cfg.use_anchor:
            anc = torch.multinomial((self.anchor_logits(h_last, f) / T_anchor).softmax(-1),
                                    K, replacement=True, generator=gen)
        else:
            anc = torch.full((B, K), cfg.n_anchor, dtype=torch.long, device=dev)

        x = normalize(torch.randn(B, K, cfg.d_e, device=dev, generator=gen))
        idx = torch.arange(B, device=dev).repeat_interleave(K)
        f_flat = f.index_select(0, idx)
        cache_c = (cache[0].index_select(0, idx), cache[1].index_select(0, idx))
        cache_u = (torch.zeros_like(cache_c[0]), torch.zeros_like(cache_c[1]))
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((B * K,), s * dt, device=dev)
            vc = self.velocity(x.reshape(B * K, -1), anc.reshape(-1), t, cache_c, f_flat, N)
            if scale != 1.0:
                vu = self.velocity(x.reshape(B * K, -1),
                                   torch.full_like(anc.reshape(-1), cfg.n_anchor), t, cache_u, f_flat, N)
                v = tangent_project(vu + scale * (vc - vu), x.reshape(B * K, -1))
            else:
                v = vc
            x = _euler(x.reshape(B * K, -1), v, dt).reshape(B, K, cfg.d_e)
        return x


def _euler(x, v, dt):
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cos(n * dt) * x + torch.sin(n * dt) * (v / n)


# ------------------------------------------------------- U2I baselines (§2.1)
class U2I(nn.Module):
    """arch='u2i': SASRec-style single vector. arch='u2i_k': ComiRec-SA multi-interest (eq.34/36)."""

    def __init__(self, cfg, item_emb: torch.Tensor, anchor_of_item: torch.Tensor, n_interest: int = 1):
        super().__init__()
        self.cfg, self.K = cfg, n_interest
        self.register_buffer("item_emb", item_emb, persistent=False)
        self.attr_a, self.attr_d = nn.Embedding(cfg.n_action, cfg.d_e), nn.Embedding(cfg.n_duration, cfg.d_e)
        self.W_f = nn.Linear(cfg.n_attr * cfg.d_e, cfg.d_m, bias=False)
        self.W_x = nn.Linear(cfg.d_e, cfg.d_m, bias=False)
        self.enc = nn.ModuleList([EncLayer(cfg.d_m, cfg.n_head) for _ in range(cfg.n_layer)])
        self.rope = RoPE(cfg.d_m // cfg.n_head)
        self.q = nn.Parameter(torch.randn(n_interest, cfg.d_m) * 0.02)
        self.W_t = nn.Linear(cfg.d_m, cfg.d_e, bias=False)

    def encode(self, items, act, dur):
        M, N = items.shape
        x = self.W_x(self.item_emb[items]) + self.W_f(torch.cat([self.attr_a(act), self.attr_d(dur)], -1))
        pos = torch.arange(N, device=items.device)[None].expand(M, N)
        cos, sin = self.rope.cos_sin(pos, x.dtype)
        mask = torch.ones(N, N, dtype=torch.bool, device=items.device).tril()[None, None]
        for layer in self.enc:
            x = layer(x, cos, sin, mask)
        w = torch.einsum("mnd,kd->kmn", x, self.q).softmax(-1)     # (K,M,N)
        pooled = torch.einsum("kmn,mnd->kmd", w, x)
        return normalize(self.W_t(pooled)).transpose(0, 1)          # (M,K,d_e)

    def loss(self, batch, temp=0.05):
        items, act, dur, tgt, *_ = batch
        trig = self.encode(items, act, dur)
        sim = torch.einsum("mkd,nd->nmk", trig, self.item_emb[tgt]).max(-1).values  # (N,M)
        labels = torch.arange(sim.shape[1], device=sim.device)
        return F.cross_entropy((sim / temp).T, labels), sim.new_zeros(())

    @torch.no_grad()
    def sample(self, items, act, dur, K=None, **kw):
        trig = self.encode(items, act, dur)
        K = K or self.K
        reps = math.ceil(K / trig.shape[1])
        return trig.repeat(1, reps, 1)[:, :K]
