"""SID-AR control (paper §A.2, X-Rec-AR): same backbone and same item space, but the trigger is a
4-token semantic ID decoded autoregressively through the residual quantizer.

Identical encoder + identical "one extra layer with cached context K/V" decoding design as X-Rec;
the only change is *what* is generated — discrete SID tokens instead of a continuous embedding.
That isolates the discrete-vs-continuous variable, which is exactly the comparison the paper makes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Attention, EncLayer, RMSNorm, RoPE, SwiGLU
from rfm import normalize
from sid import ResidualQuantizer


class ARLayer(nn.Module):
    """One pre-norm Transformer block over a single generated token with cached context K/V."""

    def __init__(self, d: int, n_head: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_head)
        self.ff = SwiGLU(d)

    def forward(self, x, cache, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, kv=cache)
        return x + self.ff(self.n2(x))


class SIDAR(nn.Module):
    def __init__(self, cfg, item_emb: torch.Tensor, anchor_of_item: torch.Tensor,
                 quantizer: ResidualQuantizer, sid_codes: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("item_emb", item_emb, persistent=False)
        self.register_buffer("sid_codes", sid_codes, persistent=False)
        self.register_buffer("code_emb", quantizer.code_embeddings(), persistent=False)  # (M,K,d_e)
        self.M, self.K = quantizer.M, quantizer.K

        self.attr_a = nn.Embedding(cfg.n_action, cfg.d_e)
        self.attr_d = nn.Embedding(cfg.n_duration, cfg.d_e)
        self.W_f = nn.Linear(cfg.n_attr * cfg.d_e, cfg.d_m, bias=False)
        self.W_x = nn.Linear(cfg.d_e, cfg.d_m, bias=False)
        self.enc = nn.ModuleList([EncLayer(cfg.d_m, cfg.n_head) for _ in range(cfg.n_layer - 1)])
        self.ar = ARLayer(cfg.d_m, cfg.n_head)
        self.step_emb = nn.Embedding(self.M, cfg.d_m)
        self.code_in = nn.Linear(cfg.d_e, cfg.d_m, bias=False)
        self.heads = nn.ModuleList([nn.Linear(cfg.d_m, self.K, bias=False) for _ in range(self.M)])
        self.bos = nn.Parameter(torch.zeros(cfg.d_m))
        self.rope = RoPE(cfg.d_m // cfg.n_head)

    # identical context encoding to X-Rec
    @torch.no_grad()
    def prefill(self, items, action, dur):
        M, N = items.shape
        dev = items.device
        x = self.W_x(self.item_emb[items]) + self.W_f(torch.cat([self.attr_a(action), self.attr_d(dur)], -1))
        pos = torch.arange(N, device=dev)[None].expand(M, N)
        cos, sin = self.rope.cos_sin(pos, x.dtype)
        mask = torch.ones(N, N, dtype=torch.bool, device=dev).tril()[None, None]
        for layer in self.enc:
            x = layer(x, cos, sin, mask)
        return self.ar.attn.keys_values(x, cos, sin), x[:, -1], N

    def _inp(self, m: int, prev: torch.Tensor | None, B: int, dev):
        """input embedding of decoding step m; prev = (B*K,) previous-level codes or None."""
        if m == 0:
            return (self.bos + self.step_emb.weight[m]).expand(B, -1)
        return self.step_emb.weight[m].expand(prev.numel(), -1) + self.code_in(self.code_emb[m - 1][prev])

    def _logits(self, m: int, inp: torch.Tensor, cache, N: int):
        dev = inp.device
        p = torch.full((inp.shape[0], 1), N + m, device=dev)
        cos, sin = self.rope.cos_sin(p, inp.dtype)
        out = self.ar(inp[:, None, :], cache, cos, sin)[:, 0]
        return self.heads[m](out)

    # ------------------------------------------------------------------ loss
    def loss(self, batch):
        items, act, dur, tgt, *_ = batch
        B = items.shape[0]
        cache, _, N = self.prefill(items, act, dur)
        codes = self.sid_codes[tgt]                                     # (B, M)
        total = items.new_zeros(())
        prev = None
        for m in range(self.M):
            inp = self._inp(m, prev, B, items.device)
            logits = self._logits(m, inp, cache, N)
            total = total + F.cross_entropy(logits, codes[:, m])
            prev = codes[:, m]
        return total / self.M, items.new_zeros(())

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, items, act, dur, K=None, steps=None, **kw):
        cfg = self.cfg
        K = K or cfg.K
        B, dev = items.shape[0], items.device
        cache, _, N = self.prefill(items, act, dur)
        kk = cache[0].repeat_interleave(K, dim=0)
        vv = cache[1].repeat_interleave(K, dim=0)

        codes = torch.zeros(B, K, self.M, dtype=torch.long, device=dev)
        scores = torch.full((B, K), -1e4, device=dev)
        scores[:, 0] = 0.0
        for m in range(self.M):
            prev = codes[:, :, m - 1].reshape(B * K) if m > 0 else None
            inp = self._inp(m, prev, B * K, dev)
            logits = self._logits(m, inp, (kk, vv), N).reshape(B, K, self.K)
            cand = (scores[:, :, None] + logits.log_softmax(-1)).reshape(B, -1)
            top = cand.topk(K, dim=-1)
            scores = top.values
            src, code = top.indices // self.K, top.indices % self.K
            if m > 0:
                codes = codes.gather(1, src[:, :, None].expand(B, K, self.M))
            codes[:, :, m] = code

        flat = codes.reshape(B * K, self.M)
        out = torch.zeros(B * K, cfg.d_e, device=dev)
        for m in range(self.M):
            out = out + self.code_emb[m][flat[:, m]]
        return normalize(out).reshape(B, K, cfg.d_e)
