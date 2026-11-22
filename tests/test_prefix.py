"""All-prefix pretraining (paper Fig.3a): N-1 targets from one encoder pass, identical to the
last-position path when the sequence has length 2."""
import torch
from config import CFG
from model import XRec

from data import build_world, make_split


def _cfg() -> CFG:
    cfg = CFG()
    cfg.n_items, cfg.n_anchor, cfg.n_modes = 300, 8, 4
    cfg.seq_len, cfg.d_e, cfg.d_m, cfg.n_head, cfg.n_layer = 4, 24, 48, 4, 2
    cfg.batch, cfg.n_noise_sft, cfg.n_noise_pre, cfg.K, cfg.steps = 5, 1, 1, 3, 2
    cfg.ctx_drop = 0.0
    return cfg


def _fixture(seed: int = 0):
    cfg = _cfg()
    torch.manual_seed(seed)
    world = build_world(cfg, seed=seed)
    split = make_split(cfg, world, n=8, seed=seed + 1)
    return cfg, world, split


def _batch(split, n, width=None):
    ctx, act, dur = split["ctx"][:n], split["act"][:n], split["dur"][:n]
    if width is not None:
        ctx, act, dur = ctx[:, :width], act[:, :width], dur[:, :width]
    return ctx, act, dur


def test_all_prefix_matches_last_prefix_on_a_length_two_sequence():
    """With one history token the two paths must agree: same cache, same position, same noise."""
    cfg, world, split = _fixture()
    model = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    ctx, act, dur = _batch(split, cfg.batch, width=2)
    tgt, tgt_act, tgt_dur = ctx[:, 1], act[:, 1], dur[:, 1]

    torch.manual_seed(11)
    a1, r1 = model.loss_all_prefix((ctx, act, dur, tgt, tgt_act, tgt_dur))
    torch.manual_seed(11)
    a2, r2 = model.loss((ctx[:, :1], act[:, :1], dur[:, :1], tgt, tgt_act, tgt_dur))

    assert torch.allclose(a1, a2, atol=1e-6), (float(a1), float(a2))
    assert torch.allclose(r1, r2, atol=1e-6), (float(r1), float(r2))


def test_all_prefix_covers_every_position():
    cfg, world, split = _fixture()
    model = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    ctx, act, dur = _batch(split, cfg.batch)
    l_anchor, l_rfm = model.loss_all_prefix((ctx, act, dur, split["tgt"][:cfg.batch],
                                             split["tgt_act"][:cfg.batch], split["tgt_dur"][:cfg.batch]))
    assert torch.isfinite(l_anchor) and torch.isfinite(l_rfm)

    # recompute the anchor term by hand over all N-1 prefixes
    with torch.no_grad():
        _, h, f, _ = model.prefill_all(ctx, act, dur)
        manual = torch.stack([
            torch.nn.functional.cross_entropy(model.anchor_logits(h[:, i], f[:, i]),
                                              model.anchor_of_item[ctx[:, i + 1]])
            for i in range(ctx.shape[1] - 1)]).mean()
    assert torch.allclose(l_anchor, cfg.alpha * manual, atol=1e-6)


def test_all_prefix_backprops_through_the_encoder_once():
    cfg, world, split = _fixture()
    model = XRec(cfg, world["emb"], world["anchor_of_item"])
    ctx, act, dur = _batch(split, cfg.batch)
    l_anchor, l_rfm = model.loss_all_prefix((ctx, act, dur, split["tgt"][:cfg.batch],
                                             split["tgt_act"][:cfg.batch], split["tgt_dur"][:cfg.batch]))
    (l_anchor + l_rfm).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.W_x.weight.grad is not None                 # shared token embedding got gradient
