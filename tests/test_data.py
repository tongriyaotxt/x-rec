"""Fixture correctness: the synthetic stream and the Recall@(K x R) implementation."""
import torch

from config import CFG
from data import (
    batches,
    build_world,
    dual_mode_hit,
    make_split,
    mode_recall,
    pairwise_cosine,
    popularity_weights,
    recall_at,
)
from rfm import normalize


def _cfg():
    cfg = CFG()
    cfg.n_items, cfg.n_anchor, cfg.n_modes, cfg.seq_len, cfg.d_e = 300, 8, 4, 8, 16
    cfg.batch = 4
    return cfg


def test_split_shapes_and_mode_semantics():
    cfg = _cfg()
    world = build_world(cfg, seed=0)
    s = make_split(cfg, world, n=32, seed=1)
    assert s["ctx"].shape == (32, cfg.seq_len)
    assert s["act"].shape == (32, cfg.seq_len)
    assert s["tgt"].shape == (32,)
    assert s["active_modes"].shape == (32, 2)
    assert int(s["ctx"].max()) < cfg.n_items and int(s["ctx"].min()) >= 0
    # every target must come from one of the user's two active modes
    tgt_mode = world["mode"][s["tgt"]]
    assert bool((tgt_mode[:, None] == s["active_modes"]).any(-1).all())


def test_recall_is_one_when_the_trigger_is_the_item():
    g = torch.Generator().manual_seed(0)
    emb = normalize(torch.randn(50, 16, generator=g))
    tgt = torch.arange(8)
    trig = emb[tgt][:, None, :]                     # (8,1,16) — the target itself
    assert recall_at(trig, tgt, emb, R=1) == 1.0


def test_recall_is_zero_for_impossible_budget():
    g = torch.Generator().manual_seed(0)
    emb = normalize(torch.randn(50, 16, generator=g))
    tgt = torch.full((8,), 3, dtype=torch.long)
    trig = -emb[tgt][:, None, :]                    # antipodal point
    assert recall_at(trig, tgt, emb, R=1) < 0.5


def test_recall_grows_with_R():
    g = torch.Generator().manual_seed(0)
    emb = normalize(torch.randn(60, 16, generator=g))
    tgt = torch.arange(12)
    trig = normalize(emb[tgt] + 0.05 * torch.randn(12, 1, 16, generator=g))
    small, big = recall_at(trig, tgt, emb, R=1), recall_at(trig, tgt, emb, R=10)
    assert big >= small


def test_pairwise_cosine_bounds_and_mode_recall_shape():
    g = torch.Generator().manual_seed(0)
    one = normalize(torch.randn(4, 1, 16, generator=g))
    many = normalize(torch.randn(4, 8, 16, generator=g))
    assert pairwise_cosine(one) == 1.0
    assert -1.0 <= pairwise_cosine(many) <= 1.0
    emb = normalize(torch.randn(40, 16, generator=g))
    active = torch.zeros(4, 2, dtype=torch.long)
    m = mode_recall(many, active, torch.arange(40) % 4, emb, R=1)
    assert 0.0 <= m <= 1.0


def test_batch_iterator_is_bounded_and_complete():
    cfg = _cfg()
    world = build_world(cfg, seed=0)
    s = make_split(cfg, world, n=16, seed=2)
    seen = list(batches(s, cfg.batch, steps=5, seed=0))
    assert len(seen) == 5
    assert all(b[0].shape == (cfg.batch, cfg.seq_len) for b in seen)


def test_dual_mode_hit_counts_both_interests():
    """K copies of one trigger can only reach one mode; two triggers aimed at the two modes reach both."""
    g = torch.Generator().manual_seed(0)
    emb = normalize(torch.randn(60, 16, generator=g))
    item_mode = torch.arange(60) // 30                     # 0..29 -> mode 0, 30..59 -> mode 1
    active = torch.tensor([[0, 1], [0, 1]])

    single = emb[0][None, None].repeat(2, 6, 1)            # both users, 6 identical triggers
    assert dual_mode_hit(single, active, item_mode, emb, R=1) == 0.0

    row = torch.stack([emb[0], emb[0], emb[30]])           # two triggers in mode 0, one in mode 1
    two = row[None].repeat(2, 1, 1)                        # (2 users, 3 triggers, d)
    assert dual_mode_hit(two, active, item_mode, emb, R=1) == 1.0


def test_popularity_weights_are_monotone_and_normalised():
    g = torch.Generator().manual_seed(0)
    n_items = 50
    ctx = torch.randint(0, 5, (200, 8), generator=g)       # items 0..4 hot, 5..49 never seen
    w = popularity_weights(ctx, n_items, alpha=1.0)
    assert abs(float(w.mean()) - 1.0) < 1e-5               # normalised: mean weight is 1
    assert float(w[:5].min()) > float(w[5:].max())         # head outranks tail
    assert torch.allclose(popularity_weights(ctx, n_items, alpha=0.0), torch.ones(n_items))
