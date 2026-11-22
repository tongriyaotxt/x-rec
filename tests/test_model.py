"""Model-level checks: shapes, on-manifold triggers, deterministic sampling, U2I control."""
import torch
from config import CFG
from model import U2I, XRec

from data import build_world, make_split


def tiny_cfg() -> CFG:
    cfg = CFG()
    cfg.n_items, cfg.n_anchor, cfg.n_modes = 400, 8, 4
    cfg.seq_len, cfg.d_e, cfg.d_m, cfg.n_head, cfg.n_layer = 10, 32, 64, 4, 2
    cfg.batch, cfg.n_noise_sft, cfg.n_eval, cfg.steps, cfg.K = 6, 2, 16, 3, 5
    cfg.n_interest = 3
    return cfg


def _fixture(seed: int = 0):
    cfg = tiny_cfg()
    torch.manual_seed(seed)
    world = build_world(cfg, seed=seed)
    split = make_split(cfg, world, n=cfg.batch * 2, seed=seed + 1)
    return cfg, world, split


def _batch(cfg, split):
    i = torch.arange(cfg.batch)
    return (split["ctx"][i], split["act"][i], split["dur"][i],
            split["tgt"][i], split["tgt_act"][i], split["tgt_dur"][i])


def test_world_anchors_identify_modes():
    _, world, _ = _fixture()
    assert world["emb"].shape == (400, 32)
    assert (world["emb"].norm(dim=-1) - 1).abs().max().item() < 1e-5
    assert world["anchor_acc"] > 0.8          # K-means clusters align with the generating mode


def test_xrec_loss_is_finite_and_backprops():
    cfg, world, split = _fixture()
    model = XRec(cfg, world["emb"], world["anchor_of_item"])
    l_anchor, l_rfm = model.loss(_batch(cfg, split))
    assert torch.isfinite(l_anchor) and torch.isfinite(l_rfm)
    (l_anchor + l_rfm).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_samples_lie_on_the_sphere():
    cfg, world, split = _fixture()
    model = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    trig = model.sample(split["ctx"][:4], split["act"][:4], split["dur"][:4], K=cfg.K, steps=cfg.steps)
    assert trig.shape == (4, cfg.K, cfg.d_e)
    assert (trig.norm(dim=-1) - 1).abs().max().item() < 1e-4


def test_sampling_is_reproducible_with_a_generator():
    cfg, world, split = _fixture()
    model = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    a = model.sample(split["ctx"][:4], split["act"][:4], split["dur"][:4], gen=torch.Generator().manual_seed(7))
    b = model.sample(split["ctx"][:4], split["act"][:4], split["dur"][:4], gen=torch.Generator().manual_seed(7))
    c = model.sample(split["ctx"][:4], split["act"][:4], split["dur"][:4], gen=torch.Generator().manual_seed(8))
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_no_anchor_variant_uses_the_null_anchor():
    cfg, world, split = _fixture()
    cfg.use_anchor = False
    model = XRec(cfg, world["emb"], world["anchor_of_item"]).eval()
    trig = model.sample(split["ctx"][:3], split["act"][:3], split["dur"][:3], K=4, steps=2)
    assert trig.shape == (3, 4, cfg.d_e)
    assert (trig.norm(dim=-1) - 1).abs().max().item() < 1e-4


def test_u2i_controls_shapes_and_loss():
    cfg, world, split = _fixture()
    for k in (1, cfg.n_interest):
        model = U2I(cfg, world["emb"], world["anchor_of_item"], n_interest=k)
        loss, _ = model.loss(_batch(cfg, split))
        assert torch.isfinite(loss)
        trig = model.sample(split["ctx"][:4], split["act"][:4], split["dur"][:4], K=4)
        assert trig.shape == (4, 4, cfg.d_e)
        assert (trig.norm(dim=-1) - 1).abs().max().item() < 1e-5
