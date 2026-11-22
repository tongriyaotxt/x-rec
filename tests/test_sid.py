"""SID-AR control: residual quantiser fidelity, AR loss/sampling, agreement with the paper's 0.94 claim."""
import torch
from config import CFG
from models_sid import SIDAR
from rfm import normalize
from sid import ResidualQuantizer

from data import build_world, make_split


def _cfg() -> CFG:
    cfg = CFG()
    cfg.n_items, cfg.n_anchor, cfg.n_modes = 400, 8, 4
    cfg.seq_len, cfg.d_e, cfg.d_m, cfg.n_head, cfg.n_layer = 10, 32, 64, 4, 2
    cfg.batch, cfg.K = 6, 5
    cfg.sid_codebooks, cfg.sid_size = 3, 64
    return cfg


def _fixture(seed: int = 0):
    cfg = _cfg()
    torch.manual_seed(seed)
    world = build_world(cfg, seed=seed)
    split = make_split(cfg, world, n=8, seed=seed + 1)
    return cfg, world, split


def test_residual_quantiser_reconstructs_with_high_cosine():
    """Real item vectors are clustered, so 4-token SIDs keep ~0.94 cosine (paper §A.1). Isotropic
    Gaussian noise is the worst case for k-means and is *not* the regime to assert on."""
    g = torch.Generator().manual_seed(0)
    centers = normalize(torch.randn(8, 32, generator=g))
    mode = torch.randint(0, 8, (600,), generator=g)
    emb = normalize(centers[mode] + 0.35 * normalize(torch.randn(600, 32, generator=g)))
    q = ResidualQuantizer(n_codebooks=4, codebook_size=64, seed=0).fit(emb)
    codes = q.encode(emb)
    assert codes.shape == (600, 4)
    assert int(codes.min()) >= 0 and int(codes.max()) < 64
    assert q.recon_cosine(emb) > 0.9                      # paper reports 0.94 for 4-token SIDs
    assert q.code_embeddings().shape == (4, 64, 32)
    assert q.vocab_size == 64


def test_quantiser_is_monotone_in_levels():
    emb = normalize(torch.randn(400, 24, generator=torch.Generator().manual_seed(1)))
    one = ResidualQuantizer(n_codebooks=1, codebook_size=64, seed=0).fit(emb).recon_cosine(emb)
    three = ResidualQuantizer(n_codebooks=3, codebook_size=64, seed=0).fit(emb).recon_cosine(emb)
    assert three > one


def test_sid_ar_loss_backprops_and_sampling_stays_on_sphere():
    cfg, world, split = _fixture()
    q = ResidualQuantizer(n_codebooks=cfg.sid_codebooks, codebook_size=cfg.sid_size, seed=0)
    q.fit(world["emb"])
    model = SIDAR(cfg, world["emb"], world["anchor_of_item"], q, q.encode(world["emb"]))
    i = torch.arange(cfg.batch)
    loss, aux = model.loss((split["ctx"][i], split["act"][i], split["dur"][i],
                            split["tgt"][i], split["tgt_act"][i], split["tgt_dur"][i]))
    assert torch.isfinite(loss) and float(aux) == 0.0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    trig = model.eval().sample(split["ctx"][:3], split["act"][:3], split["dur"][:3], K=4)
    assert trig.shape == (3, 4, cfg.d_e)
    assert (trig.norm(dim=-1) - 1).abs().max().item() < 1e-4


def test_sid_ar_decodes_one_token_per_level():
    """Decode cost must scale with M (4 serial steps), not with T (denoising steps)."""
    cfg, world, split = _fixture()
    q = ResidualQuantizer(n_codebooks=4, codebook_size=32, seed=0).fit(world["emb"])
    model = SIDAR(cfg, world["emb"], world["anchor_of_item"], q, q.encode(world["emb"])).eval()
    calls = {"n": 0}
    real = model._logits

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    model._logits = counting
    model.sample(split["ctx"][:2], split["act"][:2], split["dur"][:2], K=3)
    assert calls["n"] == 4
