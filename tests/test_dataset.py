"""Real-data path: format parsing, PPMI/SVD embeddings, leave-one-out splits, fixture roundtrip."""
import torch
from config import CFG
from dataset import Ratings, item_embeddings, load_file, read_ratings, stats_of, write_fixture


def _cfg() -> CFG:
    cfg = CFG()
    cfg.seq_len, cfg.d_e, cfg.n_action, cfg.n_duration = 8, 16, 8, 4
    cfg.n_anchor, cfg.n_modes = 8, 8
    return cfg


def test_fixture_roundtrip_and_monotone_time(tmp_path):
    p = write_fixture(str(tmp_path / "f.dat"), n_users=40, n_items=60, seed=1, min_len=8, max_len=16)
    r = read_ratings(p)
    assert r.n_users == 40 and r.n_items == 60
    key = r.user * (int(r.ts.max()) + 1) + r.ts
    assert len(set(key.tolist())) == len(r)          # no two events of a user share a timestamp
    assert float(r.rating.min()) >= 1.0 and float(r.rating.max()) <= 5.0
    s = stats_of(r)
    assert s["interactions"] == len(r) and 0 < s["density"] <= 1


def test_all_three_formats_parse_to_the_same_shape(tmp_path):
    n = 200
    ml1m = tmp_path / "a.dat"
    ml1m.write_text("\n".join(f"{i % 7 + 1}::{i % 5 + 1}::4::{1000 + i}" for i in range(n)))
    tsv = tmp_path / "b.data"
    tsv.write_text("\n".join(f"{i % 7 + 1}\t{i % 5 + 1}\t4\t{1000 + i}" for i in range(n)))
    csv = tmp_path / "c.csv"
    csv.write_text("user,item,rating,timestamp\n" +
                   "\n".join(f"{i % 7 + 1},{i % 5 + 1},4,{1000 + i}" for i in range(n)))
    parsed = [read_ratings(str(p)) for p in (ml1m, tsv, csv)]
    assert len({(r.n_users, r.n_items, len(r)) for r in parsed}) == 1
    for r in parsed:
        assert isinstance(r, Ratings) and (r.user >= 0).all() and (r.item >= 0).all()


def test_embeddings_are_unit_norm_and_finite(tmp_path):
    p = write_fixture(str(tmp_path / "f.dat"), n_users=40, n_items=64, seed=2, min_len=8, max_len=14)
    emb = item_embeddings(read_ratings(p), d=16, seed=0)
    assert emb.shape == (64, 16)
    assert torch.isfinite(emb).all()
    assert (emb.norm(dim=-1) - 1).abs().max().item() < 1e-4


def test_load_file_builds_splits_for_training(tmp_path):
    p = write_fixture(str(tmp_path / "f.dat"), n_users=50, n_items=80, seed=3, min_len=10, max_len=18)
    cfg = _cfg()
    world, splits, info = load_file(cfg, p, seed=0)
    assert cfg.n_items == 80 and cfg.n_anchor == world["n_anchor"]
    assert world["anchor_acc"] is None                        # no ground-truth modes on real logs
    assert (world["emb"].norm(dim=-1) - 1).abs().max().item() < 1e-4
    assert int(world["anchor_of_item"].max()) < cfg.n_anchor
    for name in ("train", "valid", "test"):
        s = splits[name]
        n = s["ctx"].shape[0]
        assert n > 0 and s["ctx"].shape == (n, cfg.seq_len)
        assert s["tgt"].shape == (n,) and int(s["tgt"].max()) < cfg.n_items
        assert s["active_modes"].shape == (n, 2)
        assert info["rows"][name] == n


def test_splits_are_time_ordered_leave_one_out(tmp_path):
    """The test target must be the user's last interaction, the valid target the one before it."""
    from dataset import make_splits
    p = write_fixture(str(tmp_path / "f.dat"), n_users=20, n_items=40, seed=4, min_len=12, max_len=16)
    r = read_ratings(p)
    cfg = _cfg()
    world, splits, _ = load_file(cfg, p, seed=0)
    by_user: dict[int, list[int]] = {}
    order = torch.argsort(r.user * (int(r.ts.max()) + 1) + r.ts, stable=True)
    for k in order.tolist():
        seq = by_user.setdefault(int(r.user[k]), [])
        item = int(r.item[k])
        if not seq or seq[-1] != item:            # the loader collapses consecutive repeats
            seq.append(item)
    for split, off in (("test", 1), ("valid", 2)):
        for row, u in enumerate(splits[split]["user"].tolist()):
            seq = by_user[u]
            assert int(splits[split]["tgt"][row]) == seq[-off]
    assert make_splits is not None and world["n_items"] == r.n_items
