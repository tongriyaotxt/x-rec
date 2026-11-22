"""Calibration baselines: popularity must be beaten before any ranking claim is meaningful."""
from baselines import item_frequency, popularity_recall, random_recall
from config import CFG

from data import build_world, make_split


def _splits(n_test: int = 512):
    cfg = CFG()
    world = build_world(cfg, seed=0)
    train = make_split(cfg, world, n=cfg.batch * 12, seed=2)
    test = make_split(cfg, world, n=n_test, seed=4)
    return cfg, world, train, test


def test_frequency_and_uniform_floor():
    cfg, world, train, _ = _splits()
    freq = item_frequency(train, cfg.n_items)
    assert freq.shape == (cfg.n_items,) and float(freq.sum()) == float(train["ctx"].numel())
    assert 0 < random_recall(cfg.n_items, 20) < 0.01


def test_popularity_grows_with_budget_and_beats_uniform():
    cfg, world, train, test = _splits()
    r1 = popularity_recall(train, test, world["mode"], cfg.n_items, K=1)
    r20 = popularity_recall(train, test, world["mode"], cfg.n_items, K=20)
    assert r20["popularity_K"] > r1["popularity_K"] > random_recall(cfg.n_items, 1)
    assert r20["oracle_mode_pop_K"] > 0
    assert r20["K"] == 20
