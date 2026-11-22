"""IVF index: exactness at full probe, recall monotonicity, persistence, k-means sanity."""
import torch
from index import IVFIndex, brute_force_topk, kmeans, recall_at_k, sweep


def _data(n=600, d=16, nq=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    x = x / x.norm(dim=-1, keepdim=True)
    q = x[:nq] + 0.05 * torch.randn(nq, d, generator=g)
    return x, q / q.norm(dim=-1, keepdim=True)


def test_kmeans_shapes_and_assignment_range():
    x, _ = _data()
    cent, assign = kmeans(x, 8, iters=5, seed=0)
    assert cent.shape == (8, x.shape[1])
    assert assign.shape == (x.shape[0],)
    assert int(assign.min()) >= 0 and int(assign.max()) < 8


def test_full_probe_is_exact():
    x, q = _data()
    ix = IVFIndex(d=x.shape[1], nlist=8, seed=0).train(x).add(x)
    _, got = ix.search(q, k=5, nprobe=8)
    _, want = brute_force_topk(q, x, k=5)
    for a, b in zip(got.tolist(), want.tolist(), strict=True):
        assert set(a) == set(b)


def test_recall_is_monotone_in_nprobe():
    x, q = _data(n=2000, nq=128)
    ix = IVFIndex(d=x.shape[1], nlist=32, seed=0).train(x).add(x)
    r = [recall_at_k(ix, q, x, k=10, nprobe=p) for p in (1, 2, 4, 8, 32)]
    assert r[-1] == 1.0
    assert all(b >= a - 1e-6 for a, b in zip(r, r[1:], strict=False)), r
    assert r[0] < r[-1]                      # the trade-off is real, not a plateau


def test_save_load_roundtrip(tmp_path):
    x, q = _data(n=400)
    ix = IVFIndex(d=x.shape[1], nlist=8, seed=0).train(x).add(x)
    p = str(tmp_path / "ix.pt")
    ix.save(p)
    other = IVFIndex.load(p)
    s1, i1 = ix.search(q, k=3, nprobe=4)
    s2, i2 = other.search(q, k=3, nprobe=4)
    assert torch.equal(i1, i2) and torch.allclose(s1, s2)
    assert other.stats()["n"] == 400


def test_sweep_reports_measured_rows():
    x, q = _data(n=800, nq=32)
    ix = IVFIndex(d=x.shape[1], nlist=16, seed=0).train(x).add(x)
    rows = sweep(ix, q, x, k=5, nprobes=(1, 16), repeats=1)
    assert [r["nprobe"] for r in rows] == [1, 16]
    assert rows[-1]["recall"] == 1.0
    assert all(r["ms"] > 0 and r["qps"] > 0 for r in rows)
