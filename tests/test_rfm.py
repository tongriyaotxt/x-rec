"""Numerical identities of the Riemannian flow-matching core (paper eq.20-24, eq.31)."""
import torch
from rfm import (
    cfg_combine,
    conditional_velocity,
    geodesic_interp,
    linear_interp,
    normalize,
    riemannian_euler_step,
    sample_base,
    tangent_project,
)

D, N = 32, 16


def _pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    eps = normalize(torch.randn(N, D, generator=g))
    x1 = normalize(torch.randn(N, D, generator=g))
    t = torch.rand(N, generator=g)
    return eps, x1, t


def test_geodesic_endpoints():
    eps, x1, _ = _pair()
    x0, _ = geodesic_interp(eps, x1, torch.zeros(N))
    xT, _ = geodesic_interp(eps, x1, torch.ones(N))
    assert torch.allclose(x0, eps, atol=1e-5)
    assert torch.allclose(xT, x1, atol=1e-5)


def test_velocity_matches_finite_difference():
    eps, x1, t = _pair()
    _, om = geodesic_interp(eps, x1, t)
    h = 1e-4
    xa, _ = geodesic_interp(eps, x1, t - h)
    xb, _ = geodesic_interp(eps, x1, t + h)
    fd = (xb - xa) / (2 * h)
    an = conditional_velocity(eps, x1, om, t)
    assert (fd - an).abs().max().item() < 1e-2


def test_states_stay_on_the_sphere():
    eps, x1, t = _pair()
    xt, _ = geodesic_interp(eps, x1, t)
    _, om = geodesic_interp(eps, x1, t)
    v = conditional_velocity(eps, x1, om, t)
    assert (xt.norm(dim=-1) - 1).abs().max().item() < 1e-6
    assert (v * xt).sum(-1).abs().max().item() < 1e-5   # velocity is tangent


def test_tangent_projection_removes_radial_part():
    eps, x1, t = _pair()
    xt, _ = geodesic_interp(eps, x1, t)
    v = tangent_project(torch.randn(N, D), xt)
    assert (v * xt).sum(-1).abs().max().item() < 1e-6


def test_euler_integration_is_exact_for_the_true_field():
    eps, x1, _ = _pair(seed=1)
    _, om = geodesic_interp(eps, x1, torch.rand(N))
    x, steps = eps.clone(), 200
    for s in range(steps):
        x = riemannian_euler_step(x, conditional_velocity(eps, x1, om, torch.full((N,), s / steps)), 1 / steps)
    assert (x - x1).abs().max().item() < 1e-4
    assert (x.norm(dim=-1) - 1).abs().max().item() < 1e-5


def test_euclidean_path_leaves_the_sphere():
    """Why RFM matters: the rectified-flow path is off-manifold in between."""
    eps, x1, _ = _pair()
    xm, _ = linear_interp(eps, x1, torch.full((N,), 0.5))
    assert (xm.norm(dim=-1) - 1).abs().max().item() > 0.1


def test_base_sampler_is_unit_norm():
    x = sample_base((1000, D))
    assert (x.norm(dim=-1) - 1).abs().max().item() < 1e-5


def test_cfg_combination_is_tangent():
    g = torch.Generator().manual_seed(3)
    x = normalize(torch.randn(8, D, generator=g))
    vc, vu = torch.randn(8, D, generator=g), torch.randn(8, D, generator=g)
    v = cfg_combine(vc, vu, 3.0, x)
    assert (v * x).sum(-1).abs().max().item() < 1e-5
    assert torch.allclose(v, vc, atol=1e-6) or not torch.allclose(v, vu)
