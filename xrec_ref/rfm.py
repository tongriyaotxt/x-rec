"""Riemannian Flow Matching core (sphere S^{d-1}) — X-Rec §3.1/§3.2.

All quantities follow arXiv:2609.29180v1:
  eq.20  geodesic interpolation
  eq.21  conditional velocity field        d/dt x_t = w/sin(w) * ( cos(t*w)*x1 - cos((1-t)*w)*x0 )
  eq.22  RFM regression loss
  eq.24  Riemannian Euler step
  eq.31  tangent-space projection of the predicted velocity

Run `python rfm.py --selftest` for numerical verification of every identity.
"""
from __future__ import annotations

import torch

EPS_OMEGA = 1e-5


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def geodesic_interp(eps: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """eq.20. eps,x1 unit vectors (..., d); t broadcastable to (..., 1). Returns (x_t, omega)."""
    t = t.unsqueeze(-1) if t.dim() == eps.dim() - 1 else t
    dot = (eps * x1).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.arccos(dot)
    sinw = torch.sin(omega).clamp_min(EPS_OMEGA)
    x_t = torch.sin((1.0 - t) * omega) / sinw * eps + torch.sin(t * omega) / sinw * x1
    return x_t, omega


def conditional_velocity(eps: torch.Tensor, x1: torch.Tensor, omega: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """eq.21 — the regression target of the RFM objective."""
    t = t.unsqueeze(-1) if t.dim() == omega.dim() - 1 else t
    sinw = torch.sin(omega).clamp_min(EPS_OMEGA)
    return (omega / sinw) * (torch.cos(t * omega) * x1 - torch.cos((1.0 - t) * omega) * eps)


def tangent_project(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """eq.31 — remove the radial component so v lies in T_x S^{d-1}."""
    return v - (v * x).sum(-1, keepdim=True) * x


def riemannian_euler_step(x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """eq.24 — exponential map along the geodesic spanned by tangent vector v."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cos(norm * dt) * x + torch.sin(norm * dt) * (v / norm)


# ---- ablations -------------------------------------------------------------
def linear_interp(eps: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """Euclidean rectified-flow path (X-Rec w/o RFM)."""
    t = t.unsqueeze(-1) if t.dim() == eps.dim() - 1 else t
    return (1.0 - t) * eps + t * x1, None


def euclidean_velocity(eps: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - eps


def sample_base(shape, dtype=torch.float32, generator=None) -> torch.Tensor:
    """Uniform on the unit hypersphere r(eps) = U(S^{d-1}) via normalized Gaussian."""
    return normalize(torch.randn(*shape, dtype=dtype, generator=generator))


@torch.no_grad()
def cfg_combine(v_cond: torch.Tensor, v_uncond: torch.Tensor, scale: float, x_t: torch.Tensor) -> torch.Tensor:
    """Classifier-free guidance, applied then re-projected to the tangent space."""
    return tangent_project(v_uncond + scale * (v_cond - v_uncond), x_t)


def _selftest() -> None:
    torch.manual_seed(0)
    d, n = 32, 8
    eps = sample_base((n, d))
    x1 = sample_base((n, d))
    t = torch.rand(n)

    # 1) endpoints of the geodesic
    x0, om = geodesic_interp(eps, x1, torch.zeros(n))
    xT, _ = geodesic_interp(eps, x1, torch.ones(n))
    print(f"[1] interp(t=0) == eps : {torch.allclose(x0, eps, atol=1e-5)}")
    print(f"[1] interp(t=1) == x1  : {torch.allclose(xT, x1, atol=1e-5)}")

    # 2) analytic velocity == finite difference of the geodesic
    h = 1e-4
    xa, _ = geodesic_interp(eps, x1, t - h)
    xb, _ = geodesic_interp(eps, x1, t + h)
    fd = (xb - xa) / (2 * h)
    an = conditional_velocity(eps, x1, om, t)
    err = (fd - an).abs().max().item()
    print(f"[2] |velocity - d/dt geodesic|_inf = {err:.3e}  (expect < 1e-2)")

    # 3) every intermediate state and velocity are on-manifold
    xt, _ = geodesic_interp(eps, x1, t)
    print(f"[3] max |x_t|-1 = {(xt.norm(dim=-1) - 1).abs().max().item():.2e}")
    print(f"[3] max <v,x_t>  = {(an * xt).sum(-1).abs().max().item():.2e}")

    # 4) tangent projection kills the radial part
    v = torch.randn(n, d)
    vp = tangent_project(v, xt)
    print(f"[4] max <proj(v),x_t> = {(vp * xt).sum(-1).abs().max().item():.2e}")

    # 5) closed-form integration reproduces x1 exactly when v is the true field
    steps = 200
    x = eps.clone()
    dt = 1.0 / steps
    for s in range(steps):
        ts = torch.full((n,), s * dt)
        x = riemannian_euler_step(x, conditional_velocity(eps, x1, om, ts), dt)
    print(f"[5] Euler(T={steps}) ||x_1 - x̂||_inf = {(x - x1).abs().max().item():.2e}")
    print(f"[5] Euler stays on sphere          : {(x.norm(dim=-1) - 1).abs().max().item():.2e}")

    # 6) Euclidean flow off-manifold (why RFM matters)
    xe, _ = linear_interp(eps, x1, torch.full((n,), 0.5))
    print(f"[6] Euclidean midpoint max | |x|-1 | = {(xe.norm(dim=-1) - 1).abs().max().item():.3f}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
