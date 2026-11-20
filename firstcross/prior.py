"""Bending prior on a periodic voxel grid and the quadratic solve of each round.

Each round of Alg. S1 minimises a quadratic in the field u (voxel units):

    1/2 <u, L^2 u> + 1/2 sum_i a_i u(x_i)^2 + 1/2 sum_j w_j (u(y_j) - c_j)^2
                   + 1/2 sum_v k_v (u_v - t_v)^2,

with L the 7-point Laplacian on a periodic grid (the bending prior of Eq. (7)), the pins
x_i at the returns (Eq. (6)), the one-sided samples y_j of the too-late term (Eq. (4)) and
the per-voxel weights k_v and targets t_v of the too-early term (Eqs. (3) and (7)).
Samples are trilinear. The solve is preconditioned conjugate gradients with an FFT
preconditioner (Alg. S1, line 9).
"""
from __future__ import annotations

import math

import torch


def trilinear_stencil(x_vox: torch.Tensor, n: int, dtype=torch.float32):
    """Flat indices (8, M) and weights (8, M) of trilinear sampling at voxel coordinates."""
    i0 = torch.floor(x_vox).long().clamp(0, n - 2)
    f = (x_vox - i0).to(dtype)
    lin, wts = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wts.append((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                           * (f[:, 2] if dz else 1 - f[:, 2]))
                lin.append(((i0[:, 0] + dx) * n + (i0[:, 1] + dy)) * n + (i0[:, 2] + dz))
    return torch.stack(lin), torch.stack(wts)


class BendingSystem:
    """Normal equations of one round: bending prior plus pins, one-sided samples and hinges.

    Args:
        n: grid size (voxels per side).
        pins: (M, 3) pin positions in voxel coordinates (voxel centres at integers).
        pin_weight: (M,) pin weights a_i.
        device, dtype: torch device and dtype of the field.
    """

    def __init__(self, n: int, pins: torch.Tensor, pin_weight: torch.Tensor,
                 device: torch.device, dtype=torch.float32):
        self.n, self.dev, self.dt = n, device, dtype
        self.lin, self.wts = trilinear_stencil(pins, n, dtype)
        self.a = pin_weight.to(dtype)
        kf = torch.fft.fftfreq(n, d=1.0).to(device) * 2 * math.pi
        kr = torch.fft.rfftfreq(n, d=1.0).to(device) * 2 * math.pi

        def sym(k):
            return 2 - 2 * torch.cos(k)

        lap = (sym(kf)[:, None, None] + sym(kf)[None, :, None] + sym(kr)[None, None, :]).to(dtype)
        self.lp = lap ** 2
        self.eta = float(self.a.sum() / n ** 3)
        self.pre = 1.0 / (self.lp + self.eta)
        self.kdiag = None
        self.ktar = None
        self.lin2 = self.wts2 = self.w2 = self.c2 = None

    def set_hinge(self, kdiag: torch.Tensor, target: torch.Tensor) -> None:
        """Per-voxel quadratic k_v (u_v - t_v)^2 / 2. It also refreshes the preconditioner."""
        self.kdiag, self.ktar = kdiag, target
        self.pre = 1.0 / (self.lp + self.eta + float(kdiag.mean()))

    def set_side(self, x_vox: torch.Tensor, w: torch.Tensor, c) -> None:
        """One-sided sample list. The caller sets w to zero where the hinge is inactive."""
        self.lin2, self.wts2 = trilinear_stencil(x_vox, self.n, self.dt)
        self.w2 = w.to(self.dt)
        self.c2 = c.to(self.dt) if torch.is_tensor(c) else torch.full_like(self.w2, c)

    def move_side(self, x_vox: torch.Tensor) -> None:
        """Move the one-sided samples (witness points) and keep their weights."""
        self.lin2, self.wts2 = trilinear_stencil(x_vox, self.n, self.dt)

    def sample(self, u: torch.Tensor) -> torch.Tensor:
        return (u.reshape(-1)[self.lin] * self.wts).sum(0)

    def splat(self, v: torch.Tensor) -> torch.Tensor:
        n = self.n
        out = torch.zeros(n ** 3, device=self.dev, dtype=self.dt)
        vals = (self.wts * v[None]).reshape(-1)
        return out.index_add_(0, self.lin.reshape(-1), vals).reshape(n, n, n)

    def sample_side(self, u: torch.Tensor) -> torch.Tensor:
        return (u.reshape(-1)[self.lin2] * self.wts2).sum(0)

    def splat_side(self, v: torch.Tensor) -> torch.Tensor:
        n = self.n
        out = torch.zeros(n ** 3, device=self.dev, dtype=self.dt)
        vals = (self.wts2 * v[None]).reshape(-1)
        return out.index_add_(0, self.lin2.reshape(-1), vals).reshape(n, n, n)

    def grad_at_pins(self, u: torch.Tensor) -> torch.Tensor:
        """(M, 3) central-difference gradient of u at the pins, voxel/voxel."""
        def d(ax):
            return 0.5 * (torch.roll(u, -1, ax) - torch.roll(u, 1, ax))
        return torch.stack([self.sample(d(ax)) for ax in range(3)], -1)

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        n = self.n
        out = torch.fft.irfftn(torch.fft.rfftn(u) * self.lp, s=(n,) * 3)
        out = out + self.splat(self.a * self.sample(u))
        if self.kdiag is not None:
            out = out + self.kdiag * u
        if self.w2 is not None:
            out = out + self.splat_side(self.w2 * self.sample_side(u))
        return out

    def rhs(self) -> torch.Tensor:
        r = torch.zeros(self.n, self.n, self.n, device=self.dev, dtype=self.dt)
        if self.kdiag is not None:
            r = r + self.kdiag * self.ktar
        if self.w2 is not None:
            r = r + self.splat_side(self.w2 * self.c2)
        return r

    def precondition(self, r: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(torch.fft.rfftn(r) * self.pre, s=(self.n,) * 3)

    @torch.no_grad()
    def solve(self, u0: torch.Tensor, iters: int = 400, tol: float = 1e-4):
        """PCG from u0. Returns the field, the number of iterations and the relative residual.

        The residual is tested every 25 iterations against ``tol``.
        """
        u = u0.clone()
        b = self.rhs()
        b0 = float(b.norm())
        if b0 == 0:
            return torch.zeros_like(u), 0, 0.0
        r = b - self.apply(u)
        z = self.precondition(r)
        p = z.clone()
        rz = (r * z).sum()
        it = 0
        for it in range(iters):
            ap = self.apply(p)
            al = rz / (p * ap).sum()
            u += al * p
            r -= al * ap
            if (it + 1) % 25 == 0 and float(r.norm()) / b0 < tol:
                break
            z = self.precondition(r)
            rz2 = (r * z).sum()
            p = z + (rz2 / rz) * p
            rz = rz2
        return u, it + 1, float(r.norm()) / b0
