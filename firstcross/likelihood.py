"""Majoriser of the too-early term with running-maximum transmittance (Eqs. (2), (3) and (7)).

Along ray r the occupancy samples are x_k = g((u(r(xi_k)) + m_r) / m_r) at range steps of half
a voxel, with g the Charbonnier function g(s) = 2 (sqrt(1 + s_+^2) - 1). The too-early cost is the
Lovasz extension sum_k Delta_k xbar_k of Eq. (3), where xbar_k = max_{i <= k} x_i is the running
maximum and Delta_k = l_r(xi_k) - l_r(xi_{k+1}) are the steps of the non-increasing table
l_r(t) = omega [n_r(t) - q n_r]_+ + a_r(t) of Eq. (7). The weight of step k goes to the sample that
holds the running maximum, and the Charbonnier term is majorised by a quadratic in u
(Alg. S1, line 6). The result is a per-voxel weight W and target T with
sum_v W_v (u_v - T_v)^2 / 2 as the surrogate.
"""
from __future__ import annotations

import torch

from .grid import Grid


@torch.no_grad()
def too_early_weights(u: torch.Tensor, tables: list, grid: Grid, bin0: int, bin_width: int,
                      dt: float, n_bins: int, eps: float, saturate: bool, chunk: int = 2048):
    """Per-voxel weight W, weighted target sum W T, both (n, n, n), of the too-early surrogate.

    ``saturate`` False uses g(s) = s^2 (first round, Alg. S1 line 5), True uses the Charbonnier
    majoriser. Margins m_r are read from ``tables[v]['mr']`` in voxel units.
    """
    f32 = torch.float32
    n, h, half = grid.n, grid.h, grid.half
    dev = u.device
    st = h / 2
    w_sum = torch.zeros(n ** 3, device=dev, dtype=u.dtype)
    wt_sum = torch.zeros(n ** 3, device=dev, dtype=u.dtype)
    uf = u.reshape(-1)
    for tb in tables:
        ns = (torch.floor((tb['tx'] - tb['te']) / st) + 1).long()
        for start in range(0, len(tb['d']), chunk):
            d_ = tb['d'][start:start + chunk]
            te_ = tb['te'][start:start + chunk]
            ns_ = ns[start:start + chunk]
            row = tb['row'][start:start + chunk]
            col = tb['col'][start:start + chunk]
            k = torch.arange(int(ns_.max()), device=dev)[None]
            val = k < ns_[:, None]
            t = te_[:, None] + k.to(f32) * st
            x = tb['o'][None, None] + d_[:, None] * t[..., None]
            gi = torch.floor((x + half) / h).long().clamp(0, n - 1)
            idx = (gi[..., 0] * n + gi[..., 1]) * n + gi[..., 2]
            uu = uf[idx]
            kb = torch.floor(((t + eps) / dt - bin0 + 0.5) / bin_width).long().clamp(0, n_bins)
            c = tb['cb'][row[:, None], col[:, None], kb] \
                + tb['miss'][row, col][:, None] * (tb['at'][kb] if 'at' in tb else 1.0)
            mr = tb['mr'][row, col][:, None]
            x2 = ((uu + mr).clamp(min=0) / mr) ** 2
            if saturate:
                rho = 2 * (torch.sqrt(1 + x2) - 1)
                psi = 1 / torch.sqrt(1 + x2)
            else:
                rho = x2
                psi = torch.ones_like(x2)
            rv = rho * val
            on = rv.max(1, keepdim=True).values[:, 0] > 1e-9
            if not on.any():
                continue
            cv = c[on] * val[on]
            dc = (cv - torch.cat([cv[:, 1:], torch.zeros_like(cv[:, :1])], 1)).clamp(min=0)
            cm, am = torch.cummax(rv[on], 1)
            jump = dc * (cm > 1e-9).to(dc.dtype)
            pw = torch.zeros_like(cm).scatter_add_(1, am, jump) * (rv[on] > 0)
            w_ = (psi[on] * pw * 2 / mr[on] ** 2).reshape(-1).to(w_sum.dtype)
            w_sum.index_add_(0, idx[on].reshape(-1), w_)
            tar = (-mr[on].expand_as(pw)).reshape(-1)
            wt_sum.index_add_(0, idx[on].reshape(-1), w_ * tar)
    return w_sum.reshape(n, n, n), wt_sum.reshape(n, n, n)


def pin_weight(n_ret: torch.Tensor, sigma: float, sig_s: float, h: float,
               tau: float) -> torch.Tensor:
    """Pin weight of Eq. (6) before the slope factor, in prior units.

    h^2 / (sigma^2 / N_r + sig_s^2) / tau, sigma and sig_s in metres, h the voxel size."""
    return 1.0 * (h ** 2 / (sigma ** 2 / n_ret.clamp(min=1e-3) + sig_s ** 2)) / tau


def too_late_weight(n_ret: torch.Tensor, omega: float, margin_vox: torch.Tensor | float,
                    tau: float, cap: float) -> torch.Tensor:
    """Weight 2 omega N_r / m_r^2 of the too-late term of Eq. (4), in prior units, capped."""
    return (2.0 * (omega * n_ret) / margin_vox ** 2 / tau).clamp(max=cap)
