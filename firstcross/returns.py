"""Photon returns and per-ray cost tables read off the histograms (Alg. S1, lines 1 to 3).

Ranges are one-way (metres). A ray is one histogram pixel (a 4x4 pixel block for objects).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .config import Settings
from .grid import camera_rays, ray_box


@dataclass
class Captures:
    """Training histograms of one scene.

    counts: per view, photon counts (R, RW, T) per ray and coarse time bin.
    dirs: (R * RW, 3) unit ray directions in the camera frame.
    c2w: per view, 4x4 camera-to-world matrix.
    bin0, bin_width: coarse bin k covers raw bins bin0 + k bin_width ... + bin_width - 1.
    dt: one-way range of one raw bin (m). sigma: range spread of the pulse (m).
    valid: (R, RW) usable pixels, or None.
    """
    counts: list
    dirs: np.ndarray
    c2w: list
    bin0: int
    bin_width: int
    dt: float
    sigma: float
    valid: torch.Tensor | None = None
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def range_of(self, k):
        """One-way range (m) of the centre of (fractional) coarse bin k."""
        return (self.bin0 + (k + 0.5) * self.bin_width - 0.5) * self.dt


def miss_rate(sbar: float) -> float:
    """Rate of the zero-truncated Poisson law with mean sbar (Eq. (S-trunc)).

    Solves lam / (1 - e^-lam) = sbar."""
    if sbar <= 1.0:
        return 0.0
    lam = sbar
    for _ in range(30):
        lam = sbar * (1 - math.exp(-lam))
    return lam


@dataclass
class Returns:
    """Detected returns (one per occupied ray) and the per-view cost tables."""
    x: torch.Tensor           # (M, 3) return positions, box frame (m)
    v: torch.Tensor           # (M, 3) unit vectors from the return towards the camera
    n: torch.Tensor           # (M,) weighted photon count z_r n_r of the return window
    n_raw: torch.Tensor       # (M,) photon count n_r before the ray weight
    ids: list                 # per view (view, row, col) of each return
    tables: list              # per view dict of ray geometry and the table of Eq. (7)


def _peak_count(cc: torch.Tensor, g: torch.Tensor, hw: int) -> torch.Tensor:
    r, rw, t = cc.shape
    s = F.conv1d(cc.reshape(-1, 1, t), g[None, None], padding=hw)
    kp = s[:, 0].argmax(-1)
    w = torch.arange(-hw, hw + 1, device=cc.device)[None]
    return cc.reshape(-1, t).gather(1, (kp[:, None] + w).clamp(0, t - 1)).sum(1).reshape(r, rw)


@torch.no_grad()
def detect(cap: Captures, st: Settings, half: float, eps: float, n_grid: int, device) -> Returns:
    """Background, returns and tables of every training view.

    Per view: the time profile b(t) of the background (Alg. S1, line 1), counts above
    b + bg_sigma sqrt(b), a local false discovery rate test against Poisson background with the
    same profile, which sets the smallest photon count of a return, the matched filter, its first
    response above ``first_frac`` of the peak and the centroid of the window (line 2), the
    surface rate a_r(t) = a0 / t^2 of rays without photons, and the too-early table
    omega [n_r(t) - q n_r]_+ of Eq. (7), indexed by the coarse bin of t + eps.
    Uses the torch generator once per view (Poisson background sample).
    """
    f32 = torch.float32
    sig, dt, m = cap.sigma, cap.dt, cap.bin_width
    sb = sig / (m * dt)
    hw = int(math.ceil(3 * sb))
    g = torch.exp(-.5 * (torch.arange(-hw, hw + 1, device=device).to(f32) / sb) ** 2)
    g = g / g.max()
    valid = cap.valid
    xs, vs, ns, nraw, ids, tabs = [], [], [], [], [], []
    for kc, c in enumerate(cap.counts):
        r, rw, t = c.shape
        dw, o = camera_rays(cap.dirs, cap.c2w[kc], cap.center, device)
        hb = c.reshape(-1, t).mean(0)
        b = float(hb.sort().values[:max(1, t // 4)].mean())
        bk = torch.maximum(c.reshape(-1, t).median(0).values, torch.full_like(hb, b))
        cn = (c - bk - st.bg_sigma * bk.sqrt()).clamp(min=0)
        if st.tmin > 0:
            cn[..., :int(st.tmin / dt)] = 0.0
        nb = torch.cat([torch.flip(torch.cumsum(torch.flip(cn, [-1]), -1), [-1]),
                        torch.zeros(r, rw, 1, device=device)], -1)
        tot = nb[..., 0]
        occ = tot > 0.5
        n_obs = _peak_count(cn, g, hw)
        null = (torch.poisson(bk.expand(r, rw, t)) - bk - st.bg_sigma * bk.sqrt()).clamp(min=0)
        n_null = _peak_count(null, g, hw)
        nmin = 1
        for n_ in range(1, 400):
            o_ = ((n_obs >= n_ - .5) & (n_obs < n_ + .5)).sum()
            z_ = ((n_null >= n_ - .5) & (n_null < n_ + .5)).sum()
            if z_ <= .5 * o_:
                nmin = n_
                break
        occ = occ & (n_obs >= nmin - .5)
        nb = nb * occ[..., None].to(f32)
        tot = tot * occ.to(f32)
        z = st.neff / (st.neff + tot) if st.neff > 0 else torch.ones_like(tot)
        cn = cn * z[..., None]
        nb = nb * z[..., None]
        tot = tot * z
        sbar = float(tot[occ].mean()) if occ.any() else 0.0
        lam = miss_rate(sbar)
        miss = (~occ).to(f32) * lam
        if valid is not None:
            miss = miss * valid.to(f32)
        s = F.conv1d(cn.reshape(-1, 1, t), g[None, None], padding=hw).reshape(r, rw, t)
        mx = s.max(-1, keepdim=True).values
        hit = (s >= st.first_frac * mx) & (mx > 0)
        k0 = hit.to(f32).argmax(-1)
        sel = occ.reshape(-1).nonzero()[:, 0]
        row, col = sel // rw, sel % rw
        kk = k0[row, col].long()
        w = torch.arange(0, 2 * hw + 1, device=device)[None]
        wk = (kk[:, None] + w).clamp(max=t - 1)
        kp = wk.gather(1, s[row, col].gather(1, wk).argmax(1, keepdim=True))[:, 0]
        w2 = torch.arange(-hw, hw + 1, device=device)[None]
        wk2 = (kp[:, None] + w2).clamp(0, t - 1)
        cw = cn[row, col].gather(1, wk2)
        nr = cw.sum(1)
        kh = (cw * wk2.to(f32)).sum(1) / nr.clamp(min=1e-6)
        good = nr > 1e-3
        tr = cap.range_of(kh[good])
        tab = {}
        if lam > 0:
            tk = cap.range_of(torch.arange(t, device=device).to(f32))
            at = lam * (tr ** 2).mean() / tk ** 2
            tab['at'] = torch.cat([at, at[-1:]])
            miss = (~occ).to(f32) * (valid.to(f32) if valid is not None else 1.0)
        dg = dw[sel[good]]
        xg = o[None] + dg * tr[:, None]
        xp = xg + dg * eps * 1.5
        inb = xp.abs().amax(-1) < half * (1 - 4.0 / n_grid)
        if st.ball:
            inb = inb & (xg.norm(dim=-1) < half)
        xs.append(xg[inb])
        vs.append(-dg[inb])
        ns.append(nr[good][inb])
        nraw.append((nr / z[row, col])[good][inb])
        ids.append((kc, (sel[good] // rw)[inb], (sel[good] % rw)[inb]))
        cb = st.omega * (nb - st.q * tot[..., None]).clamp(min=0)
        te, tx = ray_box(o[None], dw, half)
        s2 = (tx > te).nonzero()[:, 0]
        if valid is not None:
            s2 = s2[valid.reshape(-1)[s2]]
        tab.update(o=o, d=dw[s2], te=te[s2], tx=tx[s2], row=s2 // rw, col=s2 % rw, cb=cb, miss=miss,
                   shape=(r, rw))
        tabs.append(tab)
    return Returns(torch.cat(xs), torch.cat(vs), torch.cat(ns), torch.cat(nraw), ids, tabs)


def keep_inside(ret: Returns, half: float) -> Returns:
    """Drop returns outside 98% of the box and keep the per-view ids aligned."""
    inb = (ret.x.abs() < half * .98).all(1)
    ids, i0 = [], 0
    for kc, r_, c_ in ret.ids:
        k = inb[i0:i0 + len(r_)]
        ids.append((kc, r_[k], c_[k]))
        i0 += len(r_)
    return Returns(ret.x[inb], ret.v[inb], ret.n[inb], ret.n_raw[inb], ids, ret.tables)
