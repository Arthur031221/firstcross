"""Depth read-out of the field (Alg. S1, line 11): range of the first zero crossing of u."""
from __future__ import annotations

import torch

from .grid import Grid, ray_box, sample_field


@torch.no_grad()
def first_crossing(u: torch.Tensor, o: torch.Tensor, d: torch.Tensor, grid: Grid, step: float,
                   ball: bool, chunk: int = 65536, k: int = 64):
    """One-way range (m) of the first zero of u along rays o + t d.

    Marches at ``step`` metres and interpolates linearly between the last negative and the first
    positive sample. Returns (t_first, t_closest): t_first is NaN for rays that never cross,
    t_closest is the range of the largest value of u along the ray (closest approach).
    With ``ball`` the samples outside |x| < half count as empty.
    """
    f32 = torch.float32
    dev = u.device
    te, tx = ray_box(o[None], d, grid.half)
    ts = torch.full((len(d),), float('nan'), device=dev)
    ts2 = torch.full((len(d),), float('nan'), device=dev)
    best = torch.full((len(d),), -1e9, device=dev)
    for r0 in range(0, len(d), chunk):
        rid = torch.arange(r0, min(r0 + chunk, len(d)), device=dev)
        rid = rid[tx[rid] > te[rid]]
        prev = torch.full((len(rid),), -1e9, device=dev)
        k0 = 0
        while len(rid) > 0:
            s = te[rid][:, None] + (k0 + torch.arange(k, device=dev)[None]).to(f32) * step
            inb = s < tx[rid][:, None]
            p = (o[None, None] + d[rid][:, None] * s[..., None]).reshape(-1, 3)
            val = sample_field(u, p, grid).reshape(len(rid), k)
            vin = (inb & (p.norm(dim=-1).reshape(len(rid), k) < grid.half)) if ball else inb
            val = torch.where(vin, val, torch.full_like(val, -1e9))
            hit = val > 0
            vm, km = val.max(1)
            bt = vm > best[rid]
            best[rid[bt]] = vm[bt]
            ts2[rid[bt]] = s.gather(1, km[:, None])[:, 0][bt]
            any_ = hit.any(1)
            kh = hit.to(f32).argmax(1)
            vh = val.gather(1, kh[:, None])[:, 0]
            vp = torch.where(kh > 0, val.gather(1, (kh - 1).clamp(min=0)[:, None])[:, 0], prev)
            sh = s.gather(1, kh[:, None])[:, 0]
            fr = torch.where(vp > -1e8, ((0 - vp) / (vh - vp).clamp(min=1e-12)).clamp(0, 1),
                             torch.ones_like(vh))
            ts[rid[any_]] = (sh - step * (1 - fr))[any_]
            alive = ~any_ & inb[:, -1]
            prev = val[:, -1][alive]
            rid = rid[alive]
            k0 += k
    return ts, ts2


def depth_map(u: torch.Tensor, o: torch.Tensor, d: torch.Tensor, grid: Grid, step: float,
              ball: bool):
    """Depth (m) per ray: first crossing, or the closest approach for rays that miss the solid.
    Also returns the first crossing alone (0 where there is none)."""
    ts, ts2 = first_crossing(u, o, d, grid, step, ball)
    dep = torch.where(torch.isfinite(ts), ts, torch.nan_to_num(ts2, nan=0.0))
    return dep, torch.nan_to_num(ts, nan=0.0)
