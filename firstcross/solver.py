"""Convex solver sequence of Alg. S1 ("Photons to a solid") and the full reconstruction."""
from __future__ import annotations

import time

import numpy as np
import torch

from .config import Settings
from .grid import Grid, sample_field
from .likelihood import pin_weight, too_early_weights, too_late_weight
from .prior import BendingSystem
from .returns import Captures, Returns, detect, keep_inside


def band(cap: Captures, st: Settings) -> tuple[float, float]:
    """Band half width epsilon and occupancy margin m in metres (Alg. S1, line 3)."""
    eps = st.eps_sigma * cap.sigma if st.eps_sigma > 0 else st.eps
    margin = st.margin_sigma * cap.sigma if st.margin_sigma > 0 else st.margin
    return eps, margin


@torch.no_grad()
def solve(ret: Returns, cap: Captures, st: Settings, grid: Grid, rounds: int | None = None):
    """Rounds of Alg. S1 (lines 4 to 10). Returns the field u (voxel units, u > 0 inside) and a log.

    Round 1 starts from the empty field u = -m and uses the quadratic occupancy. Later rounds
    refresh the pin slope of Eq. (6), the margins m_r = m c_r with c_r the cosine between the ray
    and the field gradient (clamped to [cmin, 1]), and the witness of the too-late term of Eq. (4)
    at the point of [eps / 2, eps] behind the return closest to being satisfied. Too-early
    weights are damped across rounds. Each round solves one quadratic by PCG.
    """
    f32 = torch.float32
    dev = ret.x.device
    x, v, nr, tabs = ret.x, ret.v, ret.n, ret.tables
    eps, margin = band(cap, st)
    h, sig = grid.h, cap.sigma
    n_bins = cap.counts[0].shape[-1]
    wcap = st.weight_cap if st.weight_cap > 0 else 4 * st.neff * (h / sig) / st.tau_sigma
    tau = st.tau_sigma * h / sig
    m_v = margin / h
    a_vox = pin_weight(nr, sig, st.sig_s, h, tau)
    sysm = BendingSystem(grid.n, grid.to_vox(x), a_vox, dev)
    x_in = grid.to_vox(x - torch.full_like(nr, eps)[:, None] * v)
    w_full = too_late_weight(nr, st.omega, m_v, tau, wcap)
    sysm.set_side(x_in, w_full * 0, m_v)
    u = torch.full((grid.n,) * 3, -m_v, device=dev)
    for tb in tabs:
        tb['mr'] = torch.full(tb['shape'], m_v, device=dev, dtype=f32)
    act_prev = w_last = wt_last = None
    mr = None
    log = []
    for _ in range(st.rounds if rounds is None else rounds):
        w_side = w_full
        if act_prev is not None:
            gu = sysm.grad_at_pins(u)
            gt = (-(v * gu).sum(-1)).abs()
            sysm.a = a_vox / gt.clamp(min=st.gmin) ** 2
            cr = (gt / gu.norm(dim=-1).clamp(min=1e-3)).clamp(min=st.cmin, max=1.0)
            mr = m_v * cr
            sysm.c2 = mr
            w_full = too_late_weight(nr, st.omega, mr, tau, wcap)
            w_side = w_full
            i0 = 0
            for kc, row, col in ret.ids:
                tabs[kc]['mr'][row, col] = mr[i0:i0 + len(row)]
                i0 += len(row)
            if st.witness:
                sw = eps * torch.linspace(.5, 1., 6, device=dev)
                uw = sample_field(u, (x[:, None] - sw[None, :, None] * v[:, None]).reshape(-1, 3),
                                  grid).reshape(len(x), 6) / (sw / eps)[None]
                s_w = sw[5 - uw.flip(1).max(1).indices]
                fr = s_w / eps
                sysm.move_side(grid.to_vox(x - s_w[:, None] * v))
                sysm.c2 = mr * fr
                w_side = (w_full / fr ** 2).clamp(max=wcap)
        sysm.w2 = w_side * (sysm.sample_side(u) < sysm.c2).to(f32)
        w, wt = too_early_weights(u, tabs, grid, cap.bin0, cap.bin_width, cap.dt, n_bins, eps,
                                  saturate=act_prev is not None)
        if w_last is not None:
            w = (1 - st.damping) * w + st.damping * w_last
            wt = (1 - st.damping) * wt + st.damping * wt_last
        w_last, wt_last = w, wt
        tar = torch.where(w > 0, wt / w.clamp(min=1e-12), torch.full_like(u, -m_v))
        act = (w > 0) & (u > tar)
        if (st.act_tol > 0 and act_prev is not None
                and float((act != act_prev).to(f32).sum()) < st.act_tol * float(act.to(f32).sum())):
            break
        act_prev = act
        sysm.set_hinge((w / tau).clamp(max=wcap) * act.to(f32), tar)
        u, its, res = sysm.solve(u, iters=st.cg_iters, tol=st.cg_tol)
        log.append(dict(cg_iters=its, residual=res, active=int(act.sum())))
    return u, log


@torch.no_grad()
def fit_room_box(cap: Captures, st: Settings, device) -> tuple[np.ndarray, float]:
    """Room box of Sec. S6: central ``box_quantile`` of the returns, grown to hold the cameras
    and by ``box_margin`` metres, then the cube of the largest half width. Uses the torch
    generator (Poisson background per view, then a random subset of at most 500k returns)."""
    f32 = torch.float32
    eps, _ = band(cap, st)
    cap.center = np.zeros(3)
    ret = detect(cap, st, 1e4, eps, st.grid_size, device)
    cams = torch.tensor(np.array([c[:3, 3] for c in cap.c2w]), dtype=f32, device=device)
    w = (ret.n >= 1).nonzero()[:, 0]
    idx = w[torch.randperm(len(w), device=device)[:500000]]
    lo = torch.quantile(ret.x[idx], (1 - st.box_quantile) / 2, dim=0)
    hi = torch.quantile(ret.x[idx], (1 + st.box_quantile) / 2, dim=0)
    lo = torch.minimum(lo, cams.amin(0)) - st.box_margin
    hi = torch.maximum(hi, cams.amax(0)) + st.box_margin
    return ((lo + hi) / 2).cpu().numpy().astype(np.float64), float((hi - lo).amax() / 2)


@torch.no_grad()
def reconstruct(cap: Captures, st: Settings, device, room: bool = False, rounds: int | None = None):
    """Histograms to the field u of Alg. S1. Returns (u, grid, returns, info)."""
    torch.manual_seed(st.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(st.seed)
    t0 = time.time()
    if room:
        center, half = fit_room_box(cap, st, device)
    else:
        center, half = np.zeros(3), st.half_size
    cap.center = center
    grid = Grid(st.grid_size, half, center)
    eps, _ = band(cap, st)
    ret = keep_inside(detect(cap, st, half, eps, st.grid_size, device), half)
    u, log = solve(ret, cap, st, grid, rounds=rounds)
    if dev_is_cuda(device):
        torch.cuda.synchronize()
    info = dict(solve_time=time.time() - t0, samples=len(ret.x), photons=float(ret.n_raw.sum()),
                rounds=len(log), log=log)
    return u, grid, ret, info


def dev_is_cuda(device) -> bool:
    return torch.device(device).type == 'cuda'
