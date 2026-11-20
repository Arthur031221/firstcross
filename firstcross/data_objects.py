"""Captured object scenes of the Transientangelo release (512x512 SPAD, 4 ps bins).

Per view: photons are optionally thinned (lower photon counts), cut to the foreground mask,
resampled in time with the sensor ``shift``, pair-averaged to 1500 bins of 8 ps two-way and
averaged over 4x4 pixel blocks, giving a 128x128 grid of rays.
"""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import scipy.interpolate
import scipy.io
import torch
from PIL import Image
from scipy.ndimage import correlate1d

from .returns import Captures

C_LIGHT = 299792458.0
BIN = C_LIGHT * 4e-12      # one-way range of one 8 ps two-way bin (m)
N_RAW = 3000
N_BINS = 1500
ROWS = 32                  # image rows per chunk


def read_views(path: str):
    """View ids and 4x4 camera-to-world matrices from a transforms json."""
    with open(path) as f:
        frames = json.load(f)['frames']
    ids = [int(fr['file_path'].split('_')[-1]) for fr in frames]
    return ids, np.array([fr['transform_matrix'] for fr in frames], np.float64)


def load_photons(path: str):
    """Sparse photon counts of one view: indices (3, P) = (row, col, bin) and values (P,)."""
    d = torch.load(path, weights_only=True).coalesce()
    return d.indices(), d.values()


def thin_photons(idx: torch.Tensor, val: torch.Tensor, p: float, view: int):
    """Binomial thinning with keep probability p.

    The seed of view n is 1000 n + int(p 1e6) mod 997."""
    rng = np.random.default_rng(1000 * view + int(p * 1e6) % 997)
    val = torch.from_numpy(rng.binomial(val.round().numpy().astype(np.int64), p))
    nz = val > 0
    return idx[:, nz], val[nz]


def _time_resampler(shift: float):
    z = torch.arange(N_BINS * 2).float()
    zz = (z * BIN / 2 - shift) * 2 / BIN
    zn = (zz - N_BINS * 2 // 2 + 0.5) / (N_BINS * 2 // 2 - 0.5)
    src = (zn + 1) / 2 * (N_RAW - 1)
    i0 = src.floor().long().clamp(0, N_RAW - 1)
    i1 = (i0 + 1).clamp(0, N_RAW - 1)
    return i0, i1, src - src.floor(), (src >= 0) & (src <= N_RAW - 1)


def _rows(idx, val, shift):
    """Yield (r0, r), r (ROWS, 512, 1500) the resampled rate of rows r0 + [0, ROWS)."""
    i0, i1, w1, valid = _time_resampler(shift)
    keep = idx[2] < N_RAW
    idx, val = idx[:, keep], val[keep].float()
    for r0 in range(0, 512, ROWS):
        sel = (idx[0] >= r0) & (idx[0] < r0 + ROWS)
        dd = torch.zeros(ROWS, 512, N_RAW)
        dd[idx[0, sel] - r0, idx[1, sel], idx[2, sel]] = val[sel]
        r = (dd[..., i0] * (1 - w1) + dd[..., i1] * w1) * valid
        yield r0, torch.clamp((r[..., 1::2] + r[..., ::2]) / 2, min=0)


def pool_view(idx, val, shift: float, mask: np.ndarray | None = None):
    """Block histogram (128, 128, 1500) in half precision and the largest unmasked bin value."""
    out = np.zeros((128, 128, N_BINS), np.float16)
    mt = None if mask is None else torch.from_numpy(mask)
    peak = 0.0
    for r0, r in _rows(idx, val, shift):
        peak = max(peak, r.max().item())
        if mt is not None:
            r = r * mt[r0:r0 + ROWS, :, None].to(r.dtype)
        out[r0 // 4:(r0 + ROWS) // 4] = r.reshape(ROWS // 4, 4, 128, 4, N_BINS).mean((1, 3)) \
            .numpy().astype(np.float16)
    return out, peak


def gt_depth(idx, val, shift: float, pulse: np.ndarray):
    """Reference depth (512, 512) in metres and its mask from a full-flux view: argmax of the
    histogram correlated with the time-reversed pulse. The mask holds pixels with photons."""
    dep = np.zeros((512, 512), np.int16)
    tot = np.zeros((512, 512), np.float32)
    for r0, r in _rows(idx, val, shift):
        tot[r0:r0 + ROWS] = r.sum(-1).numpy()
        dep[r0:r0 + ROWS] = correlate1d(r.numpy(), pulse, axis=-1).argmax(-1).astype(np.int16)
    return dep.astype(np.float32) * np.float32(BIN), tot > 0


def read_pulse(path: str):
    """Pulse kernel (101 taps, time-reversed, unit sum) and its range spread sigma (m)."""
    laser = scipy.io.loadmat(path)['out'].squeeze()
    laser = (laser[::2] + laser[1::2]) / 2
    li = np.argmax(laser)
    lk = laser[li - 50:li + 51]
    lk = lk / lk.sum()
    bb = np.arange(-50, 51)
    sigma = float(np.sqrt((lk * bb ** 2).sum() - (lk * bb).sum() ** 2)) * BIN
    return lk[::-1].copy(), sigma


def pixel_rays(intrinsics: str) -> np.ndarray:
    """Unit ray directions (512, 512, 3) in the camera frame, float32, from the calibrated rays."""
    rays = np.load(intrinsics)['rays']
    x = np.arange(32, 480)
    tx = np.arange(0, 512)
    gx, gy = np.meshgrid(tx, tx)
    pts = np.stack([gx, gy], axis=-1).squeeze().reshape(-1, 2)
    ch = [scipy.interpolate.interpn((x, x), rays[32:-32, 32:-32, c].transpose(1, 0), pts,
                                    bounds_error=False, fill_value=None).reshape(512, 512)
          for c in range(3)]
    r = np.stack(ch, -1)
    return (r / np.linalg.norm(r, axis=-1, keepdims=True)).astype(np.float32)


def block_dirs(rays: np.ndarray, res: int = 128) -> np.ndarray:
    """Mean direction of each block of a res x res grid over the 512x512 pixel rays."""
    k = rays.shape[0] // res
    r = rays.reshape(res, k, res, k, 3).mean((1, 3))
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def block_counts(pooled: list, scale: float, bin0: int, n_bins: int, width: int, device):
    """Photon counts per block and coarse bin, (128, 128, n_bins) per view.

    Each coarse bin sums ``width`` pooled bins starting at ``bin0``. Rates are normalised by
    ``scale`` (the largest training bin) and scaled back, so the counts carry the rounding of
    the paper experiments.
    """
    out = []
    for p in pooled:
        y = (p.astype(np.float32) / np.float32(scale))[..., bin0:bin0 + n_bins * width]
        y = y.reshape(128, 128, n_bins, width).mean(-1)
        out.append(torch.tensor(y, device=device) * (scale * 16 * width * 2.0))
    return out


def flux_source(level: str, full_level: str = '10'):
    """(official level to read, keep probability) for a photon level in PPP.

    ``full`` reads the full-flux capture. Official low-flux levels are read as they are, and
    lower levels are thinned from level 10 with p = level / 10, rounded from the exact decimal.
    """
    if level == 'full':
        return 'full', 1.0
    if level in ('300', '150', '50', '10'):
        return level, 1.0
    return full_level, float(Fraction(level) / 10)


def load_object(paths: dict, scene: dict, level: str, n_views: str, device) -> tuple:
    """Training captures and test cameras of one object scene.

    ``paths`` holds format strings with fields root, scene, level, view (see configs). ``scene``
    holds name, bin0 and n_bins. Returns (captures, pixel rays, test ids, test c2w).
    """
    name = scene['name']
    fmt = dict(paths, scene=name)
    cams = paths['cameras'].format(**fmt)
    tr_ids, tr_c2w = read_views(f'{cams}/{n_views}_views/transforms_train.json')
    te_ids, te_c2w = read_views(f'{cams}/test_jsons/transforms_test.json')
    shift = float(np.load(paths['intrinsics'].format(**fmt))['shift'][0])
    _, sigma = read_pulse(paths['pulse'].format(**fmt))
    src, p = flux_source(level)
    pooled, peaks = [], []
    for n in tr_ids:
        if src == 'full':
            idx, val = load_photons(paths['full_flux'].format(**fmt, view=n))
            mask = None
        else:
            idx, val = load_photons(paths['low_flux'].format(**fmt, level=src, view=n))
            if p < 1:
                idx, val = thin_photons(idx, val, p, n)
            mask = np.array(Image.open(paths['masks'].format(**fmt, view=n))) > 127
        pv, pk = pool_view(idx, val, shift, mask)
        pooled.append(pv)
        peaks.append(pk)
    scale = float(np.float32(max(peaks)))
    counts = block_counts(pooled, scale, scene['bin0'], scene['n_bins'], 2, device)
    rays = pixel_rays(paths['intrinsics'].format(**fmt))
    cap = Captures(counts, block_dirs(rays), list(tr_c2w), scene['bin0'], 2, BIN, sigma)
    return cap, rays, te_ids, list(te_c2w)
