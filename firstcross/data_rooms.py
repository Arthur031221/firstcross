"""Room scenes of SP-TransientBench (192x256 flash SPAD, 672 bins of 750 ps)."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch
import torch.nn.functional as F

from .returns import Captures

C_LIGHT = 299792458.0
EXPOSURE = 750e-12
BIN = C_LIGHT * EXPOSURE / 2     # one-way range of one bin (m)
H, W, T = 192, 256, 672


def parse_histogram(path: str) -> np.ndarray:
    """One RawDataHistogramMap text file to a uint16 array (H, W, T)."""
    with open(path) as f:
        a = np.array(f.read().split(), dtype=np.int64)
    assert a.size == H * W * T, a.size
    assert a.min() >= 0 and a.max() < 65535
    return a.astype(np.uint16).reshape(H, W, T)


def read_sensor(cfg_dir: str):
    """Unit pixel rays (H, W, 3), per-pixel time shift (m), valid pixels and pulse sigma (m)."""
    d = np.load(f'{cfg_dir}/intrinsics_safe.npz')
    rays = d['rays'].astype(np.float32)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    shift = d['shift'].astype(np.float32)
    valid = np.loadtxt(f'{cfg_dir}/offset.txt', dtype=np.float32).reshape(H, W) <= 10.0
    with open(f'{cfg_dir}/IRF_global.csv') as f:
        rd = csv.reader(f)
        next(rd)
        irf = np.array([[float(x) for x in r] for r in rd])
    ip = int(np.argmax(irf[:, 1]))
    w = np.clip(irf[ip - 50:ip + 51, 1], 0, None)
    w = w / w.sum()
    tt = irf[ip - 50:ip + 51, 0] * 1e-12
    sigma = float(np.sqrt((w * tt ** 2).sum() - (w * tt).sum() ** 2)) * C_LIGHT / 2
    return rays, shift, valid, sigma


def shift_histogram(h: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Per-pixel time shift of h (H, W, T).

    out[b] = in[b - 2 shift / (c exposure)], linear interpolation, zero padded."""
    dev, f32 = h.device, torch.float32
    ex = C_LIGHT * EXPOSURE
    z = torch.arange(T, device=dev, dtype=f32)[:, None, None] * ex / 2.0 - shift[None]
    z = z * 2.0 / ex
    z = 2.0 * z / float(T - 1) - 1.0
    x = 2.0 * torch.arange(W, device=dev, dtype=f32) / float(W - 1) - 1.0
    y = 2.0 * torch.arange(H, device=dev, dtype=f32) / float(H - 1) - 1.0
    grid = torch.stack((x[None, None, :].expand(T, H, W), y[None, :, None].expand(T, H, W),
                        z.expand(T, H, W)), -1)[None]
    return F.grid_sample(h.permute(2, 0, 1)[None, None], grid, align_corners=True)[0, 0] \
        .permute(1, 2, 0)


def read_room_views(src_dir: str, n_views: str):
    """(view id, c2w) lists of the train and test split, view ids follow the Livox file names."""
    with open(f'{src_dir}/sp_pose_results.csv') as f:
        rows = list(csv.DictReader(f))
    ids = {r['sp_file']: int(r['livox_csv_name'].split('.')[0]) for r in rows}

    def split(name):
        with open(f'{src_dir}/json/{n_views}_views/{name}.json') as f:
            frames = json.load(f)['frames']
        return [(ids[fr['file_path']], np.array(fr['transform_matrix'], dtype=np.float64))
                for fr in frames]
    return split('train'), split('test')


def thin_histogram(h: np.ndarray, p: float, view: int) -> np.ndarray:
    """Binomial thinning of a count histogram with keep probability p.

    The seed of view n is 1000 n + int(p 1e6) mod 997."""
    rng = np.random.default_rng(1000 * view + int(p * 1e6) % 997)
    return rng.binomial(h.astype(np.int64), p).astype(np.uint16)


def read_counts(paths: dict, room: str, level: str, view: int) -> np.ndarray:
    """Raw histogram of one view at a photon level ('full' or a keep fraction such as '0.01')."""
    clean = np.load(paths['hist'].format(**paths, room=room, view=view))
    if level == 'full':
        return clean
    return thin_histogram(clean, float(level), view)


def load_room(paths: dict, room: str, level: str, n_views: str, device):
    """Training captures of one room and its test split. Returns (captures, sensor, test views)."""
    rays, shift, valid, sigma = read_sensor(paths['sensor'].format(**paths))
    train, test = read_room_views(paths['poses'].format(**paths, room=room), n_views)
    sh = torch.tensor(shift, device=device)
    vt = torch.tensor(valid, device=device)
    counts = [shift_histogram(torch.tensor(read_counts(paths, room, level, k).astype(np.float32),
                                           device=device), sh) * vt[..., None].to(torch.float32)
              for k, _ in train]
    cap = Captures(counts, rays, [c for _, c in train], 0, 1, BIN, sigma, valid=vt)
    return cap, dict(rays=rays, shift=sh, valid=vt), test


def room_reference(paths: dict, room: str, view: int, sensor: dict):
    """Reference range (m) per pixel and its mask: argmax bin of the shifted full-flux histogram,
    on valid pixels whose peak holds at least 100 counts (Sec. S2)."""
    h = np.load(paths['hist'].format(**paths, room=room, view=view)).astype(np.float32)
    gt = shift_histogram(torch.tensor(h, device=sensor['valid'].device), sensor['shift'])
    gpk, gam = gt.max(-1)
    mask = sensor['valid'] & (gpk > 0) & (gpk >= 100.0)
    return gam.to(torch.float32) * BIN, mask
