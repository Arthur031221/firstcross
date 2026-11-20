"""Cubic voxel grid of the field u and the ray geometry on it.

The grid covers the cube [-half, half]^3 (metres, relative to ``center``) with ``n`` voxels per
side. Voxel centres sit at -half + (i + 0.5) h with h = 2 half / n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Grid:
    n: int
    half: float
    center: np.ndarray

    @property
    def h(self) -> float:
        """Voxel size in metres."""
        return 2 * self.half / self.n

    def to_vox(self, x: torch.Tensor) -> torch.Tensor:
        """Box coordinates (metres) to voxel coordinates with voxel centres at integers."""
        return (x + self.half) / self.h - 0.5


def ray_box(o: torch.Tensor, d: torch.Tensor, half: float):
    """Entry and exit range (metres) of rays o + t d through the cube [-half, half]^3."""
    inv = 1.0 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t0 = (-half - o) * inv
    t1 = (half - o) * inv
    return torch.minimum(t0, t1).amax(-1).clamp(min=0), torch.maximum(t0, t1).amin(-1)


def sample_field(u: torch.Tensor, x: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Trilinear value of u at points x (metres, box frame), clamped at the border."""
    n, h, half = grid.n, grid.h, grid.half
    g = 2 * ((x + half) / h - 0.5) / (n - 1) - 1
    return F.grid_sample(u[None, None], g[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3), mode='bilinear',
                         padding_mode='border', align_corners=True).reshape(-1)


def polar(m: np.ndarray) -> np.ndarray:
    """Closest rotation to a 3x3 matrix."""
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def camera_rays(dirs_cam: np.ndarray, c2w: np.ndarray, center: np.ndarray, device):
    """World ray directions (P, 3) and the camera centre in the box frame, as float32 tensors."""
    rot = polar(c2w[:3, :3])
    d = torch.tensor(dirs_cam.reshape(-1, 3) @ rot.T, dtype=torch.float32, device=device)
    o = torch.tensor(c2w[:3, 3] - center, dtype=torch.float32, device=device)
    return d, o
