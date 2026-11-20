"""Depth metrics of Sec. S2 on the test views."""
from __future__ import annotations

import numpy as np
import torch

from .grid import Grid, camera_rays, polar
from .render import depth_map


def object_errors(dep: np.ndarray, ref: np.ndarray, mask: np.ndarray, rays: np.ndarray,
                  c2w: np.ndarray, obj_radius: float = 0.30) -> dict:
    """Depth MAE (m) over all pixels with the reference mask, and MAE_obj restricted to pixels
    whose reference point lies within ``obj_radius`` of the origin. Both are means over the
    whole 512x512 image, as in the paper tables."""
    xg = rays.reshape(-1, 3) @ polar(c2w[:3, :3]).T * ref.reshape(-1, 1) + c2w[:3, 3]
    objm = np.linalg.norm(xg, axis=1).reshape(ref.shape) <= obj_radius
    err = np.abs(ref - dep) * mask
    return dict(mae=float(err.mean()), mae_obj=float((err * objm).mean()))


@torch.no_grad()
def evaluate_objects(u: torch.Tensor, grid: Grid, rays: np.ndarray, test_c2w: list, refs: list,
                     step: float, ball: bool, obj_radius: float = 0.30):
    """Per-view and mean errors of an object field. ``refs`` holds (depth, mask) per test view."""
    dev = u.device
    per_view, depths = [], []
    for c2w, (ref, mask) in zip(test_c2w, refs):
        d, o = camera_rays(rays, c2w, np.zeros(3), dev)
        dep, _ = depth_map(u, o, d, grid, step, ball)
        dep = dep.reshape(512, 512).cpu().numpy()
        per_view.append(object_errors(dep, ref, mask, rays, c2w, obj_radius))
        depths.append(dep.astype(np.float32))
    mean = {k: float(np.mean([r[k] for r in per_view])) for k in per_view[0]}
    return mean, per_view, depths


@torch.no_grad()
def evaluate_room(u: torch.Tensor, grid: Grid, rays: np.ndarray, test_c2w: list, refs: list,
                  bin_size: float):
    """Room depth MAE (m): the read-out rounded to the sensor bin against the argmax-bin
    reference, on the reference mask. The read-out marches at one fifth of a voxel."""
    dev = u.device
    per_view, depths = [], []
    for c2w, (ref, mask) in zip(test_c2w, refs):
        d, o = camera_rays(rays, c2w, grid.center, dev)
        dep, _ = depth_map(u, o, d, grid, grid.h / 5, False)
        dep = dep.reshape(ref.shape)
        d_bin = torch.round(dep / bin_size) * bin_size
        per_view.append(dict(mae=float((d_bin - ref)[mask].abs().mean()),
                             mae_sub=float((dep - ref)[mask].abs().mean())))
        depths.append(dep.cpu().numpy().astype(np.float32))
    mean = {k: float(np.mean([r[k] for r in per_view])) for k in per_view[0]}
    return mean, per_view, depths
