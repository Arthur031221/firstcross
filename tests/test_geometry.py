"""First crossing of a ball and the zero-truncated Poisson rate."""
import math

import numpy as np
import torch

from firstcross.grid import Grid
from firstcross.render import depth_map
from firstcross.returns import miss_rate


def test_first_crossing_of_a_ball():
    grid = Grid(64, 0.32, np.zeros(3))
    c = (torch.arange(64, dtype=torch.float32) + 0.5) * grid.h - grid.half
    x, y, z = torch.meshgrid(c, c, c, indexing='ij')
    radius = 0.2
    u = (radius - torch.sqrt(x ** 2 + y ** 2 + z ** 2)) / grid.h
    o = torch.tensor([0.0, 0.0, -1.0])
    d = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.5, 1.0]])
    d = d / d.norm(dim=-1, keepdim=True)
    dep, first = depth_map(u, o, d, grid, 0.001, ball=True)
    b = (o[None] * d).sum(-1)
    exact = -b - torch.sqrt(b ** 2 - (1.0 - radius ** 2))
    assert torch.all((dep[:2] - exact[:2]).abs() < 0.5 * grid.h)
    assert first[2] == 0


def test_miss_rate():
    for sbar in (1.5, 3.0, 20.0):
        lam = miss_rate(sbar)
        assert abs(lam / (1 - math.exp(-lam)) - sbar) < 1e-6 * sbar
    assert miss_rate(0.9) == 0.0


if __name__ == '__main__':
    test_first_crossing_of_a_ball()
    test_miss_rate()
    print('ok')
