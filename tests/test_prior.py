"""The quadratic of each round is symmetric positive definite and PCG solves it."""
import torch

from firstcross.prior import BendingSystem


def make_system(n=16, m=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pins = torch.rand(m, 3, generator=g) * (n - 2) + 0.5
    sysm = BendingSystem(n, pins, torch.rand(m, generator=g) + 0.1, torch.device('cpu'))
    side = torch.rand(m, 3, generator=g) * (n - 2) + 0.5
    sysm.set_side(side, torch.rand(m, generator=g), 0.5)
    sysm.set_hinge(torch.rand(n, n, n, generator=g) * (torch.rand(n, n, n, generator=g) > .7),
                   torch.randn(n, n, n, generator=g))
    return sysm, g


def test_symmetric():
    sysm, g = make_system()
    x, y = torch.randn(2, 16, 16, 16, generator=g, dtype=torch.float64).float()
    lhs, rhs = float((sysm.apply(x) * y).sum()), float((x * sysm.apply(y)).sum())
    assert abs(lhs - rhs) <= 1e-4 * max(abs(lhs), 1.0)
    assert float((sysm.apply(x) * x).sum()) > 0


def test_pcg_solves():
    sysm, _ = make_system()
    u, its, res = sysm.solve(torch.zeros(16, 16, 16), iters=400, tol=1e-5)
    rel = float((sysm.rhs() - sysm.apply(u)).norm() / sysm.rhs().norm())
    assert res < 1e-5 and rel < 1e-4


if __name__ == '__main__':
    test_symmetric()
    test_pcg_solves()
    print('ok')
