"""Method settings. Defaults are the object settings of the paper, rooms override a few."""
from __future__ import annotations

from dataclasses import dataclass, fields

import yaml


@dataclass
class Settings:
    # grid
    grid_size: int = 256          # voxels per side
    half_size: float = 0.32       # half width of the object box (m), rooms fit their own box
    ball: bool = True             # objects: returns and read-out restricted to |x| < half_size
    # likelihood (Eqs. (1) to (7))
    omega: float = 3.7            # cost cap omega of one photon (nats)
    q: float = 0.1                # photon fraction allowed in front of the crossing
    eps: float = 0.012            # band half width epsilon (m), see eps_sigma
    eps_sigma: float = 0.0        # > 0 sets eps = eps_sigma sigma
    margin: float = 0.006         # occupancy margin m (m), see margin_sigma
    margin_sigma: float = 0.0     # > 0 sets m = margin_sigma sigma
    sig_s: float = 1e-4           # range floor of the pin variance (m)
    neff: float = 20.0            # ray weight z_r = neff / (neff + n_r)
    tau_sigma: float = 2.646      # prior weight tau, in units of voxel / sigma
    weight_cap: float = 0.0       # per-voxel weight cap, 0 = alpha / (beta sigma)^2 rule of Alg. S1
    gmin: float = 0.2             # floor of |d_t u| in the pin of Eq. (6)
    cmin: float = 0.25            # floor of the margin refresh c_r
    # detection (Alg. S1, lines 1 and 2)
    bg_sigma: float = 3.0         # background floor b(t) + bg_sigma sqrt(b(t))
    first_frac: float = 0.5       # first matched-filter response above this fraction of the peak
    tmin: float = 0.0             # sensor dead zone (m), rooms only
    # solver (Alg. S1, lines 4 to 10)
    rounds: int = 12
    damping: float = 0.5
    cg_iters: int = 400
    cg_tol: float = 1e-4
    act_tol: float = 1e-3         # stop when the hinge active set changes by less than this
    witness: bool = True
    # rooms (Sec. S6)
    box_quantile: float = 0.9
    box_margin: float = 1.0
    # read-out (Alg. S1, line 11)
    march_step: float = 0.001     # metres, 0 = one fifth of a voxel
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> 'Settings':
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f'unknown settings {sorted(unknown)}')
        return cls(**d)


def load_config(path: str) -> dict:
    """Read a YAML config with sections ``settings``, ``data`` and ``scenes``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['settings'] = Settings.from_dict(cfg.get('settings', {}))
    return cfg
