"""Depth errors of a reconstructed field on the test views (Sec. S2).

Example:
    python evaluate.py --config configs/objects.yaml --scene chef \
        --field output/chef_0.1_3v_field.npz
Objects report the depth MAE and MAE_obj in millimetres, rooms the depth MAE in metres.
Reference depths of the objects are computed once from the full-flux test views and cached.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from firstcross.config import load_config
from firstcross.data_objects import gt_depth, load_photons, pixel_rays, read_pulse, read_views
from firstcross.data_rooms import BIN as ROOM_BIN
from firstcross.data_rooms import read_room_views, read_sensor, room_reference
from firstcross.grid import Grid
from firstcross.metrics import evaluate_objects, evaluate_room


def object_references(paths: dict, scene: str, test_ids: list, cache: str) -> list:
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return [(z[f'depth_{i}'], z[f'mask_{i}']) for i in test_ids]
    fmt = dict(paths, scene=scene)
    shift = float(np.load(paths['intrinsics'].format(**fmt))['shift'][0])
    pulse, _ = read_pulse(paths['pulse'].format(**fmt))
    refs = []
    for i in test_ids:
        idx, val = load_photons(paths['full_flux'].format(**fmt, view=i))
        refs.append(gt_depth(idx, val, shift, pulse))
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez_compressed(cache, **{f'depth_{i}': r[0] for i, r in zip(test_ids, refs)},
                            **{f'mask_{i}': r[1] for i, r in zip(test_ids, refs)})
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--field', required=True, help='*_field.npz written by reconstruct.py')
    ap.add_argument('--views', default='three', choices=['two', 'three', 'five'],
                    help='rooms: split whose test views are used')
    ap.add_argument('--test_views', default='', help='comma separated subset of test view ids')
    ap.add_argument('--out', default='', help='output prefix (default: next to the field)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--data_root', default='')
    ap.add_argument('--path', action='append', default=[],
                    help='override a data path, key=template')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    paths, st = cfg['data'], cfg['settings']
    if args.data_root:
        paths['root'] = args.data_root
    for kv in args.path:
        k, v = kv.split('=', 1)
        paths[k] = v
    z = np.load(args.field)
    grid = Grid(int(z['n']), float(z['half']), z['center'].astype(np.float64))
    u = torch.tensor(z['u'], device=args.device, dtype=torch.float32)
    keep = [int(x) for x in args.test_views.split(',')] if args.test_views else None
    if cfg.get('kind') == 'rooms':
        rays, shift, valid, _ = read_sensor(paths['sensor'].format(**paths))
        sensor = dict(shift=torch.tensor(shift, device=args.device),
                      valid=torch.tensor(valid, device=args.device))
        _, test = read_room_views(paths['poses'].format(**paths, room=args.scene), args.views)
        test = [(k, c) for k, c in test if keep is None or k in keep]
        refs = [room_reference(paths, args.scene, k, sensor) for k, _ in test]
        mean, per_view, depths = evaluate_room(u, grid, rays, [c for _, c in test], refs, ROOM_BIN)
        ids = [k for k, _ in test]
        line = f'depth MAE {mean["mae"]:.3f} m'
    else:
        cams = paths['cameras'].format(**dict(paths, scene=args.scene))
        ids, c2w = read_views(f'{cams}/test_jsons/transforms_test.json')
        sel = [j for j, i in enumerate(ids) if keep is None or i in keep]
        ids, c2w = [ids[j] for j in sel], [c2w[j] for j in sel]
        cache = paths.get('reference_cache', '').format(**dict(paths, scene=args.scene))
        refs = object_references(paths, args.scene, ids, cache if keep is None else '')
        rays = pixel_rays(paths['intrinsics'].format(**dict(paths, scene=args.scene)))
        mean, per_view, depths = evaluate_objects(u, grid, rays, c2w, refs, st.march_step, st.ball)
        line = f'depth MAE {mean["mae"] * 1e3:.3f} mm, MAE_obj {mean["mae_obj"] * 1e3:.3f} mm'
    out = args.out or args.field.replace('_field.npz', '')
    with open(out + '_metrics.json', 'w') as f:
        json.dump(dict(mean=mean, per_view={str(i): r for i, r in zip(ids, per_view)}), f, indent=1)
    np.savez_compressed(out + '_depths.npz', **{f'view_{i}': d for i, d in zip(ids, depths)})
    print(f'{args.scene}: {line}')


if __name__ == '__main__':
    main()
