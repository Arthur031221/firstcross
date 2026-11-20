"""Reconstruct a solid from photon histograms with FirstCross (Alg. S1).

Example:
    python reconstruct.py --config configs/objects.yaml --scene chef --ppp 0.1 --views three \
        --out output/chef_0.1_3v
writes output/chef_0.1_3v_field.npz (field u on the grid) and output/chef_0.1_3v.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from firstcross.config import Settings, load_config
from firstcross.data_objects import load_object
from firstcross.data_rooms import load_room
from firstcross.solver import reconstruct


def apply_overrides(cfg: dict, args) -> Settings:
    st = cfg['settings']
    if args.data_root:
        cfg['data']['root'] = args.data_root
    for kv in args.path:
        k, v = kv.split('=', 1)
        cfg['data'][k] = v
    if args.grid_size:
        st.grid_size = args.grid_size
    if args.rounds:
        st.rounds = args.rounds
    return st


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True, help='configs/objects.yaml or configs/rooms.yaml')
    ap.add_argument('--scene', required=True, help='object scene or room name')
    ap.add_argument('--ppp', default='full',
                    help="objects: full, 300, 150, 50, 10, 1, 0.1, 0.03, 0.01, 0.003, "
                         "rooms: full or a keep fraction such as 0.01")
    ap.add_argument('--views', default='three', choices=['two', 'three', 'five'])
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--data_root', default='')
    ap.add_argument('--path', action='append', default=[],
                    help='override a data path, key=template')
    ap.add_argument('--grid_size', type=int, default=0, help='override the grid size')
    ap.add_argument('--rounds', type=int, default=0, help='override the number of rounds')
    return ap


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    st = apply_overrides(cfg, args)
    room = cfg.get('kind') == 'rooms'
    if room:
        cap, _, _ = load_room(cfg['data'], args.scene, args.ppp, args.views, args.device)
    else:
        scene = dict(cfg['scenes'][args.scene], name=args.scene)
        cap, _, _, _ = load_object(cfg['data'], scene, args.ppp, args.views, args.device)
    u, grid, ret, info = reconstruct(cap, st, args.device, room=room)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out + '_field.npz', u=u.cpu().numpy().astype(np.float32),
                        n=grid.n, half=grid.half, center=grid.center)
    info.update(scene=args.scene, ppp=args.ppp, views=args.views)
    with open(args.out + '.json', 'w') as f:
        json.dump(info, f, indent=1)
    print(f'{args.scene} {args.ppp} {args.views}: {info["samples"]} returns, '
          f'{info["rounds"]} rounds, {info["solve_time"]:.1f} s')


if __name__ == '__main__':
    main()
