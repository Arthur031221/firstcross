#!/usr/bin/env bash
# Room rows of the reproduction table: SP-TransientBench test rooms, three views, 1% flux.
set -e
cd "$(dirname "$0")/.."
CFG=configs/rooms.yaml

for room in c4floor2 material_building physics_building3
do
    out=output/${room}_0.01_3v
    python reconstruct.py --config $CFG --scene "$room" --ppp 0.01 --views three --out "$out"
    python evaluate.py --config $CFG --scene "$room" --field "${out}_field.npz" --views three
done
