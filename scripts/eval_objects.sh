#!/usr/bin/env bash
# Object rows of the reproduction table: Transientangelo captures, three views,
# the four test scenes at 0.1 PPP, chef at 0.003 PPP and chef at full flux.
set -e
cd "$(dirname "$0")/.."
CFG=configs/objects.yaml

run() {
    local scene=$1 ppp=$2
    local out=output/${scene}_${ppp}_3v
    python reconstruct.py --config $CFG --scene "$scene" --ppp "$ppp" --views three --out "$out"
    python evaluate.py --config $CFG --scene "$scene" --field "${out}_field.npz"
}

for scene in baskets boots chef food
do
    run "$scene" 0.1
done
run chef 0.003
run chef full
