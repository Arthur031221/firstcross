#!/usr/bin/env bash
# Download captured scenes of Transientangelo (Luo et al., WACV 2025) from its official Hugging Face
# release into data/transientangelo, in the layout read by configs/objects.yaml.
# Usage: bash scripts/download_transientangelo.sh [scene ...]
# The default is the four test scenes. Files already present are skipped.
set -e
cd "$(dirname "$0")/.."
BASE=https://huggingface.co/datasets/weihan1/transientangelo_dataset/resolve/main
ROOT=data/transientangelo
TEST_VIEWS="2 4 6 8 10 12 14 16 18 20"

# Training views of the two, three and five view splits of each scene.
declare -A TRAIN=(
    [baskets]="1 3 5 9 13 17"
    [boots]="1 5 7 9 11 13 17"
    [chef]="1 5 9 13 15 17 19"
    [food]="1 3 5 9 13 17 19"
    [cinema]="1 3 5 9 13 17"
    [carving]="1 3 5 9 13 17"
)

fetch() {
    local src=$1 dst=$2
    if [ -s "$dst" ]
    then
        return 0
    fi
    mkdir -p "$(dirname "$dst")"
    curl --fail --location --proto =https --output "$dst.part" "$BASE/$src"
    mv "$dst.part" "$dst"
}

fetch load/captured/pulse_low_flux.mat $ROOT/pulse_low_flux.mat
fetch load/captured/david_calibration_1005/intrinsics_safe.npz $ROOT/intrinsics_safe.npz

for scene in ${@:-baskets boots chef food}
do
    if [ -z "${TRAIN[$scene]}" ]
    then
        echo "unknown scene $scene"
        exit 1
    fi
    src=load/captured/$scene
    cams=final_cams
    for split in two three five
    do
        json=$cams/${split}_views/transforms_train.json
        fetch $src/$json $ROOT/$scene/$json
    done
    json=$cams/test_jsons/transforms_test.json
    fetch $src/$json $ROOT/$scene/$json
    for v in ${TRAIN[$scene]} $TEST_VIEWS
    do
        printf -v name 'transient%03d.pt' "$v"
        fetch $src/$name $ROOT/$scene/$name
    done
    for v in ${TRAIN[$scene]}
    do
        printf -v name 'train_%03d_mask.png' "$v"
        fetch $src/$name $ROOT/masks/$scene/$name
        printf -v name 'transient_%02d.pt' "$v"
        for level in 300 150 50 10
        do
            fetch low_photon_transients/captured/$scene/$level/$name $ROOT/$scene/$level/$name
        done
    done
    echo "$scene done"
done
