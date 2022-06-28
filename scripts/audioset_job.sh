#!/bin/bash

stage=1
start_stage=0
no=audioset_v000
feature=_embed
use_10sec=false
audioset_pow=21
valid_ratio=0.1
use_target_in_embed=false
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

epochs="50 100 150 200 250 300"

set -euo pipefail

# machines=("fan" "gearbox" "bearing" "valve" "slider" "ToyCar" "ToyTrain")
machines=("bearing")
resume=""
tag=${no}
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        echo "Start model training ${machine}/${no}. resume:${resume}"
        ./audioset_run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --conf "conf/tuning/asd_model.${no}.yaml" \
            --pos_machine "${machine}" \
            --resume "${resume}" \
            --tag "${tag}" \
            --feature "${feature}" \
            --use_10sec "${use_10sec}" \
            --epochs "${epochs}" \
            --valid_ratio "${valid_ratio}" \
            --use_target_in_embed "${use_target_in_embed}" \
            --audioset_pow "${audioset_pow}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    if [ "${use_10sec}" = "true" ]; then
        feature="_10sec${feature}"
    fi
    if [ "${use_target_in_embed}" = "true" ]; then
        feature="_target${feature}"
    fi
    ./local/scoring.sh \
        --no "${no}_${valid_ratio}_p${audioset_pow}" \
        --feature "${feature}" \
        --epochs "${epochs}"
fi
