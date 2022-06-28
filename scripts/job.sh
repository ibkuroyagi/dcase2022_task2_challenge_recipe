#!/bin/bash

stage=1
start_stage=0
no=000
feature=_embed
use_10sec=false
valid_ratio=0.15
use_norm=""   #_norm: Normalize feature in training anomalous detector at stage 5.
inlier_scp="" #"": using only validation data or "_dev": using all training data
use_target_in_embed=true
epochs="50 100 150 200 250 300"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

set -euo pipefail

# machines=("fan" "gearbox" "bearing" "valve" "slider" "ToyCar" "ToyTrain")
machines=("bearing")
resume=""
tag=${no}
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        # resume=exp/${machine}/${no}/best_loss/best_loss.pkl
        if [ -n "${inlier_scp}" ]; then
            inlier_scp="dump/dev/${machine}/train/dev.scp"
        fi
        echo "Start model training ${machine}/${no}. resume:${resume}, inlier_scp:${inlier_scp}"
        ./run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --conf "conf/tuning/asd_model.${no}.yaml" \
            --pos_machine "${machine}" \
            --resume "${resume}" \
            --tag "${tag}" \
            --feature "${feature}" \
            --use_10sec "${use_10sec}" \
            --epochs "${epochs}" \
            --inlier_scp "${inlier_scp}" \
            --use_norm "${use_norm}" \
            --valid_ratio "${valid_ratio}" \
            --use_target_in_embed "${use_target_in_embed}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    if [ "${use_10sec}" = "true" ]; then
        feature="_10sec${feature}"
    fi
    if [ "${use_target_in_embed}" = "true" ]; then
        feature="_target${feature}"
    fi
    feature+=${use_norm}
    ./local/scoring.sh --no "${no}_${valid_ratio}" --feature "${feature}" --epochs "${epochs}"
fi
