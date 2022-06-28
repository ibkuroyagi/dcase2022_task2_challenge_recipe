#!/bin/bash

stage=1
start_stage=0
no=domain_classifier
feature=_embed
use_10sec=false
tail_name=""
valid_ratio=0.1
use_target_in_embed=false
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
epochs="5 10 15 20 25 35 40 45 50"

set -euo pipefail

# machines=("fan" "gearbox" "bearing" "valve" "slider" "ToyCar" "ToyTrain")
machines=("bearing")
resume=""
tag=${no}
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        # resume=exp/${machine}/${no}/checkpoint-10epochs/checkpoint-10epochs.pkl
        echo "Start model training ${machine}/${no}. resume: ${resume}"
        # shellcheck disable=SC2086
        ./domain_classifier_run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --conf "conf/tuning/asd_model.${no}.yaml" \
            --pos_machine "${machine}" \
            --tag "${tag}" \
            --resume "${resume}" \
            --epochs "${epochs}" \
            --use_10sec "${use_10sec}" \
            --feature "${feature}" \
            --use_target_in_embed "${use_target_in_embed}" \
            --valid_ratio "${valid_ratio}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    if [ "${use_10sec}" = "true" ]; then
        feature="_10sec${feature}"
    fi
    tail_name=""
    if [ "${use_target_in_embed}" = "true" ]; then
        tail_name+="_target"
    fi
    # shellcheck disable=SC2086
    ./local/scoring.sh --no "${no}_${valid_ratio}" --epochs "${epochs}" --feature "${tail_name}${feature}"
fi
