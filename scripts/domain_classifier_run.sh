#!/bin/bash

# Copyright 2022 Ibuki Kuroyanagi
# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1
# basic settings
stage=1      # stage to start
stop_stage=1 # stage to stop
verbose=1    # verbosity level (lower is less info)
n_gpus=1     # number of gpus in training

conf=conf/tuning/asd_model.domain_classifier.yaml
pos_machine=fan
# directory path setting
dumpdir=dump/base # directory to dump wave
expdir=exp
# training related setting
time_stretch_rates=1.0 #"1.0" or "1.0 1.1 0.9"
tag=domain_classifier  # tag for directory to save model
resume=""
valid_ratio=0.1
# inference related setting
epochs="50 100 150 200 250 300"
checkpoints=""
use_10sec=false
feature=_embed # "": using all features in training an anomalous detector.
# _embed: using only embedding features in training an anomalous detector.
# _prediction: using only predictions in training an anomalous detector.
use_target_in_embed=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

set -euo pipefail
echo "Start run.sh"
machines=("bearing" "fan" "gearbox" "valve" "slider" "ToyTrain" "ToyCar")
if [ "${pos_machine}" = "bearing" ]; then
    neg_machines=("fan" "gearbox" "valve" "slider" "ToyTrain" "ToyCar")
elif [ "${pos_machine}" = "fan" ]; then
    neg_machines=("gearbox" "bearing" "valve" "slider" "ToyTrain" "ToyCar")
elif [ "${pos_machine}" = "gearbox" ]; then
    neg_machines=("fan" "bearing" "valve" "slider" "ToyTrain" "ToyCar")
elif [ "${pos_machine}" = "valve" ]; then
    neg_machines=("fan" "gearbox" "bearing" "slider" "ToyTrain" "ToyCar")
elif [ "${pos_machine}" = "slider" ]; then
    neg_machines=("fan" "gearbox" "valve" "bearing" "ToyTrain" "ToyCar")
elif [ "${pos_machine}" = "ToyCar" ]; then
    neg_machines=("fan" "gearbox" "valve" "bearing" "slider" "ToyTrain")
elif [ "${pos_machine}" = "ToyTrain" ]; then
    neg_machines=("fan" "gearbox" "valve" "bearing" "slider" "ToyCar")
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Download data."
    local/download_data.sh downloads
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Normalize wave data."
    for machine in "${machines[@]}"; do
        train_set="dev/${machine}/train eval/${machine}/train"
        valid_set="dev/${machine}/test"
        eval_set="eval/${machine}/test"
        statistic_path="${dumpdir}/dev/${machine}/train/statistic.json"
        for name in ${train_set} ${valid_set} ${eval_set}; do
            [ ! -e "${dumpdir}/${name}" ] && mkdir -p "${dumpdir}/${name}"
            if [ "${time_stretch_rates}" = "1.0" ]; then
                echo "See the progress via ${dumpdir}/${name}/normalize_wave${time_stretch_rates}.log."
                # shellcheck disable=SC2154,SC2086
                ${train_cmd} --num_threads 1 "${dumpdir}/${name}/normalize_wave${time_stretch_rates}.log" \
                    python -m asd_tools.bin.normalize_wave \
                    --download_dir "downloads/${name}" \
                    --dumpdir "${dumpdir}/${name}" \
                    --statistic_path "${statistic_path}" \
                    --time_stretch_rate ${time_stretch_rates} \
                    --verbose "${verbose}"
                echo "Successfully finished Normalize ${name} ${time_stretch_rate}."
            else
                for time_stretch_rate in ${time_stretch_rates}; do
                    echo "See the progress via ${dumpdir}/${name}/normalize_wave${time_stretch_rate}.log."
                    # shellcheck disable=SC2154,SC2086
                    ${train_cmd} --num_threads 1 "${dumpdir}/${name}/normalize_wave${time_stretch_rate}.log" \
                        python -m asd_tools.bin.normalize_wave \
                        --download_dir "downloads/${name}" \
                        --dumpdir "${dumpdir}/${name}" \
                        --statistic_path "${statistic_path}" \
                        --time_stretch_rate ${time_stretch_rate} \
                        --verbose "${verbose}"
                    echo "Successfully finished Normalize ${name} ${time_stretch_rate}."
                done
            fi
        done
    done
fi
end_str="_${valid_ratio}"

if [ "${time_stretch_rates}" != "1.0" ]; then
    end_str+="_sp"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Write scp."
    echo "Split training data."
    for machine in "${machines[@]}"; do
        train_set="dev/${machine}/train"
        valid_set="dev/${machine}/test"
        eval_set="eval/${machine}/test"
        # shellcheck disable=SC2154,SC2086
        ${train_cmd} "${dumpdir}/${train_set}/write_scp${end_str}.log" \
            python local/write_scp.py \
            --dumpdir ${dumpdir}/${train_set} ${dumpdir}/eval/${machine}/train \
            --valid_ratio "${valid_ratio}"
        echo "Successfully splited ${dumpdir}/${train_set} train and valid data."
        for name in ${valid_set} ${eval_set}; do
            # shellcheck disable=SC2154,SC2086
            ${train_cmd} "${dumpdir}/${name}/write_scp${end_str}.log" \
                python local/write_scp.py \
                --dumpdir "${dumpdir}/${name}" \
                --valid_ratio 0
            echo "Successfully write ${dumpdir}/${name}/eval.scp."
        done
        cat "${dumpdir}/${eval_set}/eval.scp" >>"${dumpdir}/${valid_set}/eval.scp"
    done
fi
tag+="${end_str}"
outdir="${expdir}/${pos_machine}/${tag}"
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Train a feature extractor and domain classifier."

    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    train_neg_machine_scps=""
    valid_neg_machine_scps=""
    for neg_machine in "${neg_machines[@]}"; do
        train_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/train${end_str}.scp "
        valid_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/valid${end_str}.scp "
    done
    echo "Training start. See the progress via ${outdir}/train_${pos_machine}_${tag}.log. resume: ${resume}"
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/train_${pos_machine}_${tag}.log" \
        python -m asd_tools.bin.domain_classifier_train \
        --pos_machine "${pos_machine}" \
        --train_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/train${end_str}.scp" \
        --train_neg_machine_scps ${train_neg_machine_scps} \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --valid_neg_machine_scps ${valid_neg_machine_scps} \
        --outdir "${outdir}" \
        --config "${conf}" \
        --resume "${resume}" \
        --verbose "${verbose}"
    echo "Successfully finished source training."
fi
# shellcheck disable=SC2086
if [ -z ${checkpoints} ]; then
    checkpoints+="${outdir}/best_loss/best_loss.pkl "
    for epoch in ${epochs}; do
        checkpoints+="${outdir}/checkpoint-${epoch}epochs/checkpoint-${epoch}epochs.pkl "
    done
fi

tail_name=""
if [ "${use_target_in_embed}" = "true" ]; then
    tail_name+="_target"
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Embedding calculation start. See the progress via ${outdir}/embed_${pos_machine}${tail_name}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/embed_${pos_machine}${tail_name}.log" \
        python -m asd_tools.bin.embed \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --eval_pos_machine_scp "${dumpdir}/dev/${pos_machine}/test/eval.scp" \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --use_10sec "${use_10sec}" \
        --tail_name "${tail_name}" \
        --verbose "${verbose}"
    echo "Successfully finished extracting embedding."
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Inference start. See the progress via ${outdir}/infer_${pos_machine}${feature}${tail_name}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} "${outdir}/infer_${pos_machine}${feature}${tail_name}.log" \
        python -m asd_tools.bin.infer \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --feature "${feature}" \
        --use_10sec "${use_10sec}" \
        --tail_name "${tail_name}" \
        --verbose "${verbose}"
    echo "Successfully finished Inference."
fi
