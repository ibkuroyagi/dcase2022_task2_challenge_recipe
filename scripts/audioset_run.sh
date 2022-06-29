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
n_jobs=256
conf=conf/tuning/asd_model.audioset_v000.yaml
resume=""
pos_machine=fan
# directory path setting
dumpdir=dump/use_audioset # directory to dump features
expdir=exp
# training related setting
tag=audioset_v000 # tag for directory to save model
target_mod=3
valid_ratio=0.1
audioset_dir=/path/to/AudioSet/audios
audioset_pow=21
# inference related setting
epochs="50 100 150 200 250 300"
checkpoints=""
use_10sec=false
feature=_embed # "": using all features in training an anomalous detector.
# _embed: using only embedding features in training an anomalous detector.
# _prediction: using only predictions in training an anomalous detector.
inlier_scp=""
use_target_in_embed=true

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
log "1. resume:${resume}"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

set -euo pipefail

log "Start run.sh"
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
    log "Download data."
    local/download_data.sh downloads
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Normalize wave data."
    for machine in "${machines[@]}"; do
        train_set="dev/${machine}/train eval/${machine}/train"
        valid_set="dev/${machine}/test"
        eval_set="eval/${machine}/test"
        statistic_path="${dumpdir}/dev/${machine}/train/statistic.json"
        for name in ${train_set} ${valid_set} ${eval_set}; do
            [ ! -e "${dumpdir}/${name}" ] && mkdir -p "${dumpdir}/${name}"
            log "See the progress via ${dumpdir}/${name}/normalize_wave.log."
            # shellcheck disable=SC2154
            ${train_cmd} --num_threads 1 "${dumpdir}/${name}/normalize_wave.log" \
                python -m asd_tools.bin.normalize_wave \
                --download_dir "downloads/${name}" \
                --dumpdir "${dumpdir}/${name}" \
                --statistic_path "${statistic_path}" \
                --verbose "${verbose}" \
                --no_normalize
            log "Successfully finished Normalize ${name}."
        done
    done
    log "Create Audioset scp."
    # shellcheck disable=SC1091
    local/get_audioset_scp.sh "${audioset_dir}" "${dumpdir}/audioset"
    split_scps=""
    for i in $(seq 1 "${n_jobs}"); do
        split_scps+=" ${dumpdir}/audioset/unbalanced_train_segments/log/wav.${i}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${dumpdir}/audioset/unbalanced_train_segments/wav.scp" ${split_scps}
    log "Creat dump file start. ${dumpdir}/audioset/unbalanced_train_segments/log/preprocess.*.log"
    pids=()
    (
        # shellcheck disable=SC2086,SC2154
        ${train_cmd} --max-jobs-run 64 JOB=1:"${n_jobs}" "${dumpdir}/audioset/unbalanced_train_segments/log/preprocess.JOB.log" \
            python -m asd_tools.bin.normalize_wave \
            --download_dir "${dumpdir}/audioset/unbalanced_train_segments/log/wav.JOB.scp" \
            --dumpdir "${dumpdir}/audioset/unbalanced_train_segments/part.JOB" \
            --verbose "${verbose}" \
            --no_normalize
    ) &
    pids+=($!)
    i=0
    for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1
    log "Successfully finished creat Audioset dump."
fi

end_str+="_${valid_ratio}"

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Write scp."
    log "Split training data."
    for machine in "${machines[@]}"; do
        train_set="dev/${machine}/train"
        valid_set="dev/${machine}/test"
        eval_set="eval/${machine}/test"
        # shellcheck disable=SC2154,SC2086
        ${train_cmd} "${dumpdir}/${train_set}/write_scp${end_str}.log" \
            python local/write_scp.py \
            --dumpdir ${dumpdir}/${train_set} ${dumpdir}/eval/${machine}/train \
            --valid_ratio "${valid_ratio}" \
            --target_mod "${target_mod}"
        log "Successfully splited ${dumpdir}/${train_set} train and valid data."
        log "All target data is in train${end_str}.scp"
        for name in ${valid_set} ${eval_set}; do
            # shellcheck disable=SC2154
            ${train_cmd} "${dumpdir}/${name}/write_scp.log" \
                python local/write_scp.py \
                --dumpdir "${dumpdir}/${name}" \
                --valid_ratio 0
            log "Successfully write ${dumpdir}/${name}/eval.scp."
        done
        cat "${dumpdir}/eval/${machine}/test/eval.scp" >>"${dumpdir}/dev/${machine}/test/eval.scp"
    done
    # shellcheck disable=SC2086
    ${train_cmd} "${dumpdir}/audioset/unbalanced_train_segments/log/write_scp.log" \
        python local/write_scp.py \
        --dumpdir "${dumpdir}/audioset/unbalanced_train_segments" \
        --max_audioset_size_2_pow 22
    log "Successfully write ${dumpdir}/audioset/audioset_2__${audioset_pow}.scp."
fi

tag+="${end_str}_p${audioset_pow}"
outdir="${expdir}/${pos_machine}/${tag}"
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Train embedding. resume: ${resume}"
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    train_neg_machine_scps=""
    valid_neg_machine_scps=""
    for neg_machine in "${neg_machines[@]}"; do
        train_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/train${end_str}.scp "
        valid_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/valid${end_str}.scp "
    done
    log "Training start. See the progress via ${outdir}/train_${pos_machine}_${tag}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/train_${pos_machine}_${tag}.log" \
        python -m asd_tools.bin.train \
        --pos_machine "${pos_machine}" \
        --train_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/train${end_str}.scp" \
        --train_neg_machine_scps ${train_neg_machine_scps} \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --valid_neg_machine_scps ${valid_neg_machine_scps} \
        --outlier_scp "${dumpdir}/audioset/unbalanced_train_segments/audioset_2__${audioset_pow}.scp" \
        --statistic_path "${dumpdir}/dev/${pos_machine}/train/statistic.json" \
        --outdir "${outdir}" \
        --config "${conf}" \
        --resume "${resume}" \
        --verbose "${verbose}"
    log "Successfully finished source training."
fi
# shellcheck disable=SC2086
if [ -z ${checkpoints} ]; then
    checkpoints+="${outdir}/best_loss/best_loss.pkl "
    for epoch in ${epochs}; do
        checkpoints+="${outdir}/checkpoint-${epoch}epochs/checkpoint-${epoch}epochs.pkl "
    done
fi
if [ -z "${inlier_scp}" ]; then
    inlier_scp="${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp"
    tail_name=""
else
    tail_name="_dev"
fi
if [ "${use_target_in_embed}" = "true" ]; then
    tail_name+="_target"
fi
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Embedding calculation start. See the progress via ${outdir}/embed_${pos_machine}_${tag}${tail_name}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/embed_${pos_machine}_${tag}${tail_name}.log" \
        python -m asd_tools.bin.embed \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --eval_pos_machine_scp "${dumpdir}/dev/${pos_machine}/test/eval.scp" \
        --statistic_path "${dumpdir}/dev/${pos_machine}/train/statistic.json" \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --use_10sec "${use_10sec}" \
        --tail_name "${tail_name}" \
        --verbose "${verbose}"
    log "Successfully finished extracting embedding."
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Stage 5: Inference start. See the progress via ${outdir}/infer_${pos_machine}_${feature}${tail_name}_${tag}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} "${outdir}/infer_${pos_machine}_${feature}${tail_name}_${tag}.log" \
        python -m asd_tools.bin.infer \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --feature "${feature}" \
        --use_10sec "${use_10sec}" \
        --tail_name "${tail_name}" \
        --verbose "${verbose}"
    log "Successfully finished Inference."
fi
