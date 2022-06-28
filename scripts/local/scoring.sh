#!/bin/bash

. ./cmd.sh || exit 1
. ./path.sh || exit 1

no=000
epochs="50 100 150 200 250 300"
feature=""
. utils/parse_options.sh || exit 1
set -euo pipefail

echo "Start scoring arregated anomaly scores."
agg_checkpoints=""
for machine in bearing fan gearbox valve slider ToyCar ToyTrain; do
    agg_checkpoints+="exp/${machine}/${no}/best_loss/best_loss${feature}_agg.csv "
done
mkdir -p "exp/all/${no}/best_loss"
echo "See log via  exp/all/${no}/best_loss/scoring${feature}.log"
# shellcheck disable=SC2154,SC2086
${train_cmd} "exp/all/${no}/best_loss/scoring${feature}.log" \
    python -m asd_tools.bin.scoring --feature "${feature}" --agg_checkpoints ${agg_checkpoints}
score_checkpoints="exp/all/${no}/best_loss/score${feature}.csv "
score_checkpoints=""
for epoch in ${epochs}; do
    echo "Start scoring arregated anomaly scores in ${epoch} epoch."
    agg_checkpoints=""
    for machine in bearing fan gearbox valve slider ToyCar ToyTrain; do
        agg_checkpoints+="exp/${machine}/${no}/checkpoint-${epoch}epochs/checkpoint-${epoch}epochs${feature}_agg.csv "
    done
    mkdir -p "exp/all/${no}/checkpoint-${epoch}epochs"
    echo "See log via  exp/all/${no}/checkpoint-${epoch}epochs/scoring${feature}.log"
    # shellcheck disable=SC2154,SC2086
    ${train_cmd} "exp/all/${no}/checkpoint-${epoch}epochs/scoring${feature}.log" \
        python -m asd_tools.bin.scoring --feature "${feature}" --agg_checkpoints ${agg_checkpoints}
    score_checkpoints+="exp/all/${no}/checkpoint-${epoch}epochs/score${feature}.csv "
done
# shellcheck disable=SC2086
python -m asd_tools.bin.scoring --feature "${feature}" --agg_checkpoints ${score_checkpoints} --concat
echo "Successfully finished scoring${feature}."
