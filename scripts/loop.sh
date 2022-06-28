#!/bin/bash
# 030 -> 005, 040 -> 0.15, 050 -> 0.2, 060 -> 0.25, 0.70 -> 0.3, 008 -> 0.1

# feature=""

# for use_target_in_embed in "true" "false"; do
#     for no in {008..015}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 4 --valid_ratio 0.1 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}" --epochs "20 40 60 80 100"
#         sleep 90
#     done
# done
# sleep 3600
# for use_target_in_embed in "true" "false"; do
#     for no in {008..015}; do
#         sbatch ./job.sh --no "${no}" --stage 2 --valid_ratio 0.1 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}" --epochs "20 40 60 80 100"
#     done
# done
# sbatch ./job.sh --stage 2 --no 000 --valid_ratio 0.1
# sbatch ./job.sh --stage 2 --no 001 --valid_ratio 0.2
# sbatch ./job.sh --stage 2 --no 103 --valid_ratio 0.15
# sbatch ./job.sh --stage 2 --no 202 --valid_ratio 0.1
machines=("fan" "gearbox" "bearing" "valve" "slider" "ToyCar" "ToyTrain")
for machine in "${machines[@]}"; do
    rm -rf exp/${machine}/domain_classifier_0.1
done
