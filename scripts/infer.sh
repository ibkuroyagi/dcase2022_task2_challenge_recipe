#!/bin/bash
# 030 -> 005, 040 -> 0.15, 050 -> 0.2, 060 -> 0.25, 0.70 -> 0.3, 008 -> 0.1

feature=""
# for use_target_in_embed in "true" "false"; do
#     for no in {000..007} {016..024} {080..083} {100..109} {200..203}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 5 --valid_ratio 0.1 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}"
#         sleep 60
#     done
#     for no in {030..033}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 5 --valid_ratio 0.05 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}"
#         sleep 60
#     done
#     for no in {040..043}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 5 --valid_ratio 0.15 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}"
#         sleep 60
#     done
#     for no in {050..053}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 5 --valid_ratio 0.2 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}"
#         sleep 60
#     done
#     for no in {060..063}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 5 --valid_ratio 0.25 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}"
#         sleep 60
#     done
#     for no in {070..073}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 5 --valid_ratio 0.3 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}"
#         sleep 60
#     done
# done
# for use_target_in_embed in "true" "false"; do
#     for no in {008..015}; do
#         ./job.sh --no "${no}" --stage 1 --start_stage 5 --valid_ratio 0.1 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}" --epochs "20 40 60 80 100"
#         sleep 60
#     done
# done

# sleep 2600
for feature in "" "_embed"; do
    for use_target_in_embed in "true" "false"; do
        for no in {000..007} {016..024} {080..083} {100..109} {200..203} {030..033} {040..043} {050..053} {060..063} {070..073}; do
            sbatch ./job.sh --no "${no}" --stage 2 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}"
        done
        sleep 60
    done
    for use_target_in_embed in "true" "false"; do
        for no in {008..015}; do
            sbatch ./job.sh --no "${no}" --stage 2 --valid_ratio 0.1 --use_target_in_embed "${use_target_in_embed}" --feature "${feature}" --epochs "20 40 60 80 100"
        done
        sleep 60
    done
done
