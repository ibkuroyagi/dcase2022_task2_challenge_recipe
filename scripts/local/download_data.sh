#!/bin/bash

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# check arguments
if [ $# != 1 ]; then
  echo "Usage: $0 <download_dir>"
  echo "e.g. $0 downloads"
  exit 1
fi

download_dir=$1

set -euo pipefail

# check machine type
available_machine_types=(
  "bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve"
)

for machine in "${available_machine_types[@]}"; do
  if [ ! -e "${download_dir}/zip/dev_${machine}.zip" ]; then
    mkdir -p "${download_dir}/zip"
    cd "${download_dir}/zip"
    wget "https://zenodo.org/record/6355122/files/dev_${machine}.zip"
    cd ../..
    log "Successfully downloaded dataset for development."
  else
    log "Development dataset was already downloaded. Skipped."
  fi

  if [ ! -d "${download_dir}/dev/${machine}" ]; then
    mkdir -p "${download_dir}/dev"
    unzip "${download_dir}/zip/dev_${machine}.zip" \
      -d "${download_dir}/dev"
    log "Successfully unzipped dataset for development."
  else
    log "Development dataset was already unzipped. Skipped."
  fi

  if [ ! -e "${download_dir}/zip/eval_data_${machine}_train.zip" ]; then
    cd "${download_dir}/zip"
    wget "https://zenodo.org/record/6462969/files/eval_data_${machine}_train.zip"
    cd ../..
    log "Successfully downloaded additional training dataset."
  else
    log "Additional training dataset was already downloaded. Skipped."
  fi

  if [ ! -d "${download_dir}/eval/${machine}/train" ]; then
    mkdir -p "${download_dir}/eval"
    unzip "${download_dir}/zip/eval_data_${machine}_train.zip" \
      -d "${download_dir}/eval"
    log "Successfully unzipped additional training dataset."
  else
    log "Additional training dataset was already unzipped. Skipped."
  fi

  if [ ! -e "${download_dir}/zip/eval_data_${machine}_test.zip" ]; then
    cd "${download_dir}/zip"
    wget "https://zenodo.org/record/6586456/files/eval_data_${machine}_test.zip"
    cd ../..
    log "Successfully downloaded evaluation dataset."
  else
    log "Evaluation dataset was already downloaded. Skipped."
  fi

  if [ ! -d "${download_dir}/eval/${machine}/test" ]; then
    mkdir -p "${download_dir}/eval"
    unzip "${download_dir}/zip/eval_data_${machine}_test.zip" \
      -d "${download_dir}/eval"
    log "Successfully unzipped evaluation dataset."
  else
    log "Evaluation dataset was already unzipped. Skipped."
  fi
done
