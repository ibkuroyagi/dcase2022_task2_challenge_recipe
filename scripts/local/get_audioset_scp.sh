#!/bin/bash

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <dump_audio_dir>"
    echo "e.g. $0 /path/to/AudioSet/audios dump/audioset"
    exit 1
fi

audioset_dir=$1
dump_audioset_dir=$2
[ ! -e "${dump_audioset_dir}/unbalanced_train_segments/log" ] && mkdir -p "${dump_audioset_dir}/unbalanced_train_segments/log"
[ ! -e "${dump_audioset_dir}/balanced_train_segments/log" ] && mkdir -p "${dump_audioset_dir}/balanced_train_segments/log"
echo -n >${dump_audioset_dir}/balanced_train_segments/wav.scp
ls ${audioset_dir}/balanced_train_segments/*.wav >>${dump_audioset_dir}/balanced_train_segments/wav.scp
log "Created ${dump_audioset_dir}/balanced_train_segments/wav.scp"

echo -n >${dump_audioset_dir}/unbalanced_train_segments/wav.scp
for i in {00..40}; do
    for pathfile in ${audioset_dir}/unbalanced_train_segments/unbalanced_train_segments_part${i}/*.wav; do
        echo $pathfile >>${dump_audioset_dir}/unbalanced_train_segments/wav.scp
    done
done
log "Created ${dump_audioset_dir}/unbalanced_train_segments/wav.scp"
