export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=8
stage=0
stop_stage=3

dl_dir=/dataset/100h-labeled-data

. shared/parse_options.sh || exit 1

vocab_sizes=(
  # 5000
  2000
  # 1000
  # 500
)

mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Running prepare.sh"

log "dl_dir: $dl_dir"


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare supervised manifest"
  mkdir -p data/manifests
  if [ ! -e data/manifests/.supervised.done ]; then
    python local/prepare_manifest.py --num-jobs $nj --corpus-dir $dl_dir --output-dir data/manifests --language vietnamese
    touch data/manifests/.supervised.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.supervised.done ]; then
    ./local/compute_fbank.py
    touch data/fbank/.supervised.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}  # change prefix file-name if need
    mkdir -p $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate data for BPE training"
      find -L "$dl_dir/transcripts" -name "*.txt" -exec awk '1' {} + > $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt
    fi
  done
fi
