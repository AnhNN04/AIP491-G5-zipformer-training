#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0

pip install lilcom
mkdir -p data
ln -sfn ../../ASR/data/fbank data/fbank

python zipformer_fbank/finetune.py \
    --world-size 1 \
    --num-epochs 20 \
    --start-epoch 1 \
    --use-fp16 1 \
    --sample-rate 100 \
    --manifest-dir ../ASR/data/fbank \
    --bpe-model ../ASR/data/lang_bpe_2000/bpe.model \
    --exp-dir zipformer_fbank/exp \
    --max-duration 300 \
    --enable-musan 0 \
    --enable-spec-aug 0 \
    --mask-before-cnn 1 \
    --mask-prob 0.65 \
    --mask-channel-prob 0.5 \
    --mask-channel-length 20 \
    --accum-grad 1 \
    --seed 1556 \
    --base-lr 0.002 \
    --max-lr-update 80000 \
    --phase-ratio "(0.1, 0.4, 0.5)" \
    --pretrained-checkpoint-path zipformer_fbank/exp/epoch-12.pt \
    --pretrained-checkpoint-type SSL \
    --init-encoder-only 1 \
    --use-layer-norm 0 \
    --final-downsample 1 \
    --causal 0 \
    --master-port 12356

