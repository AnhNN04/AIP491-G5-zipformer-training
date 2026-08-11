import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import finetune
import joblib
import numpy as np
import sentencepiece as spm
import torch
import tqdm
from asr_datamodule import FinetuneAsrDataModule
from einops import rearrange
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import str2bool
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch import einsum, nn
from torch.utils.data import DataLoader
from utils import get_avg_checkpoint

logger = logging.getLogger("dump_km_label")

def get_model(params, device):
    if params.checkpoint_type == "ASR":
        params.use_layer_norm = False

    if params.checkpoint_type == "pretrain":
        model = finetune.get_model(params)
    else:
        params.final_downsample = True  
        params.do_final_downsample = False  
        model = finetune.get_model(params)
        model.to(device)
        checkpoint = get_avg_checkpoint(
            params.pretrained_dir,
            params.epoch,
            params.avg,
            params.use_averaged_model,
            params.iter,
            device,
        )
        if params.checkpoint_type == "ASR":
            for item in list(checkpoint):
                if not item.startswith("encoder.") and not item.startswith(
                    "encoder_embed."
                ):
                    checkpoint.pop(item)
            checkpoint.pop("encoder.downsample_output.bias", None)
            missing_keys, unexpected_keys = model.encoder.load_state_dict(
                checkpoint, strict=False
            )
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
        logging.info(
            f"Init checkpoint, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
        )

    model.eval()
    model.to(device)
    return model

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--task-list", type=str)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--src-dir", type=str, nargs="*", help="for build list")

    parser.add_argument("--checkpoint-type", type=str, default="pretrain")

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
The pretrained model dir.
        It specifies the directory where the pretrained checkpoint is saved.""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    finetune.add_model_arguments(parser)

    return parser

class ApplyKmeans(object):
    def __init__(self, km_path, device):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    @torch.no_grad()
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1)
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

def extract_feature(batch, model):
    if model is None:
        return batch["features"]
    device = next(model.parameters()).device
    audio = batch["audio"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    encoder_out, encoder_out_lens = model.forward_encoder(
        audio, padding_mask, do_final_down_sample=False
    )
    b, l, d = encoder_out.shape
    holder = []
    for i in range(b):
        holder.append(encoder_out[i, : encoder_out_lens[i], :])
    encoder_out = torch.cat(holder, dim=0)
    return encoder_out, encoder_out_lens

def sub_routine(batch, model, km_model, km_dict, device):
    feat, len_lis = extract_feature(batch, model)
    kmeans = km_model(feat).to(torch.device("cpu"))
    offset = 0
    cut_ids = [cut.id for cut in batch["cuts"]]
    for cut_id, feat_len in zip(cut_ids, len_lis):
        label = [str(int(item)) for item in kmeans[offset : offset + feat_len]]
        km_dict[cut_id] = " ".join(label)
        offset += feat_len

def main(args):
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    args.blank_id = sp.piece_to_id("<blk>")
    args.vocab_size = sp.get_piece_size()

    args.feature_dim = 80

    logging.info(str(args))
    device = torch.device("cuda:0")
    model = ApplyKmeans(args.model_path, device)

    feature_model = get_model(args, device)

    task_file = args.task_list
    with open(task_file, "r") as f:
        task_lis = f.readlines()
    task_lis = [item.split() for item in task_lis]

    if args.start is not None:
        task_lis = task_lis[args.start : args.end]

    for src, tgt in tqdm.tqdm(task_lis):
        cuts = CutSet.from_file(src)
        km_dict = {}
        finetune_datamoddule = FinetuneAsrDataModule(args)
        test_dl = finetune_datamoddule.test_dataloaders(cuts)

        for i, batch in enumerate(tqdm.tqdm(test_dl, desc=f"Processing {os.path.basename(src)}")):
            sub_routine(batch, feature_model, model, km_dict, device)

        def add_label(km_dict):
            def f(cut):
                cut.custom = dict()
                cut.custom["kmeans"] = km_dict[cut.id]
                return cut

            return f

        cuts = cuts.map(add_label(km_dict))

        cuts.to_file(tgt)
        logger.info("finished successfully")

if __name__ == "__main__":
    parser = get_parser()
    FinetuneAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    main(args)
