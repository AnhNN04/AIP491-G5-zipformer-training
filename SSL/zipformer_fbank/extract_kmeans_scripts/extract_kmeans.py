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

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    import sys
    args = Namespace(
        model_path=None,
        task_list=None,
        suffix=None,
        start=None,
        end=None,
        src_dir=None,
        checkpoint_type="pretrain",
        epoch=30,
        iter=0,
        context_size=2,
        prune_range=5,
        pretrained_dir=None,
        bpe_model="../ASR/data/lang_bpe_2000/bpe.model",
        use_averaged_model=False,
        avg=1,

        # FinetuneAsrDataModule defaults
        manifest_dir=Path("data/wav"),
        max_duration=200.0,
        bucketing_sampler=True,
        num_buckets=30,
        shuffle=True,
        on_the_fly_feats=False,
        drop_last=True,
        num_workers=2,
        do_normalize=True,
        return_cuts=True,
        enable_spec_aug=True,
        spec_aug_time_warp_factor=80,
        enable_musan=True,
        input_strategy="PrecomputedFeatures",

        # 68M Parameter Zipformer Defaults
        num_encoder_layers="2,2,3,4,3,2",
        downsampling_factor="1,2,4,8,4,2",
        feedforward_dim="512,768,1024,1536,1024,768",
        num_heads="4,4,4,8,4,4",
        encoder_dim="192,256,384,512,384,256",
        query_head_dim=32,
        value_head_dim=12,
        pos_head_dim=4,
        pos_dim=48,
        encoder_unmasked_dim="192,192,256,256,256,192",
        cnn_module_kernel="31,31,15,15,15,31",
        decoder_dim=512,
        joiner_dim=512,
    )

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--model-path":
            args.model_path = sys.argv[i+1]
        elif sys.argv[i] == "--task-list":
            args.task_list = sys.argv[i+1]
        elif sys.argv[i] == "--suffix":
            args.suffix = sys.argv[i+1]
        elif sys.argv[i] == "--start":
            args.start = int(sys.argv[i+1])
        elif sys.argv[i] == "--end":
            args.end = int(sys.argv[i+1])
        elif sys.argv[i] == "--src-dir":
            args.src_dir = []
            j = i + 1
            while j < len(sys.argv) and not sys.argv[j].startswith("--"):
                args.src_dir.append(sys.argv[j])
                j += 1
        elif sys.argv[i] == "--checkpoint-type":
            args.checkpoint_type = sys.argv[i+1]
        elif sys.argv[i] == "--epoch":
            args.epoch = int(sys.argv[i+1])
        elif sys.argv[i] == "--iter":
            args.iter = int(sys.argv[i+1])
        elif sys.argv[i] == "--context-size":
            args.context_size = int(sys.argv[i+1])
        elif sys.argv[i] == "--prune-range":
            args.prune_range = int(sys.argv[i+1])
        elif sys.argv[i] == "--pretrained-dir":
            args.pretrained_dir = sys.argv[i+1]
        elif sys.argv[i] == "--bpe-model":
            args.bpe_model = sys.argv[i+1]
        elif sys.argv[i] == "--use-averaged-model":
            args.use_averaged_model = str2bool(sys.argv[i+1])
        elif sys.argv[i] == "--avg":
            args.avg = int(sys.argv[i+1])

    main(args)
