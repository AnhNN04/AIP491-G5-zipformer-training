
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import finetune
import joblib
import lhotse
import numpy as np
import sentencepiece as spm
import torch
from asr_datamodule import FinetuneAsrDataModule
from asr_datamodule import FinetuneAsrDataModule
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
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from utils import get_avg_checkpoint

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")

class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)

def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

def get_cuts(cut_files, src_dir):
    if cut_files is not None:
        cuts = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_files)
    else:
        cut_files = os.listdir(src_dir)
        cut_files = [
            os.path.join(src_dir, item)
            for item in cut_files
            if item.endswith(".jsonl.gz") and item.find("_raw") <= 0
        ]
        sorted_filenames = sorted(cut_files)
        cuts = lhotse.combine(lhotse.load_manifest_lazy(p) for p in sorted_filenames)
    return cuts

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
    encoder_out = torch.cat(holder, dim=0).to(torch.device("cpu")).detach().numpy()
    return encoder_out



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

def learn_kmeans(
    args,
    do_training,
    files,
    src_dir,
    km_path,
    n_clusters,
    seed,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    if do_training:
        km_model = get_km_model(
            n_clusters,
            init,
            max_iter,
            batch_size,
            tol,
            max_no_improvement,
            n_init,
            reassignment_ratio,
        )
    else:
        km_model = joblib.load(km_path)
    cuts = get_cuts(files, src_dir)
    finetune_datamoddule = FinetuneAsrDataModule(args)
    train_dl = finetune_datamoddule.test_dataloaders(cuts)

    fix_random_seed(args.seed)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    logging.info(f"Device: {device}")

    part_feats_holder = []
    model = get_model(args, device)

    for batch in train_dl:
        feats = extract_feature(batch, model)
        if args.frame_stride > 1:
            feats = feats[::args.frame_stride]
        part_feats_holder.append(feats)

    part_feats = np.concatenate(part_feats_holder, axis=0)
    logging.info(f"data size: {part_feats.shape}")
    if do_training:
        km_model.fit(part_feats)
        joblib.dump(km_model, km_path)
    inertia = -km_model.score(part_feats) / len(part_feats)
    logging.info(f"Total inertia: {inertia:.5f}")

    logger.info("finished successfully")

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    args = Namespace(
        km_path="data/kmeans.pt",
        n_clusters=500,
        files=None,
        do_training=False,
        src_dir=None,
        frame_stride=1,
        init="k-means++",
        max_iter=100,
        batch_size=10000,
        tol=0.0,
        max_no_improvement=100,
        n_init=20,
        reassignment_ratio=0.0,
        seed=42,
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
        if sys.argv[i] == "--km-path":
            args.km_path = sys.argv[i+1]
        elif sys.argv[i] == "--n-clusters":
            args.n_clusters = int(sys.argv[i+1])
        elif sys.argv[i] == "--files":
            args.files = []
            j = i + 1
            while j < len(sys.argv) and not sys.argv[j].startswith("--"):
                args.files.append(sys.argv[j])
                j += 1
        elif sys.argv[i] == "--do-training":
            args.do_training = True
        elif sys.argv[i] == "--src-dir":
            args.src_dir = sys.argv[i+1]
        elif sys.argv[i] == "--frame-stride":
            args.frame_stride = int(sys.argv[i+1])
        elif sys.argv[i] == "--max-iter":
            args.max_iter = int(sys.argv[i+1])
        elif sys.argv[i] == "--batch-size":
            args.batch_size = int(sys.argv[i+1])
        elif sys.argv[i] == "--tol":
            args.tol = float(sys.argv[i+1])
        elif sys.argv[i] == "--max-no-improvement":
            args.max_no_improvement = int(sys.argv[i+1])
        elif sys.argv[i] == "--n-init":
            args.n_init = int(sys.argv[i+1])
        elif sys.argv[i] == "--reassignment-ratio":
            args.reassignment_ratio = float(sys.argv[i+1])
        elif sys.argv[i] == "--seed":
            args.seed = int(sys.argv[i+1])
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

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    args.blank_id = sp.piece_to_id("<blk>")
    args.vocab_size = sp.get_piece_size()

    args.feature_dim = 80
    logging.info(str(args))

    learn_kmeans(
        args,
        do_training=args.do_training,
        files=args.files,
        src_dir=args.src_dir,
        km_path=args.km_path,
        n_clusters=args.n_clusters,
        seed=args.seed,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        tol=args.tol,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,
        max_no_improvement=args.max_no_improvement,
    )
