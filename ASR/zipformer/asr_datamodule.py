import inspect
import logging
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import lhotse
import torch
from dataset import PseudoRecognitionDataset
from icefall.utils import str2bool
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)

class AsrDataModule:

    def __init__(self, args: Optional[Any] = None):
        self.manifest_dir = Path("data/fbank")
        self.max_duration = 300
        self.bucketing_sampler = True
        self.num_buckets = 30
        self.concatenate_cuts = False
        self.duration_factor = 1.0
        self.gap = 1.0
        self.on_the_fly_feats = False
        self.shuffle = True
        self.drop_last = True
        self.return_cuts = True
        self.num_workers = 2
        self.enable_spec_aug = True
        self.spec_aug_time_warp_factor = 80
        self.enable_musan = False
        self.input_strategy = "PrecomputedFeatures"

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        use_kmeans: bool = False,
    ) -> DataLoader:
        transforms = []
        if self.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest(self.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable")

        if self.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.duration_factor} and gap {self.gap}."
            )
            transforms = [
                CutConcatenate(
                    duration_factor=self.duration_factor, gap=self.gap
                )
            ] + transforms

        input_transforms = []
        if self.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.spec_aug_time_warp_factor}")
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        if use_kmeans:
            train = PseudoRecognitionDataset(
                input_strategy=eval(self.input_strategy)(),
                cut_transforms=transforms,
                input_transforms=input_transforms,
                return_cuts=self.return_cuts,
            )
        else:
            train = K2SpeechRecognitionDataset(
                input_strategy=eval(self.input_strategy)(),
                cut_transforms=transforms,
                input_transforms=input_transforms,
                return_cuts=self.return_cuts,
            )

        if self.on_the_fly_feats:
            if use_kmeans:
                train = PseudoRecognitionDataset(
                    cut_transforms=transforms,
                    input_strategy=OnTheFlyFeatures(
                        Fbank(FbankConfig(num_mel_bins=80))
                    ),
                    input_transforms=input_transforms,
                    return_cuts=self.return_cuts,
                )
            else:
                train = K2SpeechRecognitionDataset(
                    cut_transforms=transforms,
                    input_strategy=OnTheFlyFeatures(
                        Fbank(FbankConfig(num_mel_bins=80))
                    ),
                    input_transforms=input_transforms,
                    return_cuts=self.return_cuts,
                )

        if self.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.max_duration,
                shuffle=self.shuffle,
                num_buckets=self.num_buckets,
                buffer_size=self.num_buckets * 2000,
                shuffle_buffer_size=self.num_buckets * 5000,
                drop_last=self.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.max_duration,
                shuffle=self.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(
        self, cuts_valid: CutSet, use_kmeans: bool = False
    ) -> DataLoader:
        transforms = []
        if self.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.duration_factor, gap=self.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.on_the_fly_feats:
            if use_kmeans:
                validate = PseudoRecognitionDataset(
                    cut_transforms=transforms,
                    input_strategy=OnTheFlyFeatures(
                        Fbank(FbankConfig(num_mel_bins=80))
                    ),
                    return_cuts=self.return_cuts,
                )
            else:
                validate = K2SpeechRecognitionDataset(
                    cut_transforms=transforms,
                    input_strategy=OnTheFlyFeatures(
                        Fbank(FbankConfig(num_mel_bins=80))
                    ),
                    return_cuts=self.return_cuts,
                )
        else:
            if use_kmeans:
                validate = PseudoRecognitionDataset(
                    cut_transforms=transforms,
                    return_cuts=self.return_cuts,
                )
            else:
                validate = K2SpeechRecognitionDataset(
                    cut_transforms=transforms,
                    return_cuts=self.return_cuts,
                )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        return load_manifest_lazy(
            self.manifest_dir / "capstone_cuts_train.jsonl.gz"
        )

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        return load_manifest_lazy(self.manifest_dir / "capstone_cuts_dev.jsonl.gz")

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        return load_manifest_lazy(self.manifest_dir / "capstone_cuts_test.jsonl.gz")
