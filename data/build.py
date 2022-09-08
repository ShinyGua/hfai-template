import os

import numpy as np
import torch
import torch.distributed as dist
from timm.data import create_transform, Mixup
from torchvision import datasets

from .samplers import SubsetRandomSampler


def build_loader(config):
    dsets = {
        "train": {},
        "val": {},
    }

    dset_loaders = {
        "train": {},
        "val": {},
    }

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    traindir = os.path.join(config.data.data_path, 'train')
    valdir = os.path.join(config.data.data_path, 'val')

    dsets['train'] = build_dataset(config=config, data_path=traindir, is_train=True)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build train dataset")
    dsets['val'] = build_dataset(config=config, data_path=valdir, is_train=False)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build val dataset")

    sampler_train = torch.utils.data.DistributedSampler(
        dsets['train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices = np.arange(dist.get_rank(), len(dsets['val']), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    dset_loaders['train'] = torch.utils.data.DataLoader(
        dsets['train'],
        sampler=sampler_train,
        batch_size=config.model.batch_size,
        num_workers=config.workers,
        pin_memory=config.pin_mem,
        drop_last=True,
        shuffle=False,
    )

    dset_loaders['val'] = torch.utils.data.DataLoader(
        dsets['val'],
        sampler=sampler_val,
        batch_size=config.model.batch_size,
        num_workers=config.workers,
        pin_memory=config.pin_mem,
        drop_last=False,
        shuffle = False,
    )

    mixup_fn = None
    mixup_active = config.aug.mixup > 0 or config.aug.cutmix > 0. or config.aug.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.aug.mixup, cutmix_alpha=config.aug.cutmix, cutmix_minmax=config.aug.cutmix_minmax,
            prob=config.aug.mixup_prob, switch_prob=config.aug.mixup_switch_prob, mode=config.aug.mixup_mode,
            label_smoothing=config.aug.smoothing, num_classes=config.aug.num_classes)

    return dsets, dset_loaders, mixup_fn


def build_dataset(config, data_path, is_train=True):
    transform = build_transform(config, is_train)
    dataset = datasets.ImageFolder(data_path, transform)
    return dataset


def build_transform(config, is_train=True):
    if is_train:
        transforms = create_transform(
            input_size=config.model.img_size,
            is_training=True,
            no_aug=config.aug.no_aug,
            scale=config.aug.scale,
            ratio=config.aug.ratio,
            hflip=config.aug.hflip,
            vflip=config.aug.vflip,
            color_jitter=config.aug.color_jitter,
            auto_augment=config.aug.aa,
            interpolation=config.aug.interpolation,
            mean=config.model.mean,
            std=config.model.std,
            re_prob=config.aug.reprob,
            re_mode=config.aug.remode,
            re_count=config.aug.recount,
            re_num_splits=config.aug.aug_splits,
        )
    else:
        transforms = create_transform(
            input_size=config.model.img_size,
            interpolation=config.aug.interpolation,
            mean=config.model.mean,
            std=config.model.std,
            crop_pct=config.model.crop_pct
        )
    return transforms
