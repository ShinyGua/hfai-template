import pickle

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from timm.data import create_transform, Mixup
from torch.utils.data.distributed import DistributedSampler
from .samplers import SubsetRandomSampler


def build_loader(config):
    dsets = {}
    dset_loaders = {}

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # traindir = config.data.train_data_path
    # valdir = config.data.val_data_path

    # set the dataset root path
    data_root_path = config.data.data_root_path

    dsets['train'] = build_dataset(config, is_train=True)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build train dataset")
    dsets['val'] = build_dataset(config, is_train=False)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build val dataset")

    sampler_train = DistributedSampler(
        dsets['train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices = np.arange(dist.get_rank(), len(dsets['val']), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    dset_loaders['train'] = DataLoader(
        dataset=dsets['train'],
        sampler=sampler_train,
        batch_size=config.model.batch_size,
        num_workers=config.workers,
        pin_memory=config.pin_mem,
        drop_last=True,
        shuffle=False,
    )

    dset_loaders['val'] = DataLoader(
        dataset=dsets['val'],
        sampler=sampler_val,
        batch_size=config.model.batch_size,
        num_workers=config.workers,
        pin_memory=config.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    mixup_fn = None
    mixup_active = config.aug.mixup > 0 or config.aug.cutmix > 0. or config.aug.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.aug.mixup, cutmix_alpha=config.aug.cutmix, cutmix_minmax=config.aug.cutmix_minmax,
            prob=config.aug.mixup_prob, switch_prob=config.aug.mixup_switch_prob, mode=config.aug.mixup_mode,
            label_smoothing=config.aug.smoothing, num_classes=config.aug.num_classes)

    return dsets, dset_loaders, mixup_fn


def build_dataset(config, is_train=True):
    transform = build_transform(config, is_train)
    dataset = None
    if config.data.dataset_name == "CIFAR10":
        from hfai.datasets import CIFAR10, set_data_dir
        set_data_dir(config.data.data_root_path)
        dataset = CIFAR10(split="train", transform=transform)
    else:
        assert f"It not supports the dataset {config.data.dataset_name}!"
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
