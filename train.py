from pathlib import Path

import hfai_env

hfai_env.set_env('diff_hfai')

import hfai
import argparse
import datetime
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import BinaryCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter, distribute_bn, accuracy
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import Process
from hfai.distributed import get_rank
from hfai.client import receive_suspend_command, go_suspend

from configs.config import get_config
from data import build_loader
from model import build_model
from utils import optimizer_kwargs, create_scheduler, get_grad_norm, reduce_tensor, load_checkpoint, save_checkpoint
from utils.logger import create_logger

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
except ImportError:
    amp = None

hfai.client.bind_hf_except_hook(Process)


def parse_option():
    parser = argparse.ArgumentParser('pytorch template training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default="configs/CIFAR10_Test.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-root-path', type=str, help='path to dataset root')
    parser.add_argument('--train-data-path', type=str, help='path to dataset')
    parser.add_argument('--val-data-path', type=str, help='path to dataset')
    parser.add_argument('--dataset-name', type=str, help='Dataset type')
    parser.add_argument('--num-classes', type=int, help='Number of label classes')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', default="hfai_test", help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(local_rank):
    # 超参数导入
    # load hyperparameter
    _, config = parse_option()

    # 检查是否安装amp
    # check whether user is installed the amp or not
    if config.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"
    Path(config.output).mkdir(parents=True, exist_ok=True)

    # Multi-node communication
    # 多机通信
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # number of node 机器个数
    rank = int(os.environ['RANK'])  # rank of current node 当前机器编号
    gpus = torch.cuda.device_count()  # Number of GPUs per node 每台机器的GPU个数

    # world_size is the number of global GPU, rank is the global index of current GPU
    # world_size是全局GPU个数，rank是当前GPU全局编号
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    # fix the random seed
    # 设置随机种子
    seed = config.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # create the logger
    # 建立logger
    logger = create_logger(output_dir=config.output, dist_rank=local_rank, name=f"{config.model.name}")

    if dist.get_rank() == 0:
        path = Path(config.output).joinpath("config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(config)
        logger.info(f"Full config saved to {path}")

    # 创建 datasets, dataset_loars以及mixup
    # build datasets, dataset_loars and mixup
    dsets, dset_loaders, mixup_fn = build_loader(config)

    # 创建模型
    # build the model
    model = build_model(config)
    model.cuda()
    if dist.get_rank() == 0:
        logger.info(str(model))

    # 创建优化器
    # build the optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(config))

    # 学习率调整器
    # learning rate scheduler
    lr_scheduler, num_epochs = create_scheduler(config, optimizer)

    max_accuracy = 0.0

    start_epoch = step = 0
    if config.auto_resume and Path(config.output).joinpath("latest.pt").is_dir():
        start_epoch, step, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)

    # 混合精度以及多卡训练设置
    # Mixed-Precision and distributed training
    if config.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.amp_opt_level)
        # Apex DDP preferred unless native amp is activated
        model = ApexDDP(model, delay_allreduce=True)
        logger.info("Using NVIDIA APEX DistributedDataParallel.")
    else:
        model = DistributedDataParallel(model,
                                        device_ids=[config.local_rank],
                                        broadcast_buffers=False)
        logger.info("Using native Torch DistributedDataParallel.")
    model_without_ddp = model.module

    # 损失函数
    # loss function
    if config.aug.mixup > 0.:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if config.aug.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=config.aug.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif config.aug.smoothing > 0.:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=config.aug.smoothing)
    else:
        train_loss_fn = torch.nn.CrossEntropyLoss()

    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    for epoch in range(start_epoch, num_epochs):
        s_time = time.time()
        dset_loaders["train"].sampler.set_epoch(epoch)

        train_one_epoch(epoch, model, dset_loaders, optimizer, train_loss_fn,
                        config, lr_scheduler, logger, mixup_fn, max_accuracy, step)
        step = 0

        distribute_bn(model, dist.get_world_size(), config.dist_bn == 'reduce')

        acc1, acc5, loss = validate(config, dset_loaders["val"], model, validate_loss_fn)

        max_accuracy = max(max_accuracy, acc1)
        e_time = time.time() - s_time
        logger.info(f' * Acc@1 {acc1:.3f} Acc@5 {acc5:.3f} Max accuracy {max_accuracy:.3f}')

        if config.local_rank == 0 and (epoch % config.save_freq == 0 or epoch == (num_epochs - 1)):
            save_checkpoint(config, model.module, optimizer, lr_scheduler, epoch + 1, step, max_accuracy, logger)

    print("Done!")


def train_one_epoch(epoch, model, dset_loaders, optimizer, loss_fn,
                    config, lr_scheduler, logger, mixup_fn=None, max_accuracy=0.0, start_step=0):
    model.train()
    optimizer.zero_grad()

    data_loader = dset_loaders["train"]
    iters_per_epoch = len(data_loader)
    num_updates = epoch * iters_per_epoch

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        # 获取当前节点序号。在0号节点的0号进程上接收集群调度信息
        if hfai.distributed.get_rank() == 0 and config.local_rank == 0:
            if receive_suspend_command():
                # 挂起
                print("成功接受信号")
                save_checkpoint(config, model.module, optimizer, lr_scheduler, epoch, idx, max_accuracy, logger)
                go_suspend()

        if idx < start_step:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = loss_fn(outputs, targets)

        # compute gradient and do step
        optimizer.zero_grad()
        if config.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.opt.clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.opt.clip_grad)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
            loss.backward()
            if config.opt.clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.clip_grad)
            else:
                grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        num_updates = num_updates + 1
        lr_scheduler.step_update(num_updates=num_updates, metric=loss_meter.avg)

        torch.cuda.synchronize()
        # record loss
        loss_meter.update(loss.item(), samples.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]['lr']

        if idx % config.log_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, validate_loss_fn):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = validate_loss_fn(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
