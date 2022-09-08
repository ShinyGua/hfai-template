import hf_env
hf_env.set_env('202111')

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
from timm.utils import AverageMeter, accuracy, distribute_bn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from configs.config import get_config
from data import build_loader
from model import build_model
from utils import optimizer_kwargs, create_scheduler, reduce_tensor, get_grad_norm
from utils.logger import create_logger

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('pytorch template training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default="", metavar="FILE", help='path to config file', )
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--dataset-name', type=str, help='Dataset type')
    parser.add_argument('--num-classes', type=int, help='Number of label classes')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', default="test", help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(config):
    dsets, dset_loaders, mixup_fn = build_loader(config)

    model = build_model(config)
    model.cuda()
    if dist.get_rank() == 0:
        logger.info(str(model))

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(config))

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

    lr_scheduler, num_epochs = create_scheduler(config, optimizer)

    if config.eval:
        validate(config, dset_loaders["val"], model, validate_loss_fn)
        return

    max_accuracy = 0.0

    start_epoch = 0

    summary_writer = SummaryWriter("runs/" + config.tag) if config.local_rank == 0 else None

    for epoch in range(start_epoch, num_epochs):
        s_time = time.time()
        dset_loaders["train"].sampler.set_epoch(epoch)

        train_one_epoch(epoch, model, dset_loaders, optimizer, train_loss_fn,
                        config, lr_scheduler, summary_writer, mixup_fn)

        distribute_bn(model, dist.get_world_size(), config.dist_bn == 'reduce')

        acc1, acc5, loss = validate(config, dset_loaders["val"], model, validate_loss_fn)

        max_accuracy = max(max_accuracy, acc1)
        e_time = time.time() - s_time
        logger.info(f' * Acc@1 {acc1:.3f} Acc@5 {acc5:.3f} Max accuracy {max_accuracy:.3f}')
        if config.local_rank == 0:
            summary_writer.add_scalar('Test/Acc@1', acc1, epoch)
            summary_writer.add_scalar('Test/Acc@5', acc5, epoch)
            summary_writer.add_scalar('Test/Max Acc@1', max_accuracy, epoch)
            summary_writer.add_scalar('Test/loss', loss, epoch)
            summary_writer.add_scalar('Time/epoch time', e_time, epoch)
            print(f"EPOCH {epoch} takes {datetime.timedelta(seconds=int(e_time))}")


def train_one_epoch(epoch, model, dset_loaders, optimizer, loss_fn,
                    config, lr_scheduler,summary_writer, mixup_fn=None):
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

        if config.local_rank == 0:
            summary_writer.add_scalar("Train/loss", loss_meter.val, num_updates)
            summary_writer.add_scalar("Train/lr", lr, num_updates)
            summary_writer.add_scalar("Train/grad norm", norm_meter.val, num_updates)
            summary_writer.add_scalar("Time/data time", data_time.val, num_updates)
            summary_writer.add_scalar("Time/batch time", batch_time.val, num_updates)

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

        if idx % config.log_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    _, config = parse_option()

    if config.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.output, exist_ok=True)
    logger = create_logger(output_dir=config.output, dist_rank=dist.get_rank(), name=f"{config.model.name}")

    if dist.get_rank() == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    if dist.get_rank() == 0:
        logger.info(config.dump())

    main(config)
