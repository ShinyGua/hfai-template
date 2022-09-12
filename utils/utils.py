from pathlib import Path

import torch
import torch.distributed as dist
from hfai.checkpoint import save, load


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def load_checkpoint(config, model, optimizer, lr_scheduler, logger, name='latest.pt'):
    logger.info(f"==============> Resuming form {Path(config.output).joinpath(name)}....................")
    ckpt = load(Path(config.output).joinpath(name), map_location='cpu', nthreads=8)
    model.load_state_dict(ckpt['model'])
    if ckpt['epoch'] is not None:
        start_epoch = ckpt['epoch']
    if ckpt['step'] is not None:
        step = ckpt['step']
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if ckpt['lr_scheduler'] is not None:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    if ckpt['max_accuracy'] is not None:
        max_accuracy = ckpt['max_accuracy']
    logger.info(f"==============> success resume form to {Path(config.output).joinpath(name)}....................")
    return start_epoch, step, max_accuracy


def save_checkpoint(config, model, optimizer, lr_scheduler, epoch, step, max_accuracy, logger, name="latest.pt"):
    logger.info(f"==============> Saving to {Path(config.output).joinpath(name)}....................")
    save_state = {'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'step': step,
                  'config': config}
    save(fname=Path(config.output).joinpath(name),
         model=model,
         optimizer=optimizer,
         others=save_state)
    logger.info(f"==============> success save to {Path(config.output).joinpath(name)}....................")