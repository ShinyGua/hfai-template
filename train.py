# import hf_env
#
# hf_env.set_env('diff_hfai')

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
    parser.add_argument('--tag', default="test_hfai", help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(local_rank):
    _, config = parse_option()
    if config.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"
    os.makedirs(config.output, exist_ok=True)

    # Multi-node communication
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # number of node
    rank = int(os.environ['RANK'])  # rank of current node
    gpus = torch.cuda.device_count()  # Number of GPUs per node

    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    # fix the random seed
    seed = config.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # create the logger
    logger = create_logger(output_dir=config.output, dist_rank=local_rank, name=f"{config.model.name}")

    if dist.get_rank() == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
