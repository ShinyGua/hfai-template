import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.base = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.data = CN()
# path to dataset
_C.data.data_path = ""
# dataset type
_C.data.dataset_name = ""

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.model = CN()
# Name of model to train (default: "resnet50")
_C.model.name = "resnet50"
# Start with pretrained version of specified network (if avail)
_C.model.pretrained = False
# Initialize model from this checkpoint (default: none)
_C.model.initial_checkpoint = ""
# Resume full model and optimizer state from checkpoint (default: none)
_C.model.resume = ""
# number of label classes (Model default if None)
_C.model.num_classes = None
# Image patch size (default: 224)
_C.model.img_size = 224
# Input image center crop percent (for validation only)
_C.model.crop_pct = 0.825
# Mean pixel value of dataset
_C.model.mean = (0.485, 0.456, 0.406)
# Std deviation of dataset
_C.model.std = (0.229, 0.224, 0.225)
# Input batch size for training (default: 128)
_C.model.batch_size = 128
# Validation batch size override (default: None)
_C.model.validation_batch_size = None

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
_C.opt = CN()
# Optimizer (default: "sgd")
_C.opt.name = 'sgd'
# Optimizer Epsilon (default: None, use opt default)
_C.opt.eps = None
# Optimizer Betas (default: None, use opt default)
_C.opt.betas = None
# Optimizer momentum (default: 0.9)
_C.opt.momentum = 0.9
# weight decay (default: 1e-4)
_C.opt.weight_decay = 1e-4
# Clip gradient norm (default: None, no clipping)
_C.opt.clip_grad = None
# Gradient clipping mode. One of ("norm", "value", "agc")
_C.opt.clip_mode = 'norm'
# layer-wise learning rate decay (default: None)
_C.opt.layer_decay = None

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.train = CN()
# LR scheduler (default: "step")
_C.train.sched = 'cosine'
# learning rate (default: 0.005)
_C.train.lr = 5e-3
# learning rate noise on/off epoch percentages
_C.train.lr_noise = None
# learning rate noise limit percent (default: 0.67)
_C.train.lr_noise_pct = 0.67
# learning rate noise std-dev (default: 1.0)
_C.train.lr_noise_std = 1.0
# learning rate cycle len multiplier (default: 1.0)
_C.train.lr_cycle_mul = 1.0
# learning rate cycle limit, cycles enabled if > 1
_C.train.lr_cycle_limit = 1
# learning rate k-decay for cosine/poly (default: 1.0)
_C.train.lr_k_decay = 1.0
# warmup learning rate (default: 0.0001)
_C.train.warmup_lr = 1e-4
# lower lr bound for cyclic schedulers that hit 0 (1e-6)
_C.train.min_lr = 1e-6
# number of epochs to train (default: 100)
_C.train.epochs = 100
# epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).
_C.train.epoch_repeats = 0.
# manual epoch number (useful on restarts)
_C.train.start_epoch = 0
# list of decay epoch indices for multistep lr. must be increasing
_C.train.decay_milestones = [30, 60]
# epoch interval to decay LR
_C.train.decay_epochs = 100
# epochs to warmup LR, if scheduler supports
_C.train.warmup_epochs = 5
# epochs to cool down LR at min_lr, after cyclic schedule ends
_C.train.cooldown_epochs = 0
# patience epochs for Plateau LR scheduler (default: 10)
_C.train.patience_epochs = 10
# LR decay rate (default: 0.1)
_C.train.decay_rate = 0.1

# -----------------------------------------------------------------------------
# Augmentation and regularization settings
# -----------------------------------------------------------------------------
_C.aug = CN()
# Disable all training augmentation, override other train aug args
_C.aug.no_aug = False
# Random resize scale (default: 0.08 1.0)')
_C.aug.scale = [0.08, 1.0]
# Random resize aspect ratio (default: 0.75 1.33)
_C.aug.ratio = [3. / 4., 4. / 3.]
# Horizontal flip training aug probability
_C.aug.hflip = 0.5
# Vertical flip training aug probability
_C.aug.vflip = 0.
# Color jitter factor (default: 0.4)
_C.aug.color_jitter = 0.4
# Use AutoAugment policy. "v0" or "original". (default: None)
_C.aug.aa = None
# Number of augmentation repetitions (distributed training only) (default: 0)
_C.aug.aug_repeats = 0
# Number of augmentation splits (default: 0, valid: 0 or >=2)
_C.aug.aug_splits = 0
# Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.
_C.aug.jsd_loss = False
# Enable BCE loss w/ Mixup/CutMix use.
_C.aug.bce_loss = False
# Threshold for binarizing softened BCE targets (default: 0.2)
_C.aug.bce_target_thresh = 0.2
# Random erase prob (default: 0.)
_C.aug.reprob = 0.
# Random erase mode (default: "pixel")
_C.aug.remode = 'pixel'
# Random erase count (default: 1)
_C.aug.recount = 1
# Do not random erase first (clean) augmentation split
_C.aug.resplit = False
# mixup alpha, mixup enabled if > 0. (default: 0.)
_C.aug.mixup = 0.0
# cutmix alpha, cutmix enabled if > 0. (default: 0.)
_C.aug.cutmix = 0.0
# cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
_C.aug.cutmix_minmax = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.aug.mixup_prob = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.aug.mixup_switch_prob = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.aug.mixup_mode = 'batch'
# Turn off mixup after this epoch, disabled if 0 (default: 0)
_C.aug.mixup_off_epoch = 0
# Label smoothing (default: 0.1)
_C.aug.smoothing = 0.1
# Interpolation (random, bilinear, bicubic default: "random")
_C.aug.interpolation = "bicubic"
# Dropout rate (default: 0.)
_C.aug.drop = 0.0
# Drop connect rate, DEPRECATED, use drop-path (default: None)
_C.aug.drop_connect = None
# Drop path rate (default: None)
_C.aug.drop_path = None
# Drop block rate (default: None)
_C.aug.drop_block = None

# random seed (default: 42)
_C.seed = 42
# Frequency to logging info
_C.log_freq = 10
# Frequency to save checkpoint
_C.save_freq = 10
# how many training processes to use
_C.workers = 8
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.pin_mem = False
# path to output folder (default: /results)
_C.output = "results"
# local rank for DistributedDataParallel, given by command line argument
_C.local_rank = 0
# Tag of experiment, overwritten by command line argument
_C.tag = 'test'
# Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")
_C.dist_bn='reduce'

_C.eval = False
_C.throughput = False
_C.amp_opt_level = 'O1'



def _update_config_from_file(config, cfg_file):
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def update_config(config, args):
    config.defrost()
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    # merge from specific arguments
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.data_path:
        config.data.data_path = args.data_path
    if args.dataset_name:
        config.data.dataset_name = args.dataset_name
    if args.num_classes:
        config.model.num_classes = args.num_classes
    if args.tag:
        config.tag = args.tag
    if args.amp_opt_level:
        config.amp_opt_level = args.amp_opt_level
    if args.eval:
        config.eval = True
    if args.throughput:
        config.throughput = True

    config.local_rank = args.local_rank

    config.output = os.path.join(config.output, config.model.name, config.tag)

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
