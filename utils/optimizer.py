import torch.optim as optim
from timm.optim import Lamb


def optimizer_kwargs(config):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=config.opt.name,
        lr=config.train.lr,
        weight_decay=config.opt.weight_decay,
        momentum=config.opt.momentum)
    if config.opt.eps is not None:
        kwargs['eps'] = config.opt_eps
    if config.opt.betas is not None:
        kwargs['betas'] = config.opt_betas
    if config.opt.layer_decay  is not None:
        kwargs['layer_decay'] = config.layer_decay
    return kwargs
