from timm.scheduler import CosineLRScheduler, TanhLRScheduler, StepLRScheduler


def create_scheduler(config, optimizer):
    num_epochs = config.train.epochs

    if getattr(config.train, 'lr_noise', None) is not None:
        lr_noise = getattr(config.train, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(config.train, 'lr_noise_pct', 0.67),
        noise_std=getattr(config.train, 'lr_noise_std', 1.),
        noise_seed=getattr(config, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(config.train, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(config.train, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(config.train, 'lr_cycle_limit', 1),
    )

    if config.train.sched  == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=config.train.min_lr,
            warmup_lr_init=config.train.warmup_lr,
            warmup_t=config.train.warmup_epochs,
            k_decay=getattr(config.train, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + config.train.cooldown_epochs
    elif config.train.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=config.train.min_lr,
            warmup_lr_init=config.train.warmup_lr,
            warmup_t=config.train.warmup_epochs,
            t_in_epochs=True,
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + config.train.cooldown_epochs
    elif config.train.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=config.train.decay_epochs,
            decay_rate=config.train.decay_rate,
            warmup_lr_init=config.train.warmup_lr,
            warmup_t=config.train.warmup_epochs,
            **noise_args,
        )
    else:
        raise f"{config.train.sched} is not supported now!"

    return lr_scheduler, num_epochs

