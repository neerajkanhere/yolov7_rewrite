import torch
import numpy as np
import random


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(
        self, optimizer, warmup_epochs, train_epochs, train_loader, last_epoch=-1
    ):
        self.warmup_steps = warmup_epochs * len(train_loader)
        self.t_total = train_epochs * len(train_loader)
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )
