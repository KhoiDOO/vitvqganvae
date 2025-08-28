from contextlib import nullcontext, contextmanager
from typing import Optional, Type

from accelerate import Accelerator
from functools import partial

from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from ..utils.helpers import exists

import pytorch_warmup as warmup


ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)


class OptimizerWithWarmupSchedule(nn.Module):
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 0,
        max_grad_norm: Optional[float] = None
    ):
        super().__init__()
        self.max_grad_norm = max_grad_norm
        has_warmup = warmup_steps > 0

        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = ConstantLRScheduler(optimizer)

        self.warm_steps = warmup_steps
        self.warmup = warmup.LinearWarmup(optimizer, warmup_period = self.warm_steps) if has_warmup else None

        self.optimizer = optimizer

        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator
    
    def prepare(self):
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

    def state_dict(self):
        pkg = dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict()
        )

        if exists(self.warmup):
            pkg['warmup'] = self.warmup.state_dict()

        self.accelerator.step

        return pkg

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])

        try:
            self.scheduler.load_state_dict(pkg['scheduler'])
        except Exception as e:
            # raise Warning(f"The scheduler is different from the previous run. causing error: {e}")
            self.accelerator.print(f"Warning: The scheduler is different from the previous run. Error: {e}")

        if exists(self.warmup):
            self.warmup.load_state_dict(pkg['warmup'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if exists(self.max_grad_norm):
            for param_group in self.optimizer.param_groups:
                self.accelerator.clip_grad_norm_(param_group['params'], self.max_grad_norm)

        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            context = nullcontext if not exists(self.warmup) else self.warmup.dampening

            with context():
                if exists(self.warmup):
                    if self.warmup.last_step + 1 >= self.warm_steps:
                        self.scheduler.step()
                else:
                    self.scheduler.step()