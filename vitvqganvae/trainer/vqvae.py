import os
import time
import torch

from pathlib import Path
from functools import partial
from contextlib import nullcontext
from dataclasses import dataclass

from torch.nn import Module

from torch.utils.data import Dataset, DataLoader
from vitvqganvae import data

from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from .utils import OptimizerWithWarmupSchedule

from torchvision.utils import save_image
from pytorch_custom_utils import add_wandb_tracker_contextmanager

from accelerate import Accelerator
from accelerate.utils import DistributedType, DistributedDataParallelKwargs, DataLoaderConfiguration

from beartype import beartype
from beartype.typing import Optional

from ema_pytorch import EMA

from . import opt
from ..utils.helpers import cycle, divisible_by, accum_log, default


DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

@dataclass
class VQVAETrainerConfig:
    num_train_steps: int = 10000,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    grad_accum_every: int = 1,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.,
    max_grad_norm: float | None = None,
    val_every: int = 1,
    val_num_batches: int = 5,
    val_num_images: int = 32,
    scheduler: str | None = None,
    scheduler_kwargs: dict = dict(),
    ema_kwargs: dict | None = None,
    accelerator_kwargs: dict = dict(),
    optimizer_name: str = "Adam",
    optimizer_kwargs: dict = dict(),
    loss_lambda: dict = dict(),
    checkpoint_every: int | None = None,
    save_results_every: int | None = None,
    warmup_steps: int = 1000,
    use_wandb_tracking: bool = False,
    resume: bool = False,
    from_checkpoint: str | None = None,
    from_checkpoint_type: str | None = None,


@beartype
@add_wandb_tracker_contextmanager()
class VQVAETrainer(Module):
    def __init__(
        self,
        model: Module,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset],
        trial_dir: str,
        num_train_steps: int = 10000,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        grad_accum_every: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.,
        max_grad_norm: float | None = None,
        val_every: int = 1,
        val_num_batches: int = 5,
        val_num_images: int = 32,
        scheduler: str | None = None,
        scheduler_kwargs: dict = dict(),
        ema_kwargs: dict | None = None,
        accelerator_kwargs: dict = dict(),
        optimizer_name: str = "Adam",
        optimizer_kwargs: dict = dict(),
        loss_lambda: dict = dict(),
        checkpoint_every: int | None = None,
        save_results_every: int | None = None,
        checkpoint_key: str | None = None,
        warmup_steps: int = 1000,
        use_wandb_tracking: bool = False,
        resume: bool = False,
        from_checkpoint: str | None = None,
        from_checkpoint_type: str | None = None,
    ):
        super().__init__()

        self._model = model
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._trial_dir = trial_dir
        self._num_train_steps = num_train_steps
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._grad_accum_every = grad_accum_every
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._max_grad_norm = max_grad_norm
        self._val_every = val_every
        self._val_num_batches = val_num_batches
        self._val_num_images = val_num_images
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs
        self._ema_kwargs = ema_kwargs
        self._accelerator_kwargs = accelerator_kwargs
        self._optimizer_name = optimizer_name
        self._optimizer_kwargs = optimizer_kwargs
        self._loss_lambda = loss_lambda
        self._checkpoint_every = checkpoint_every
        self._save_results_every = save_results_every
        self._checkpoint_key = default(checkpoint_key, "val_recon_loss")
        self._warmup_steps = warmup_steps
        self._use_wandb_tracking = use_wandb_tracking
        self._resume = resume
        self._from_checkpoint = from_checkpoint
        self._from_checkpoint_type = from_checkpoint_type
        self.register_buffer('step', torch.tensor(0))

        if resume:
            raise NotImplementedError("Resume training is not implemented yet.")

        self.checkpoint_folder = os.path.join(trial_dir, "checkpoints")
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        self.generation_folder = os.path.join(trial_dir, "generation")
        os.makedirs(self.generation_folder, exist_ok=True)
        
        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        accelerator_kwargs['dataloader_config'] = DataLoaderConfiguration(
            split_batches = True,
            even_batches = True,
        )

        self.accelerator = Accelerator(
            **accelerator_kwargs
        )

        optimizer: Optimizer = getattr(opt, self._optimizer_name)(
            [t for t in self._model.parameters() if t.requires_grad],
            lr=self._learning_rate * self._warmup_steps if self._warmup_steps > 0 else self._learning_rate,
            weight_decay=self._weight_decay,
            **self._optimizer_kwargs
        )

        scheduler: _LRScheduler = getattr(lr_scheduler, self._scheduler) if self._scheduler else None

        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator=self.accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm
        )

        self.train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=True,
            drop_last=True,
        )
    
        self.val_dataloader = DataLoader(
            self._valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
            drop_last=True,
        )

        self.custom_make_grid = None
        dataset_name = self.train_dataset.__class__.__name__.lower()
        self.custom_make_grid = getattr(data, f"make_grid_{dataset_name}", None)
        
        if self.custom_make_grid is None:
            raise NotImplementedError(f"Custom make_grid function not found for {dataset_name}.")
        
        (
            self._model,
            self.train_dataloader,
            self.val_dataloader,
            self.optimizer.optimizer, 
            self.optimizer.scheduler
        ) = self.accelerator.prepare(
            self._model,
            self.train_dataloader,
            self.val_dataloader,
            self.optimizer.optimizer, 
            self.optimizer.scheduler
        )

        self.use_ema = self._ema_kwargs is not None
        if self.use_ema:
            self.ema_model = EMA(model, **self._ema_kwargs)
            self.ema_model = self.accelerator.prepare(self.ema_model)
            print(f"Total EMA model parameters: {sum(p.numel() for p in self.ema_model.parameters())}")

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self._model)
    
    @property
    def unwrapped_ema_model(self):
        return self.accelerator.unwrap_model(self.ema_model)

    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO \
                and self.accelerator.num_processes == 1)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)
    
    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item()
        )

        torch.save(pkg, str(path))

    def save_ema(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_ema_model.state_dict(),
            step = self.step.item()
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cpu", weights_only=False)

        # Load model state dict
        if isinstance(self._model, torch.nn.DataParallel) or isinstance(self._model, torch.nn.parallel.DistributedDataParallel):
            self._model.module.load_state_dict(pkg['model'])
        else:
            self._model.load_state_dict(pkg['model'])

        # Load optimizer state dict with proper device handling
        self.optimizer.load_state_dict(pkg['optimizer'])
        
        # Move optimizer state to correct device if needed
        for state in self.optimizer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.accelerator.device)

        # Load step
        self.step.copy_(pkg['step'])
    
    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())
    
    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        forward_kwargs = {
            'img': data,
            'return_loss': True,
            'return_recons': False
        }

        return forward_kwargs
    
    def forward(self):
        step = self.step.item()
        train_dl_iter: DataLoader = cycle(self.train_dataloader)
        val_dl_iter: DataLoader = cycle(self.val_dataloader)

        best_loss = float('inf')
        best_step = 0

        self._model.train()

        if self.use_ema:
            ema_model = self.ema_model.module if self.is_distributed else self.ema_model

        while step < self._num_train_steps:

            train_log = {}

            step_start_time = time.time()
            total_forward_time = 0
            total_backward_time = 0

            for i in range(self._grad_accum_every):
                is_last = i == (self._grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self._model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(train_dl_iter)

                forward_start_time = time.time()
                with self.accelerator.autocast(), maybe_no_sync():
                    train_loss_dct: dict = self._model(**forward_kwargs)
                    
                    train_loss = 0.0
                    for key, value in train_loss_dct.items():
                        if key in self._loss_lambda:
                            if self._loss_lambda[key] is None:
                                raise ValueError(f"Value for loss_lambda['{key}'] is None.")
                            value = value * self._loss_lambda[key]
                        else:
                            raise ValueError(f"Loss key '{key}' not found in loss_lambda dictionary.")
                        train_loss += value
                total_forward_time += time.time() - forward_start_time

                backward_start_time = time.time()
                self.accelerator.backward(train_loss / self._grad_accum_every)
                total_backward_time += time.time() - backward_start_time

                accum_log(
                    train_log, 
                    {key: value / self._grad_accum_every for key, value in train_loss_dct.items()}
                )

            optimizer_start_time = time.time()
            self.optimizer.step()
            self.optimizer.zero_grad()
            optimizer_time = time.time() - optimizer_start_time

            total_step_time = time.time() - step_start_time

            self.log(
                **train_log,
                forward_time = total_forward_time,
                backward_time = total_backward_time,
                optimizer_time = optimizer_time,
                total_step_time = total_step_time,
                lr = self.optimizer.optimizer.param_groups[0]['lr'],
            )

            if self.use_ema:
                ema_model.update()

            step += 1
            self.step.add_(1)

            self.wait()

            if divisible_by(step, self._val_every) or step == self._num_train_steps:
                self._model.eval()
                
                val_total_loss_dct = {}
                
                if step == self._num_train_steps:
                    num_val_batches = len(self.val_dataloader)
                else:
                    num_val_batches = self._val_num_batches
                
                with torch.no_grad():
                    for _ in range(num_val_batches):
                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)
                        val_loss_dct = self._model(**forward_kwargs)
                        accum_log(val_total_loss_dct, val_loss_dct)

                for key in val_total_loss_dct:
                    val_total_loss_dct[key] /= num_val_batches

                gathered_val_losses = self.accelerator.gather_for_metrics(val_total_loss_dct)

                if self.is_main:
                    val_losses = {f"val_{key}": value.mean().item() for key, value in gathered_val_losses.items()}
                    self.log(**val_losses)
                    self.print(f"Step {step}: Train Loss: {train_loss.item()} - Validation Loss: {val_losses}")

                    if val_losses[self._checkpoint_key] < best_loss:
                        best_loss = val_losses[self._checkpoint_key]
                        best_step = step

                        self.save(os.path.join(self.checkpoint_folder, f'model_ckpt_best.pt'))
                        if self.use_ema:
                            self.save_ema(os.path.join(self.checkpoint_folder, f'model_ckpt_ema_best.pt'))
                        
                    self.save(os.path.join(self.checkpoint_folder, f'model_ckpt_last.pt'))

                self._model.train()

            if self._checkpoint_every:
                if self.is_main and divisible_by(step, self._checkpoint_every):
                    self.save(os.path.join(self.checkpoint_folder, f'model_ckpt_{step}.pt'))
                    if self.use_ema:
                        self.save_ema(os.path.join(self.checkpoint_folder, f'model_ckpt_ema_{step}.pt'))

            if self._save_results_every:
                if self.is_main and divisible_by(step, self._save_results_every):
                    models_to_evaluate = ((self.unwrapped_model, str(step)),)
                    if self.use_ema:
                        models_to_evaluate += ((self.unwrapped_ema_model, f"{step}_ema"),)

                    for model, filename in models_to_evaluate:
                        model.eval()

                        val_data = next(val_dl_iter)
                        val_data = val_data[:self._val_num_images].to(self.device)

                        _, recons = model(val_data, return_loss = True, return_recons = True)

                        grid = self.custom_make_grid(val_data, recons, nrow=2)

                        save_image(grid, os.path.join(self.generation_folder, f'{filename}.png'))
                        
                        model.train()

        self.print(f'Training Complete: best loss {best_loss} at step {best_step}')

    @property
    def model(self):
        return self._model
    
    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def trial_dir(self):
        return self._trial_dir

    @property
    def num_train_steps(self):
        return self._num_train_steps

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def pin_memory(self):
        return self._pin_memory

    @property
    def grad_accum_every(self):
        return self._grad_accum_every

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def weight_decay(self):
        return self._weight_decay

    @property
    def max_grad_norm(self):
        return self._max_grad_norm

    @property
    def val_every(self):
        return self._val_every

    @property
    def val_num_batches(self):
        return self._val_num_batches

    @property
    def val_num_images(self):
        return self._val_num_images

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_kwargs(self):
        return self._scheduler_kwargs

    @property
    def ema_kwargs(self):
        return self._ema_kwargs

    @property
    def accelerator_kwargs(self):
        return self._accelerator_kwargs

    @property
    def optimizer_name(self):
        return self._optimizer_name

    @property
    def optimizer_kwargs(self):
        return self._optimizer_kwargs

    @property
    def loss_lambda(self):
        return self._loss_lambda

    @property
    def checkpoint_every(self):
        return self._checkpoint_every

    @property
    def warmup_steps(self):
        return self._warmup_steps

    @property
    def use_wandb_tracking(self):
        return self._use_wandb_tracking

    @property
    def resume(self):
        return self._resume

    @property
    def from_checkpoint(self):
        return self._from_checkpoint

    @property
    def from_checkpoint_type(self):
        return self._from_checkpoint_type

