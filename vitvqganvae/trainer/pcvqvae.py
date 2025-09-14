from .vqvae import VQVAETrainer, VQVAETrainerConfig

from dataclasses import dataclass
from beartype import beartype
from pytorch_custom_utils import add_wandb_tracker_contextmanager


@dataclass
class PCVQVAETrainerConfig(VQVAETrainerConfig):
    save_results_every: int | None = None


@beartype
@add_wandb_tracker_contextmanager()
class PCVQVAETrainer(VQVAETrainer):
    def __init__(self, *args, **kwargs):
        assert kwargs.get('save_results_every') is None, "save_results_every is not allowed for PCVQVAETrainer"
        super().__init__(*args, **kwargs)