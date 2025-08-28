import os
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob

from omegaconf import OmegaConf, DictConfig
from typing import Optional, Any, Union

from sympy import false

OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("mod", lambda a, b: a % b)
OmegaConf.register_new_resolver("pow", lambda a, b: a ** b)
OmegaConf.register_new_resolver("last", lambda dir: os.path.basename(os.path.normpath(dir)))

def convert_timestamp_format(timestamp: int) -> str:
    """
    Convert timestamp from format '20250726200718' to '2025_07_26_20_07_18'
    
    Args:
        timestamp: Timestamp in format 'YYYYMMDDHHMMSS'
        
    Returns:
        Formatted timestamp string in format 'YYYY_MM_DD_HH_MM_SS'
    """
    if len(str(timestamp)) != 14:
        raise ValueError(f"Timestamp must be 14 characters long, got {len(str(timestamp))}")

    timestamp_str = str(timestamp)
    year = timestamp_str[:4]
    month = timestamp_str[4:6]
    day = timestamp_str[6:8]
    hour = timestamp_str[8:10]
    minute = timestamp_str[10:12]
    second = timestamp_str[12:14]
    
    return f"{year}_{month}_{day}_{hour}_{minute}_{second}"

@dataclass
class ExperimentConfig:
    name: str = "default"
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = ""
    trial_dir: str = ""
    n_gpus: int = 1
    seed: int = 42
    train: bool = False  # whether to train the model
    resume: bool = False  # whether to resume from a previous run

    ds_type: str = ""
    ds: dict = field(default_factory=dict)

    model: str = ""
    model_config: str = ""
    model_kwargs: dict = field(default_factory=dict)

    trainer: str = ""
    trainer_config: str = ""
    trainer_kwargs: dict = field(default_factory=dict)

    wandb: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.exp_root_dir is None:
            self.exp_root_dir = "outputs"
        os.makedirs(self.exp_root_dir, exist_ok=True)

        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        os.makedirs(self.exp_dir, exist_ok=True)

        if self.resume:
            raise ValueError("Resume is not supported yet.")
            if 'resume' not in self.wandb['kwargs']:
                raise ValueError("Resume must be set in wandb kwargs when resuming a run.")
            if self.wandb['run_name'] is None:
                raise ValueError("Run name must be set when resuming a run.")
            self.wandb['run_name'] = convert_timestamp_format(self.wandb['run_name'])
            trial_dir = os.path.join(self.exp_dir, self.wandb['run_name'])
            print(f"Resuming from trial directory: {trial_dir}")
            if not os.path.exists(trial_dir):
                raise ValueError(f"Trial directory {trial_dir} does not exist for resuming.")

            self.trial_dir = os.path.join(trial_dir, f"resume_{len(glob(os.path.join(trial_dir, 'resume_*')))}")
            os.makedirs(self.trial_dir, exist_ok=True)
        elif self.train:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.wandb['run_name'] = now

            self.trial_dir = os.path.join(self.exp_dir, now)
            os.makedirs(self.trial_dir, exist_ok=True)
        else:
            raise ValueError("At least one of 'train', 'resume' must be set to True.")

def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> ExperimentConfig:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)

def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg