from torch import Tensor

from vitvqganvae.utils.config import (
    ExperimentConfig, 
    load_config, 
    parse_structured, 
    config_to_primitive,
    dump_config
)
from vitvqganvae import model, trainer
from vitvqganvae.trainer.utils import trackers

from torchinfo import summary

from vitvqganvae.utils.helpers import set_seed
from accelerate import Accelerator

import os
import argparse
import copy


def main(args, extras):
    accelerator = Accelerator()
    n_gpus = accelerator.num_processes
    selected_gpus = [str(i) for i in range(n_gpus)]

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras, **{
        "n_gpus": n_gpus,
        "distributed_type": accelerator.distributed_type,
        "mixed_precision": accelerator.mixed_precision,
        "train": args.train,
        "resume": args.resume
    })

    set_seed(cfg.seed)

    print(f"Running with {n_gpus} GPU(s): {', '.join(selected_gpus)}")
    dump_config(os.path.join(cfg.trial_dir, "config.yaml"), cfg)
    
    # dataset
    if cfg.dataset_source == "torchvision":
        from vitvqganvae.data import tv
        from vitvqganvae.data.tv.wrapper import TVDataset
        dataset_getter = getattr(tv, f"get_{cfg.dataset_name}")
        train_ds, valid_ds = dataset_getter(**cfg.dataset_kwargs)
        train_ds, valid_ds = TVDataset(train_ds, cfg.dataset_img_key), TVDataset(valid_ds, cfg.dataset_img_key)
    elif cfg.dataset_source == "custom":
        from vitvqganvae.data import custom
        dataset_getter = getattr(custom, f"get_{cfg.dataset_name}")
        train_ds, valid_ds = dataset_getter(**cfg.dataset_kwargs)
    elif cfg.dataset_source == "hf":
        from vitvqganvae.data import hf
        dataset_getter = getattr(hf, f"get_{cfg.dataset_name}")
        train_ds, valid_ds = dataset_getter(**cfg.dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset source: {cfg.dataset_source}")

    print(f'Number of training samples: {len(train_ds)}')
    print(f'Number of validation samples: {len(valid_ds)}')

    # model
    model_config_cls = getattr(model, cfg.model_config)
    model_config = parse_structured(model_config_cls, cfg.model_kwargs)
    model_config = config_to_primitive(model_config)
    model_cls = getattr(model, cfg.model)
    model_module = model_cls(**model_config)
    try:
        sample: Tensor = train_ds[0].unsqueeze(0)
        summary(
            copy.deepcopy(model_module),
            input_data=sample,
            col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"],
            # depth=2
        )
    except Exception as e:
        print(f"Cannot run model summary: {e}")

    # trainer
    trainer_config_cls = getattr(trainer, cfg.trainer_config)
    trainer_config = parse_structured(trainer_config_cls, cfg.trainer_kwargs)
    trainer_config = config_to_primitive(trainer_config)
    trainer_cls = getattr(trainer, cfg.trainer)
    trainer_module = trainer_cls(
        model=model_module,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        trial_dir=cfg.trial_dir,
        **trainer_config
    )

    print(f"Trial directory: {trainer_module.trial_dir}")

    if cfg.trainer_kwargs.use_wandb_tracking:
        with trackers(
            trainer_module, 
            project_name=cfg.wandb['project_name'], 
            run_name=cfg.wandb['run_name'],
            hps=config_to_primitive(config=cfg, resolve=True), 
            init_kwargs=cfg.wandb['kwargs']
        ):
            trainer_module()
    else:
        trainer_module()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="run training")
    group.add_argument("--resume", action="store_true", help="resume training from a previous run")

    args, extras = parser.parse_known_args()

    main(args, extras)