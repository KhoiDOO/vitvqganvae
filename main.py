import os
import argparse

from omegaconf import OmegaConf

from vitvqganvae.utils.config import (
    ExperimentConfig, 
    load_config, 
    parse_structured, 
    config_to_primitive,
    dump_config
)
from vitvqganvae import model, trainer, data
from vitvqganvae.trainer.utils import trackers

from torchinfo import summary

from vitvqganvae.utils.helpers import set_seed


def main(args, extras):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras, n_gpus=n_gpus, **{
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
        raise NotImplementedError("Custom dataset not yet implemented.")
        from vitvqganvae.data import custom
    else:
        raise ValueError(f"Unknown dataset source: {cfg.dataset_source}")

    dataset_config = config_to_primitive(cfg.dataset_kwargs)
    img_size = dataset_config.get("image_size", 64)

    print(f'Number of training samples: {len(train_ds)}')
    print(f'Number of validation samples: {len(valid_ds)}')

    # model
    model_config_cls = getattr(model, cfg.model_config)
    model_config = parse_structured(model_config_cls, cfg.model_kwargs)
    model_config = config_to_primitive(model_config)
    model_cls = getattr(model, cfg.model)
    model_module = model_cls(**model_config)
    try:
        summary(model_module, input_size=(1, 3, img_size, img_size))
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
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="run training")
    group.add_argument("--resume", action="store_true", help="resume training from a previous run")

    args, extras = parser.parse_known_args()

    main(args, extras)