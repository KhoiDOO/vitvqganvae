import os
import argparse

from vitvqganvae.utils.config import ExperimentConfig, load_config, parse_structured, config_to_primitive
from vitvqganvae import model, trainer
from vitvqganvae.trainer.utils import trackers

from accelerate.utils import DistributedDataParallelKwargs


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
    
    # dataset

    # model
    model_config_cls = getattr(model, cfg.model_config)
    model_config = parse_structured(model_config_cls, cfg.model_kwargs)
    model_config = config_to_primitive(model_config)
    model_cls = getattr(model, cfg.model)
    model_module = model_cls(**model_config)

    # trainer
    # trainer_config_cls = getattr(trainer, cfg.trainer_config)
    # trainer_config = parse_structured(trainer_config_cls, cfg.trainer_kwargs)
    # trainer_cls = getattr(trainer, cfg.trainer)
    # trainer_module = trainer_cls(**trainer_config)

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