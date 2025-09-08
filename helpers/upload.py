from vitvqganvae import model
from vitvqganvae.model import hf
from huggingface_hub import PyTorchModelHubMixin

from vitvqganvae.utils.config import load_config, parse_structured, config_to_primitive

import argparse
import os
import shutil


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=False, default=None, help="Repository name to push the model to")
    parser.add_argument("--path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--model_class", type=str, required=False, default="VQVAEPipeline", help="Model class to use (e.g., VQVAEPipeline)")
    args = parser.parse_args()

    trial_dir = os.path.normpath(os.path.join(args.path, '..', '..'))

    cfg = load_config(os.path.join(trial_dir, "config.yaml"))

    model_config_cls = getattr(model, cfg.model_config)
    model_config = parse_structured(model_config_cls, cfg.model_kwargs)
    model_config = config_to_primitive(model_config)
    model_cls = getattr(hf, args.model_class)
    model_module: PyTorchModelHubMixin = model_cls(**model_config)
    model_module.load(args.path)

    repo_id = cfg.name if args.repo is None else args.repo

    try:
        model_module.push_to_hub(
            repo_id=repo_id,
        )
    except Exception as e:
        print(f"Failed to push model to Hugging Face: {e}")

    shutil.rmtree(cfg.trial_dir, ignore_errors=True)