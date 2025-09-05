import os
import argparse
import torch

from vitvqganvae.utils.config import (
    ExperimentConfig, 
    load_config
)

from vitvqganvae.utils.helpers import set_seed

from PIL import Image
from tqdm import tqdm

def main(args, extras):

    config_path = os.path.join(args.trial_dir, "config.yaml")

    cfg: ExperimentConfig = load_config(config_path)

    set_seed(cfg.seed)

    # dataset
    if cfg.dataset_source == "torchvision":
        from vitvqganvae.data import tv
        from vitvqganvae.data.tv.wrapper import TVDataset
        dataset_getter = getattr(tv, f"get_{cfg.dataset_name}")
        train_ds, valid_ds = dataset_getter(**cfg.dataset_kwargs)
        train_ds, valid_ds = TVDataset(train_ds, cfg.dataset_img_key), TVDataset(valid_ds, cfg.dataset_img_key)

    print(f'Number of training samples: {len(train_ds)}')
    print(f'Number of validation samples: {len(valid_ds)}')

    save_dir = os.path.join(args.trial_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm(range(args.num_samples), desc="Generating train samples"):
        sample = train_ds[idx]
        img = sample + 0.5
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f"train_{idx}.png"))
    
    for idx in tqdm(range(args.num_samples), desc="Generating valid samples"):
        sample = valid_ds[idx]
        img = sample + 0.5
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f"valid_{idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir", required=True, help="path to trial directory")
    parser.add_argument("--num_samples", type=int, default=20, help="number of samples to generate")

    args = parser.parse_args()

    main(args, None)