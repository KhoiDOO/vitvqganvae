import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision.utils import make_grid, save_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset root directory")
    parser.add_argument("--root", type=str, required=False, default=None, help="path to dataset root directory")
    parser.add_argument("--num_images", type=int, required=False, default=25, help="number of images to load (-1 for all)")
    args = parser.parse_args()

    if args.dataset == "imagenet":
        from vitvqganvae.data.custom.imagenet import get_imagenet
        from vitvqganvae.data.custom.imagenet import denorm_imagenet as denorm

        train_ds, test_ds = get_imagenet(root=args.root, image_size=128)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    nrow = int(args.num_images**0.5)

    imgs = []

    for idx in range(args.num_images):
        img = test_ds[idx]
        imgs.append(denorm(img))

    grid = make_grid(imgs, nrow=nrow)

    os.makedirs("./.samples", exist_ok=True)
    save_image(grid, f"./.samples/{args.dataset}_samples.png")