# Datasets

This document provides instructions on how to use the datasets available in this project.

## Dataset Sources

The datasets are organized into three sources:
- `custom`: Custom datasets that require manual download and setup.
- `hf`: Datasets hosted on the Hugging Face Hub.
- `tv`: Datasets available through `torchvision.datasets`.

---

## Custom Datasets

### FFHQ (Flickr-Faces-HQ)

- **File:** `vitvqganvae/data/custom/ffhq.py`
- **Description:** A high-quality image dataset of human faces.
- **Setup:** Download the `images1024x1024` folder from the official FFHQ repository. The directory should contain the `.png` image files.
- **Usage:**
  - The `get_ffhq` function returns training and validation `torch.utils.data.Dataset` objects.
- **Parameters for `get_ffhq`:**
  - `root` (str): The root directory where the FFHQ dataset `images1024x1024` folder is located. This is a required parameter.
  - `image_size` (int, default=64): The desired output image size. Supported sizes are `64`, `128`, `256`, `512`, `1024`.
  - `split` (list[int], default=[60000, 10000]): A list of two integers specifying the number of samples for the training and validation sets, respectively. The sum of these values must equal the total number of images in the dataset directory.
- **Example:**
  ```python
  from vitvqganvae.data.custom import get_ffhq
  train_dataset, val_dataset = get_ffhq(root="/path/to/ffhq/images1024x1024", image_size=256)
  ```

### ImageNet

- **File:** `vitvqganvae/data/custom/imagenet.py`
- **Description:** The ImageNet dataset.
- **Setup:** The dataset should be organized as follows:
  ```
  <root>/
    train/
      n01440764/
        n01440764_10026.JPEG
        ...
      ...
    val/
      ILSVRC2012_val_00000001.JPEG
      ...
    test/
      ILSVRC2012_test_00000001.JPEG
      ...
  ```
- **Usage:**
  - The `get_imagenet` function returns training and test `torch.utils.data.Dataset` objects. The validation set is concatenated with the training set.
- **Parameters for `get_imagenet`:**
  - `root` (str): The root directory of the ImageNet dataset. This is a required parameter.
  - `image_size` (int, default=64): The desired output image size. Supported sizes are `64`, `128`, `256`.
- **Example:**
  ```python
  from vitvqganvae.data.custom import get_imagenet
  train_dataset, test_dataset = get_imagenet(root="/path/to/imagenet", image_size=256)
  ```

---

## Hugging Face Datasets

These datasets are downloaded automatically from the Hugging Face Hub. You just need to provide a cache directory.

### Ellipsoid1024

- **File:** `vitvqganvae/data/hf/epllipsoid1024.py`
- **Description:** A dataset of 1024-point ellipsoids.
- **Hugging Face Hub:** [kohido/ellipsoid_1024pts](https://huggingface.co/datasets/kohido/ellipsoid_1024pts)
- **Usage:**
  - The `get_ellipsoid1024` function returns training and validation `torch.utils.data.Dataset` objects.
- **Parameters for `get_ellipsoid1024`:**
  - `root` (str): The directory to cache the downloaded dataset. This is a required parameter.
- **Example:**
  ```python
  from vitvqganvae.data.hf import get_ellipsoid1024
  train_dataset, val_dataset = get_ellipsoid1024(root="/path/to/cache/dir")
  ```

### Mesh500

- **File:** `vitvqganvae/data/hf/mesh500.py`
- **Description:** A dataset of 3D meshes with either 1024 or 4096 points. The points are scaled to a `[-0.5, 0.5]` cube. Data augmentation (random rotation) is applied to the training set by default.
- **Hugging Face Hub:**
    - [kohido/mesh500_1024pts](https://huggingface.co/datasets/kohido/mesh500_1024pts)
    - [kohido/mesh500_4096pts](https://huggingface.co/datasets/kohido/mesh500_4096pts)
- **Usage:**
  - `get_mesh500`, `get_mesh500_1024`, and `get_mesh500_4096` functions are available.
- **Parameters for `get_mesh500`:**
  - `root` (str): The directory to cache the downloaded dataset. This is a required parameter.
  - `num_points` (int, default=1024): The number of points per mesh. Can be `1024` or `4096`.
  - `split` (float, default=0.8): The proportion of the dataset to use for training.
  - `augment` (bool, default=True): Whether to apply random rotation augmentation to the training set.
- **Example:**
  ```python
  from vitvqganvae.data.hf import get_mesh500_1024
  train_dataset, val_dataset = get_mesh500_1024(root="/path/to/cache/dir")
  ```

---

## Torchvision Datasets

These datasets are downloaded automatically via `torchvision`.

### CelebA

- **File:** `vitvqganvae/data/tv/celeba.py`
- **Description:** CelebFaces Attributes Dataset.
- **Usage:**
  - The `get_celeba` function returns training and test `torch.utils.data.Dataset` objects. The original `train` and `valid` splits are concatenated to form the training set.
- **Parameters for `get_celeba`:**
  - `root` (str): The directory to download the dataset to.
  - `download` (bool, default=True): Whether to download the dataset if not found.
  - `image_size` (int, default=64): The desired output image size. Must be less than or equal to 128.
- **Example:**
  ```python
  from vitvqganvae.data.tv import get_celeba
  train_dataset, test_dataset = get_celeba(root="/path/to/data", image_size=128)
  ```

### CIFAR

- **File:** `vitvqganvae/data/tv/cifar.py`
- **Description:** CIFAR-10 and CIFAR-100 datasets.
- **Usage:**
  - `get_cifar10` and `get_cifar100` functions return training and validation `torch.utils.data.Dataset` objects.
- **Parameters:**
  - `root` (str): The directory to download the dataset to.
  - `download` (bool, default=True): Whether to download the dataset if not found.
- **Example:**
  ```python
  from vitvqganvae.data.tv import get_cifar10, get_cifar100

  # CIFAR-10
  train_ds_10, val_ds_10 = get_cifar10(root="/path/to/data")

  # CIFAR-100
  train_ds_100, val_ds_100 = get_cifar100(root="/path/to/data")
  ```

---

## Dataset Utilities

### TVDataset Wrapper

- **File:** `vitvqganvae/data/tv/wrapper.py`
- **Description:** A wrapper for `torchvision` datasets to simplify accessing the image tensor. Many `torchvision` datasets return a tuple `(image, label)`. This wrapper ensures that only the image is returned by `__getitem__`.
- **Usage:** This is used internally in `main.py` when `dataset_source` is `torchvision`.
- **Parameters:**
  - `dataset` (Dataset): The `torchvision` dataset to wrap.
  - `key` (str or int): The key or index to access the image in the item returned by the original dataset. If `None`, it defaults to index `0`.
- **Example:**
  ```python
  from torchvision.datasets import CIFAR10
  from vitvqganvae.data.tv.wrapper import TVDataset

  cifar_dataset = CIFAR10(root="data", train=True, download=True)
  wrapped_dataset = TVDataset(cifar_dataset) # or TVDataset(cifar_dataset, key=0)

  image = wrapped_dataset[0] # returns only the image tensor
  ```

---

## Configuration

To use a dataset, you need to specify it in your `.yaml` configuration file. Here are the relevant keys:

- `dataset_name` (str): The name of the dataset getter function to use (e.g., `celeba`, `cifar10`, `ffhq`).
- `dataset_source` (str): The source of the dataset. Must be one of `torchvision`, `custom`, or `hf`.
- `dataset_img_key` (str or int, optional): For `torchvision` datasets, this specifies the key or index to access the image from the returned item. For most image datasets, this can be set to `null` or `0`.
- `dataset_kwargs` (dict): A dictionary of arguments to pass to the dataset getter function.

### Example Configuration

Here is an example of how to configure the `CelebA` dataset from `torchvision` in a `.yaml` file:

```yaml
dataset_name: celeba
dataset_source: torchvision
dataset_img_key: null
dataset_kwargs:
  root: ./.cache/celeba  # Path to download/cache the data
  download: True
  image_size: 64
```

For a custom dataset like `FFHQ`, the configuration would look like this:

```yaml
dataset_name: ffhq
dataset_source: custom
dataset_kwargs:
  root: /path/to/ffhq/images1024x1024
  image_size: 256
  split: [60000, 10000]
```

For a Hugging Face dataset like `mesh500_1024`, the configuration would be:

```yaml
dataset_name: mesh500_1024
dataset_source: hf
dataset_kwargs:
  root: ./.cache/mesh500_1024
  split: 0.8
  augment: True
```
