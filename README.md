
<p align="center">
    <img src="https://img.shields.io/github/license/KhoiDOO/vitvqganvae?style=flat-square" alt="License">
    <img src="https://img.shields.io/github/stars/KhoiDOO/vitvqganvae?style=flat-square" alt="Stars">
    <img src="https://img.shields.io/github/issues/KhoiDOO/vitvqganvae?style=flat-square" alt="Issues">
    <img src="https://img.shields.io/pypi/pyversions/torch?style=flat-square" alt="Python">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
</p>

<h1 align="center">VitVqGanVae <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg" width="32"/> <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" width="32"/></h1>

<p align="center">
<b>Benchmark for Evaluating Data Reconstruction using Vector Quantization</b>
</p>


## 🚀 Setup


<details>
<summary>🛠️ Environment Information</summary>

- Python: 3.10.0
- CUDA: 12.9
- torch==2.8.0
- tqdm==4.67.1
- beartype==0.21.0
- omegaconf==2.3.0
- pillow==11.0.0
- opencv-python==4.12.0.88
- scikit-image==0.25.2
- albumentationsx==2.0.10
- scikit-learn==1.7.1
- wandb==0.21.1
- tensorboard==2.20.0
- datasets==4.0.0
- einops==0.8.1
- ema-pytorch==0.7.7
- pytorch-warmup==0.2.0
- pytorch-custom-utils==0.0.21
- memory-efficient-attention-pytorch==0.1.6
- sentencepiece==0.2.1
- transformers==4.55.3
- vector-quantize-pytorch==1.23.1
- accelerate==1.10.0
- torchinfo==1.8.0
- gdown==5.2.0
- onnx==1.19.0
- onnxruntime==1.22.1
- diffusers==0.35.1
- ninja==1.13.0

</details>

### 🐍 Python
```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip wheel
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install pytest # for development only
pip install .
```

### 🐾 Conda
For easier reproducibility, using conda is a recommended way to setup the environment.
```bash
bash setup.sh vitvqganvae
conda activate vitvqganvae
```

### 🐳 Docker
We provide docker image file in folder <code>docker</code> and services in <code>docker-compose.yaml</code>. All images are based on [kohido/base_dl_cuda129:v0.0.6](https://hub.docker.com/r/kohido/base_dl_cuda129). In this image all required packages are built. 

### ☸️ Kubernetes
We provide [Kubernetes](https://github.com/kubernetes/kubernetes) <code>*.yaml</code> files in folder <code>deploy</code>. The image is also built upon the image [kohido/base_dl_cuda129:v0.0.6](https://hub.docker.com/r/kohido/base_dl_cuda129). For multi-gpu/multi-node training with Kubernetes, we provide corresponding <code>*.yaml</code> in folder <code>deploy/mg</code>.


### 📊 Logging with Wandb
This project uses [wandb](https://github.com/wandb/wandb) for logging metrics. You will need to log into wandb using their token. Perform <code>wandb login</code> then provide the token to login wandb.

For Docker user, you will need to create a file ```.env```, so that the ```docker-compose``` will read the ```wandb``` token from that file to login ```wandb``` inside the container. A template for ```.env``` is as follows:

```
WANDB_PROJECT_NAME=<your_project_name>
WANDB_ENTITY=<your_project_entity>
WANDB_API_KEY=<your_wandb_api_key>
```

For Docker user, you will need to create a file ```.env```, which has the same template as above and add that file to a secret and then modift the secret name in ```.yaml``` file. An example is shown as follows:
```yaml
env:
- name: GITHUB_TOKEN
    valueFrom:
    secretKeyRef:
        name: <your-secret-name>
        key: GITHUB_TOKEN
- name: WANDB_API_KEY
    valueFrom:
    secretKeyRef:
        name: <your-secret-name>
        key: WANDB_API_KEY
```

## 🎬 Results & Demo

We provide checkpoints and demo for using checkpoint via GoogleColab notebook. More checkpoints with different datasets can be found [here](https://huggingface.co/kohido).

### VQVAE Result on CelebA
| 📚 <b>Dataset</b> | 📏 <b>Size</b> | 📉 <b>Loss</b> | 🧠 <b>Method</b> | 🏆 <b>Checkpoint</b> | 🚀 <b>Demo</b> |
|:---:|:---:|:----------------------:|:------:|:-----------:|:----:|
| CelebA | 64 x 64 | 0.00014 | 🟩 VQ| <a href="https://huggingface.co/kohido/celeba_vqvae">🤗 HuggingFace</a> | |
| CelebA | 64 x 64 | 0.00024 | 🟩 RVQ| <a href="https://huggingface.co/kohido/celeba_vqvae_rvq">🤗 HuggingFace</a> | <a href="https://colab.research.google.com/drive/138j09tvXXMVN6sHfWwKlNo3Z8TAMg6hl?usp=sharing">📓 Colab</a> |
| CelebA | 64 x 64 | 0.00009 | 🟩 GRVQ| <a href="https://huggingface.co/kohido/celeba_vqvae_grvq">🤗 HuggingFace</a> | <a href="https://colab.research.google.com/drive/1InSAa_8FBvw5VLKhuo_yVBIkiGKpjJxc?usp=sharing">📓 Colab |

## 🏋️‍♂️ Train
For data preparation, please refer to [DATA.md](docs/md/DATA.md)

### 🐍 Python Environment & Conda
This project is built upon [Accelerate](https://github.com/huggingface/accelerate) to easily perform single, multi-gpus, and multi-nodes training. To train a Residual Vector Quantization (RVQ) VAE on the CelebA dataset, you can use the following command

```bash
accelerate launch \
    --mixed_precision=no \
    --num_processes=1 \
    --num_machines=1 \
    --dynamo_backend=no \
    main.py \
    --config config/celeba_vqvae_rvq.yaml \
    --train \
    trainer_kwargs.use_wandb_tracking=True
```

### 🐳 Docker
To train using Docker, use <code>docker-compose</code> to run the provided services in <code>docker-compose.yaml</code>
```bash
docker compose up <service-name>
```
In case, some non-related errors happens, after fixing that you might consider rebuild or recreate images.
```bash
docker compose up <service-name> --build --force-create
```
To train a Residual Vector Quantization (RVQ) VAE on the CelebA dataset, you can use the following command
```bash
docker compose up vitvqganvae-train-celeba128-rvq --build --force-create
```

### ☸️ Kubernetes
To train using Kubernetes, after setting up the environment, use the following command to run a pod.
```bash
cd deploy
kubectl apply -f <your_yaml_file>
```
To train a Residual Vector Quantization (RVQ) VAE on the CelebA dataset, you can use the following command
```bash
cd deploy
kubectl apply -f celeba_vqvae_rvq.yaml
```

## 🧪 Testing

To perform testing all core functions of ```vitvqganvae```, please refer to folder ```test```, which contains ```*.py``` test files. To easily test them all, perform the following command:
```bash
bash test.sh
```

## 🙏 Acknowledgement
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [x-transformers](https://github.com/lucidrains/x-transformers)
