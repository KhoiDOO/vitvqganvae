from huggingface_hub import PyTorchModelHubMixin
from ..cnn.vqvae import VQVAE
from ..cnn.variant import FlexVQVAE

from pathlib import Path

import torch

class VQVAEPipeline(
    VQVAE, 
    PyTorchModelHubMixin,
    repo_url="https://github.com/KhoiDOO/vitvqganvae",
    docs_url="https://github.com/KhoiDOO/vitvqganvae",
    pipeline_tag="image-to-image",
    license="mit",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load(self, path: str):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cpu", weights_only=False)

        self.load_state_dict(pkg['model'])

class FlexVQVAEPipeline(
    FlexVQVAE, 
    PyTorchModelHubMixin,
    repo_url="https://github.com/KhoiDOO/vitvqganvae",
    docs_url="https://github.com/KhoiDOO/vitvqganvae",
    pipeline_tag="image-to-image",
    license="mit",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, path: str):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cpu", weights_only=False)

        self.load_state_dict(pkg['model'])
