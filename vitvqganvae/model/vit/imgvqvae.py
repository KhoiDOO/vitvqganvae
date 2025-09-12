from torch import nn
from einops import rearrange

from .encoder import ImgVITEncoder
from .decoder import ImgVITDecoder
from ..utils import rebuild_save_load
from ...utils.helpers import count_parameters
from pytorch_custom_utils import total_parameters

from beartype.typing import Union
from beartype import beartype
from dataclasses import dataclass, field

import torch.nn.functional as F
import vector_quantize_pytorch
import torch

import copy
import os


@dataclass
class ImgVITVQVAEConfig:
    image_size: int = 256
    patch_size: int = 8
    in_channel: int = 3
    out_channel: int = 3
    dim: int = 512
    depth: int = 6
    heads: int = 8
    encoder_attn_kwargs: dict = field(default_factory=dict)
    decoder_attn_kwargs: dict = field(default_factory=dict)
    quantizer: str = "VectorQuantize"
    codebook_size: int = 512
    quantizer_kwargs: dict = field(default_factory=lambda: {
        "codebook_dim": 64,
        "decay" : 0.99,
        "commitment_weight": 0.25,
        "kmeans_init": True,
        "use_cosine_sim": True
    })
    l2_recon_loss: bool = True


@rebuild_save_load()
@total_parameters()
@beartype
class ImgVITVQVAE(nn.Module):
    def __init__(
        self,
        image_size: Union[tuple[int, int], int] = 256,
        patch_size: Union[tuple[int, int], int] = 8,
        in_channel: int = 3,
        out_channel: int = 3,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        encoder_attn_kwargs: dict = None,
        decoder_attn_kwargs: dict = None,
        quantizer: str = "VectorQuantize",
        codebook_size: int = 512,
        quantizer_kwargs: dict = None,
        l2_recon_loss: bool = True
    ):
        super().__init__()

        self._image_size = image_size
        self._patch_size = patch_size
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._dim = dim
        self._depth = depth
        self._heads = heads
        self._encoder_attn_kwargs = encoder_attn_kwargs if encoder_attn_kwargs is not None else {}
        self._decoder_attn_kwargs = decoder_attn_kwargs if decoder_attn_kwargs is not None else {}
        self._fmap = (self._image_size // self._patch_size) ** 2

        if quantizer_kwargs is None:
            quantizer_kwargs = {
                "codebook_dim": 64,
                "decay" : 0.99,
                "commitment_weight": 0.25,
                "kmeans_init": True,
                "use_cosine_sim": True
            }

        self.encoder = ImgVITEncoder(
            image_size = self._image_size,
            patch_size=self._patch_size,
            in_channel=self._in_channel,
            dim=self._dim,
            depth=self._depth,
            heads=self._heads,
            **self._encoder_attn_kwargs
        )

        self.decoder = ImgVITDecoder(
            image_size=self._image_size,
            patch_size=self._patch_size,
            out_channel=self._out_channel,
            dim=self._dim,
            depth=self._depth,
            heads=self._heads,
            **self._decoder_attn_kwargs
        )

        self._quantizer = getattr(vector_quantize_pytorch, quantizer)(
            dim=self._dim,
            codebook_size=codebook_size,
            **quantizer_kwargs
        )
        self._codebook_size = codebook_size
        self._quantizer_kwargs = quantizer_kwargs

        self._l2_recon_loss = l2_recon_loss
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        print(f"Total parameters in VQVAE: {self.total_parameters}")
        print(f"Total trainable parameters in VQVAE: {count_parameters(self, True)}")  
        print(f"Total encoder parameters in VQVAE: {count_parameters(self.encoder)}")
        print(f"Total encoder trainable parameters in VQVAE: {count_parameters(self.encoder, True)}")
        print(f"Total decoder parameters in VQVAE: {count_parameters(self.decoder)}")
        print(f"Total decoder trainable parameters in VQVAE: {count_parameters(self.decoder, True)}")
        print(f"Total quantizer parameters in VQVAE: {count_parameters(self._quantizer)}")
        print(f"Total quantizer trainable parameters in VQVAE: {count_parameters(self._quantizer, True)}")
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    def fmap(self) -> int:
        return self._fmap

    def copy_for_eval(self) -> nn.Module:
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        vae_copy.eval()
        return vae_copy.to(device)

    def state_dict(self, *args, **kwargs) -> dict:
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        self.load_state_dict(torch.load(path, weights_only=False))
    
    def encode(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = img.shape
        assert h == self._image_size and w == self._image_size, f"Input image size must be {self._image_size}x{self._image_size}, but got {h}x{w}"
        assert c == self._in_channel, f"Input image channel must be {self._in_channel}, but got {c}"

        x = self.encoder(img)
        x, indices, vq_aux_loss = self._quantizer(x)
        return x, indices, vq_aux_loss
    
    def decode(self, fmap: torch.Tensor) -> torch.Tensor:
        b, hw, c = fmap.shape
        assert c == self._dim, f"Input feature map channel must be {self._dim}, but got {c}"
        assert hw == self._fmap, f"Input feature map size must be {self._fmap}, but got {hw}"

        img_recon = self.decoder(fmap)
        return img_recon
    
    def decode_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        fmap = self._quantizer.get_output_from_indices(ids)
        return self.decode(fmap)
    
    def forward(
        self,
        img: torch.Tensor,
        return_loss: bool = False,
        return_recons: bool = False
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        b, c, h, w = img.shape
        assert h == self._image_size and w == self._image_size, f"Input image size must be {self._image_size}x{self._image_size}, but got {h}x{w}"
        assert c == self._in_channel, f"Input image channel must be {self._in_channel}, but got {c}"

        fmap, indices, commit_loss = self.encode(img)
        fmap = self.decode(fmap)
        if not return_loss:
            return fmap
        
        recon_loss = self.recon_loss_fn(fmap, img)
        
        if return_recons:
            return recon_loss, fmap
        
        return {"recon_loss": recon_loss, "quantizer_loss": commit_loss.mean()}
    
    @property
    def quantizer(self) -> nn.Module:
        return self._quantizer
    
    @property
    def codebook_size(self) -> int:
        return self._codebook_size
    
    @property
    def quantizer_kwargs(self) -> dict:
        return self._quantizer_kwargs
    
    @property
    def l2_recon_loss(self) -> bool:
        return self._l2_recon_loss
    
    @property
    def image_size(self) -> Union[tuple[int, int], int]:
        return self._image_size
    
    @property
    def patch_size(self) -> Union[tuple[int, int], int]:
        return self._patch_size
    
    @property
    def in_channel(self) -> int:
        return self._in_channel 
    
    @property
    def out_channel(self) -> int:
        return self._out_channel
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def depth(self) -> int:
        return self._depth
    
    @property
    def heads(self) -> int:
        return self._heads  
    
    @property
    def encoder_attn_kwargs(self) -> dict:
        return self._encoder_attn_kwargs    
    
    @property
    def decoder_attn_kwargs(self) -> dict:
        return self._decoder_attn_kwargs