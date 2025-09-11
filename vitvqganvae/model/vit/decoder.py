from x_transformers.x_transformers import AttentionLayers

from einops.layers.torch import Rearrange

from torch import nn, Tensor
from beartype.typing import Union
from beartype import beartype

from ..utils import init_weights

import torch

@beartype
class ImgDecoder(nn.Module):
    def __init__(
        self,
        image_size: Union[tuple[int, int], int],
        patch_size: Union[tuple[int, int], int],
        out_channel: int = 3,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        **attn_kwargs
    ):
        super().__init__()

        self._image_size = image_size
        self._patch_size = patch_size
        self._out_channel = out_channel
        self._dim = dim
        self._depth = depth
        self._heads = heads

        self._image_height, self._image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self._patch_height, self._patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert self._image_height % self._patch_height == 0 and self._image_width % self._patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert 'causal' not in attn_kwargs, 'cannot set causality on ImgEncoder'

        self._num_patches = (self._image_height // self._patch_height) * (self._image_width // self._patch_width)
        self._patch_dim = self._out_channel * self._patch_height * self._patch_width

        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=self._image_height // self._patch_height),
            nn.ConvTranspose2d(self._dim, self._out_channel, kernel_size=self._patch_size, stride=self._patch_size)
        )
        self.de_pos_embedding = nn.Parameter(torch.randn(1, self._num_patches, self._dim))
        self.attn_layers = AttentionLayers(
            dim = self._dim,
            depth = self._depth,
            heads = self._heads,
            **attn_kwargs
        )

        self.apply(init_weights)
    
    def forward(self, token: Tensor) -> Tensor:
        x = token + self.de_pos_embedding
        x = self.attn_layers(x)
        x = self.to_pixel(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight

    @property
    def image_size(self) -> Union[tuple[int, int], int]:
        return self._image_size

    @property
    def patch_size(self) -> Union[tuple[int, int], int]:
        return self._patch_size

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
    def num_patches(self) -> int:
        return self._num_patches
    
    @property
    def patch_dim(self) -> int:
        return self._patch_dim
    
    def __str__(self):
        return f'ImgEncoder(image_size={self._image_size}, patch_size={self._patch_size}, out_channel={self._out_channel}, dim={self._dim}, depth={self._depth}, heads={self._heads})'