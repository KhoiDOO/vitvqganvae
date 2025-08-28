from torch import nn, einsum
from torch.autograd import grad as torch_grad

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from .encoder import Encoder
from .decoder import Decoder
from ...utils.model.layer_map import cnn_mapping
from ...utils.helpers import exists, default

from typing import Union
from beartype import beartype

import torch.nn.functional as F
import vector_quantize_pytorch
import torch

import copy
import os

@beartype
class VQVAE(nn.Module):
    def __init__(
        self, 
        dim: int,
        in_channel: int = 3,
        out_channel: int = 3,
        layers: int = 4,
        layer_mults: Union[list[int], None] = None,
        num_res_blocks: Union[int, tuple[int, ...]] = 1,
        group: int = 16,
        conv_type: str = "conv2d",
        enc_act_func: str = "LeakyReLU",
        dec_act_func: str = "GLU",
        enc_act_kwargs: dict = {"negative_slope": 0.1},
        dec_act_kwargs: dict = {"dim": 1},
        first_conv_kernel_size: int = 5,
        quantizer: str = "VectorQuantize",
        codebook_size: int = 512,
        quantizer_kwargs: dict = {
            "codebook_dim": 64,
            "decay" : 0.99,
            "commitment_weight": 0.25,
            "kmeans_init": True,
            "use_cosine_sim": True
        },
        l2_recon_loss: bool = False
    ):
        super().__init__()

        if conv_type not in cnn_mapping:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        if conv_type in ["conv1d", "conv3d"]:
            raise NotImplementedError(f"VQVAE only supports conv2d for now, got {conv_type}")

        self._dim = dim
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._layers = layers
        self._layer_mults = layer_mults
        self._num_res_blocks = num_res_blocks
        self._group = group
        self._conv_type = conv_type
        self._enc_act_func = enc_act_func
        self._dec_act_func = dec_act_func
        self._enc_act_kwargs = enc_act_kwargs
        self._dec_act_kwargs = dec_act_kwargs
        self._first_conv_kernel_size = first_conv_kernel_size
        self._dim_divisor = 2 ** layers

        self.encoder = Encoder(
			dim=self._dim,
			in_channel=self._in_channel,
			layers=self._layers,
			layer_mults=self._layer_mults,
			num_res_blocks=self._num_res_blocks,
			group=self._group,
			conv_type=self._conv_type,
			act_func=self._enc_act_func,
			act_kwargs=self._enc_act_kwargs,
			first_conv_kernel_size=self._first_conv_kernel_size
		)

        self.decoder = Decoder(
            dim=self._dim,
            out_channel=self._out_channel,
            layers=self._layers,
            layer_mults=self._layer_mults,
            num_res_blocks=self._num_res_blocks,
            group=self._group,
            conv_type=self._conv_type,
            act_func=self._dec_act_func,
            act_kwargs=self._dec_act_kwargs
        )

        self._quantizer = getattr(vector_quantize_pytorch, quantizer)(
            dim=self.encoder.encoded_dim,
            codebook_size=codebook_size,
            **quantizer_kwargs
        )
        self._codebook_size = codebook_size
        self._quantizer_kwargs = quantizer_kwargs

        self._l2_recon_loss = l2_recon_loss
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def encoded_dim(self) -> int:
        return self.encoder.encoded_dim

    def fmap_size(self, image_size) -> int:
        return self.encoder.fmap_size(image_size=image_size)

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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x, indices, vq_aux_loss = self.quantizer(x)
        return x, indices, vq_aux_loss

    def decode(self, fmap: torch.Tensor) -> torch.Tensor:
        # fmap: (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        batch, hw, c = fmap.shape
        h = w = int(hw ** 0.5)
        assert h * w == hw, f"Cannot reshape: {hw} is not a perfect square."
        fmap = rearrange(fmap, 'b (h w) c -> b h w c', h=h, w=w)
        fmap = rearrange(fmap, 'b h w c -> b c h w')
        return self.decoder(fmap)

    def decode_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        fmap = self.quantizer.get_output_from_indices(ids)
        return self.decode(fmap)

    def forward(
        self,
        img: torch.Tensor,
        return_loss: bool = False,
        return_recons: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        conv_type = getattr(self, '_conv_type', 'conv2d')
        device = img.device
        shape = img.shape
        batch = shape[0]
        channels = shape[1]
        if conv_type == 'conv1d' or conv_type == 'conv3d':
            raise NotImplementedError(f"VQVAE only supports conv2d for now, got {conv_type}")
        elif conv_type == 'conv2d':
            height, width = shape[2], shape[3]
            for dim_name, size in (('height', height), ('width', width)):
                assert (size % self._dim_divisor) == 0, f'{dim_name} must be divisible by {self._dim_divisor}'
            assert channels == self._in_channel, 'number of channels on image or sketch is not equal to the channels set on this VQVAE'
        else:
            raise ValueError(f'Unknown conv_type: {conv_type}')

        fmap, indices, commit_loss = self.encode(img)
        fmap = self.decode(fmap)
        if not return_loss:
            return fmap
    
        recon_loss = self.recon_loss_fn(fmap, img)

        loss = recon_loss + commit_loss

        if return_recons:
            return recon_loss, fmap
        
        return loss

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def in_channel(self) -> int:
        return self._in_channel

    @property
    def out_channel(self) -> int:
        return self._out_channel

    @property
    def layers(self) -> int:
        return self._layers

    @property
    def layer_mults(self) -> list[int]:
        return self._layer_mults

    @property
    def num_res_blocks(self) -> int | tuple[int]:
        return self._num_res_blocks

    @property
    def group(self) -> int:
        return self._group

    @property
    def conv_type(self) -> str:
        return self._conv_type

    @property
    def enc_act_func(self) -> str:
        return self._enc_act_func

    @property
    def dec_act_func(self) -> str:
        return self._dec_act_func

    @property
    def enc_act_kwargs(self) -> dict:
        return self._enc_act_kwargs

    @property
    def dec_act_kwargs(self) -> dict:
        return self._dec_act_kwargs
    
    @property
    def first_conv_kernel_size(self) -> int:
        return self._first_conv_kernel_size

    @property
    def quantizer(self) -> nn.Module:
        return self._quantizer

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def quantizer_kwargs(self) -> dict:
        return self._quantizer_kwargs