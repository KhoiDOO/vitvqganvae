from torch import nn
from einops import rearrange

from ..encoder import Encoder
from ..decoder import Decoder
from ...layers.layer_map import cnn_mapping, cnn_2_ndim, rearrange_map
from ...utils import rebuild_save_load
from ....utils.helpers import count_parameters
from pytorch_custom_utils import total_parameters

from typing import Union
from beartype import beartype
from dataclasses import dataclass, field

import torch.nn.functional as F
import vector_quantize_pytorch
import torch

import copy
import os


@dataclass
class FlexVQVAEConfig:
    dim: int = 64
    in_channel: int = 3
    out_channel: int = 3
    layers: int = 4
    layer_mults: list[int] | None = None
    num_in_res_blocks: int = 1
    num_out_res_blocks: int = 1
    group: int = 16
    conv_type: str = "conv2d"
    enc_act_func: str = "LeakyReLU"
    dec_act_func: str = "GLU"
    enc_act_kwargs: dict = field(default_factory=lambda: {"negative_slope": 0.1})
    dec_act_kwargs: dict = field(default_factory=lambda: {"dim": 1})
    first_conv_kernel_size: int = 5
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
class FlexVQVAE(nn.Module):
    def __init__(
        self, 
        dim: int,
        in_channel: int = 3,
        out_channel: int = 3,
        layers: int = 4,
        layer_mults: Union[list[int], None] = None,
        num_in_res_blocks: Union[int, tuple[int, ...]] = 1,
        num_out_res_blocks: Union[int, tuple[int, ...]] = 1,
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
        l2_recon_loss: bool = True
    ):
        super().__init__()

        if conv_type not in cnn_mapping:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self._dim = dim
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._layers = layers
        self._layer_mults = layer_mults
        self._num_in_res_blocks = num_in_res_blocks
        self._num_out_res_blocks = num_out_res_blocks
        self._group = group
        self._conv_type = conv_type
        self._enc_act_func = enc_act_func
        self._dec_act_func = dec_act_func
        self._enc_act_kwargs = enc_act_kwargs
        self._dec_act_kwargs = dec_act_kwargs
        self._first_conv_kernel_size = first_conv_kernel_size
        self._dim_divisor = 2 ** layers
        self._ndim = cnn_2_ndim[conv_type]

        self.encoder = Encoder(
			dim=self._dim,
			in_channel=self._in_channel,
			layers=self._layers,
			layer_mults=self._layer_mults,
			num_res_blocks=self._num_in_res_blocks,
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
            num_res_blocks=self._num_out_res_blocks,
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
        x = rearrange(x, rearrange_map[x.ndim])
        x, indices, vq_aux_loss = self._quantizer(x)
        return x, indices, vq_aux_loss

    def decode(self, fmap: torch.Tensor) -> torch.Tensor:
        
        if self._ndim == 3:
            fmap = rearrange(fmap, 'b h c -> b c h')
            return self.decoder(fmap)
        elif self._ndim == 4:
            batch, hw, c = fmap.shape
            h = w = int(hw ** 0.5)
            assert h * w == hw, f"Cannot reshape: {hw} is not a perfect square."
            fmap = rearrange(fmap, 'b (h w) c -> b c h w', h=h, w=w)
            return self.decoder(fmap)
        elif self._ndim == 5:
            batch, dhw, c = fmap.shape
            d = h = w = round(dhw ** (1/3))
            assert d * h * w == dhw, f"Cannot reshape: {dhw} is not a perfect cube."
            fmap = rearrange(fmap, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
            return self.decoder(fmap)
        else:
            raise ValueError(f"target_ndim must be 3, 4, or 5, got {self._ndim}")

    def decode_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        fmap = self._quantizer.get_output_from_indices(ids)
        return self.decode(fmap)

    def forward(
        self,
        img: torch.Tensor,
        return_loss: bool = False,
        return_recons: bool = False
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        conv_type = getattr(self, '_conv_type', 'conv2d')
        device = img.device
        shape = img.shape
        batch = shape[0]
        channels = shape[1]

        if conv_type == 'conv1d':
            length = shape[2]
            assert (length % self._dim_divisor) == 0, f'length must be divisible by {self._dim_divisor}'
            assert channels == self._in_channel, 'number of channels on audio is not equal to the channels set on this VQVAE'
        elif conv_type == 'conv2d':
            height, width = shape[2], shape[3]
            for dim_name, size in (('height', height), ('width', width)):
                assert (size % self._dim_divisor) == 0, f'{dim_name} must be divisible by {self._dim_divisor}'
            assert channels == self._in_channel, 'number of channels on image or sketch is not equal to the channels set on this VQVAE'
        elif conv_type == 'conv3d':
            depth, height, width = shape[2], shape[3], shape[4]
            for dim_name, size in (('depth', depth), ('height', height), ('width', width)):
                assert (size % self._dim_divisor) == 0, f'{dim_name} must be divisible by {self._dim_divisor}'
            assert channels == self._in_channel, 'number of channels on video is not equal to the channels set on this VQVAE'
        else:
            raise ValueError(f'Unknown conv_type: {conv_type}')

        fmap, indices, commit_loss = self.encode(img)
        fmap = self.decode(fmap)
        if not return_loss:
            return fmap
    
        recon_loss = self.recon_loss_fn(fmap, img)

        if return_recons:
            return recon_loss, fmap

        return {"recon_loss": recon_loss, "quantizer_loss": commit_loss.mean()}

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
    def num_in_res_blocks(self) -> int | tuple[int]:
        return self._num_in_res_blocks

    @property
    def num_out_res_blocks(self) -> int | tuple[int]:
        return self._num_out_res_blocks

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

    @property
    def ndim(self) -> int:
        return self._ndim