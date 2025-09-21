from einops import repeat
from torch import nn

from .discriminator import Discriminator
from .vae import VAEConfig, VAE
from .vqvae import dataconv_consistency_check
from ..layers.gaussian import DiagonalGaussianDistribution
from ..utils import rebuild_save_load
from ...utils.helpers import default, exists
from pytorch_custom_utils import total_parameters

from ..loss.discriminator import (
    hinge_discr_loss,
    hinge_gen_loss,
    bce_discr_loss,
    bce_gen_loss,
    gradient_penalty,
    grad_layer_wrt_loss,
    safe_div
)

from beartype.typing import Union
from beartype import beartype
from dataclasses import dataclass, field

import torch.nn.functional as F
import torch

import timm


@dataclass
class LPIPSVAEGANConfig(VAEConfig):
    dim: int = 64
    in_channel: int = 3
    out_channel: int = 3
    layers: int = 4
    layer_mults: list[int] | None = None
    num_res_blocks: int = 1
    group: int = 16
    conv_type: str = "conv2d"
    enc_act_func: str = "LeakyReLU"
    dec_act_func: str = "GLU"
    enc_act_kwargs: dict = field(default_factory=lambda: {"negative_slope": 0.1})
    dec_act_kwargs: dict = field(default_factory=lambda: {"dim": 1})
    first_conv_kernel_size: int = 5
    l2_recon_loss: bool = True
    use_variational: bool = True
    discr_layers: int = 3
    discr_dims: list[int] | None = None
    discr_groups: int | None = None
    discr_init_kernel_size: int | None = None
    discr_act_func: str | None = None
    discr_act_kwargs: dict | None = None
    use_perceptual_loss: bool = False
    use_gan_loss: bool = True
    use_hinge_loss: bool | None = None
    perceptual_model: str | None = None


@rebuild_save_load()
@total_parameters()
@beartype
class LPIPSVAEGAN(VAE):
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
        l2_recon_loss: bool = True,
        use_variational: bool = True,
        discr_layers: int = 3,
        discr_dims: list[int] | None = None,
        discr_groups: int | None = None,
        discr_init_kernel_size: int | None = None,
        discr_act_func: str | None = None,
        discr_act_kwargs: dict | None = None,
        use_perceptual_loss: bool = False,
        use_gan_loss: bool = False,
        use_hinge_loss: bool | None = None,
        perceptual_model: str | None = None,
    ):
        super().__init__(
            dim=dim,
            in_channel=in_channel,
            out_channel=out_channel,
            layers=layers,
            layer_mults=layer_mults,
            num_res_blocks=num_res_blocks,
            group=group,
            conv_type=conv_type,
            enc_act_func=enc_act_func,
            dec_act_func=dec_act_func,
            enc_act_kwargs=enc_act_kwargs,
            dec_act_kwargs=dec_act_kwargs,
            first_conv_kernel_size=first_conv_kernel_size,
            l2_recon_loss=l2_recon_loss,
            use_variational=use_variational
        )

        self._discr_layers = discr_layers
        self._discr_dims = discr_dims
        self._discr_groups = discr_groups
        self._discr_init_kernel_size = discr_init_kernel_size
        self._discr_act_func = discr_act_func
        self._discr_act_kwargs = discr_act_kwargs
        self._use_perceptual_loss = use_perceptual_loss
        self._use_gan_loss = use_gan_loss
        self._use_hinge_loss = use_hinge_loss
        self._perceptual_model = perceptual_model

        layer_mults = list(map(lambda t: 2 ** t, range(self._discr_layers)))
        layer_dims = [self._dim * mult for mult in layer_mults]
        dims = (self._dim, *layer_dims)

        if self._use_gan_loss:
            self._discr = Discriminator(
                dims = dims, 
                channels = self._out_channel, 
                groups = default(discr_groups, self._group),
                init_kernel_size = default(discr_init_kernel_size, self._first_conv_kernel_size), 
                act_func = default(discr_act_func, self._enc_act_func), 
                act_kwargs = default(discr_act_kwargs, self._enc_act_kwargs)
            )

            self._discr_loss = hinge_discr_loss if self._use_hinge_loss else bce_discr_loss
            self._gen_loss = hinge_gen_loss if self._use_hinge_loss else bce_gen_loss
        else:
            self._discr = None
            self._discr_loss = None
            self._gen_loss = None
        
        if self._use_perceptual_loss:
            self.lpips_model = timm.create_model(
                self._perceptual_model if self._perceptual_model else 'vgg16', 
                pretrained=True, 
                cache_dir='./.cache', 
                num_classes=0
            ).eval()
    
    @property
    def discr_layers(self) -> int:
        return self._discr_layers

    @property
    def discr_dims(self) -> list[int] | None:
        return self._discr_dims

    @property
    def discr_groups(self) -> int | None:
        return self._discr_groups
    
    @property
    def discr_init_kernel_size(self) -> int | None:
        return self._discr_init_kernel_size
    
    @property
    def discr_act_func(self) -> str | None:
        return self._discr_act_func

    @property
    def discr_act_kwargs(self) -> dict | None:
        return self._discr_act_kwargs

    @property
    def use_perceptual_loss(self) -> bool:
        return self._use_perceptual_loss
    
    @property
    def use_gan_loss(self) -> bool:
        return self._use_gan_loss
    
    @property
    def use_hinge_loss(self) -> bool | None:
        return self._use_hinge_loss
    
    @property
    def perceptual_model(self) -> str | None:
        return self._perceptual_model

    def forward(
        self,
        img: torch.Tensor,
        return_loss: bool = False,
        return_recons: bool = False
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        conv_type = getattr(self, '_conv_type', 'conv2d')
        dataconv_consistency_check(
            img, 
            conv_type=conv_type,
            dim_divisor=self._dim_divisor,
            in_channel=self._in_channel
        )

        posterior: DiagonalGaussianDistribution = self.encode(img)
        z = posterior.sample()
        fmap = self.decode(z)

        if not return_loss:
            return fmap
        
        recon_loss = self.recon_loss_fn(fmap, img)
        
        if return_recons:
            return recon_loss, fmap
        
        if self._use_variational:
            kl_loss = posterior.kl().mean()
        
        if not self._use_perceptual_loss:
            return {"recon_loss": recon_loss, "kl_loss": kl_loss}
        
        img_lpips_input = img
        fmap_lpips_input = fmap

        if img.shape[1] == 1 and img.ndim == 4:
            img_lpips_input, fmap_lpips_input = map(
                lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), 
                (img_lpips_input, fmap_lpips_input)
            )

        img_lpips_feats = self.lpips_model(img_lpips_input)
        recon_lpips_feats = self.lpips_model(fmap_lpips_input)
        perceptual_loss = F.mse_loss(img_lpips_feats, recon_lpips_feats)
        
        if exists(self._discr) and self._use_gan_loss:
            gen_loss = self._gen_loss(self._discr(fmap))

            last_dec_layer = self.decoder[-1].weight

            norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

            adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
            adaptive_weight.clamp_(max=1e4)

            return {
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "perceptual_loss": perceptual_loss,
                "gen_loss": adaptive_weight * gen_loss,
            }
        
        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "perceptual_loss": perceptual_loss
        }

    def forward_discr(
        self,
        img: torch.Tensor,
        add_gradient_penalty: bool = True
    ) -> dict[str, torch.Tensor]:
        
        conv_type = getattr(self, '_conv_type', 'conv2d')
        dataconv_consistency_check(
            img, 
            conv_type=conv_type,
            dim_divisor=self._dim_divisor,
            in_channel=self._in_channel
        )

        posterior: DiagonalGaussianDistribution = self.encode(img)
        z = posterior.sample()
        fmap: torch.Tensor = self.decode(z)

        fmap.detach_()
        img.requires_grad_()

        fmap_discr_logits, img_discr_logits = map(self._discr, (fmap, img))

        discr_loss = self._discr_loss(fmap_discr_logits, img_discr_logits)

        loss = discr_loss
        if add_gradient_penalty:
            gp = gradient_penalty(img, img_discr_logits)
            loss = discr_loss + gp

        return {"discr_loss": loss}