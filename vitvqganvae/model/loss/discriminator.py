from beartype import beartype
from einops import rearrange

from torch import Tensor
from torch.autograd import grad as torch_grad

import torch
import torch.nn.functional as F

@beartype
def log(t: Tensor, eps: float = 1e-10) -> Tensor:
    return torch.log(t + eps)

@beartype
def hinge_discr_loss(fake: Tensor, real: Tensor) -> Tensor:
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

@beartype
def hinge_gen_loss(fake: Tensor) -> Tensor:
    return -fake.mean()

@beartype
def bce_discr_loss(fake: Tensor, real: Tensor) -> Tensor:
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

@beartype
def bce_gen_loss(fake: Tensor) -> Tensor:
    return -log(torch.sigmoid(fake)).mean()

@beartype
def gradient_penalty(images: Tensor, output: Tensor, weight: float = 10) -> Tensor:
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def grad_layer_wrt_loss(loss: Tensor, layer: Tensor) -> Tensor:
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

def safe_div(numer: Tensor, denom: Tensor, eps: float = 1e-8) -> Tensor:
    return numer / denom.clamp(min=eps)