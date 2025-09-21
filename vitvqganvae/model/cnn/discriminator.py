from torch import nn
from torch.nn import ModuleList
from ...utils.helpers import default


class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 16,
        init_kernel_size = 5,
        act_func: str = "LeakyReLU",
        act_kwargs: dict = {"negative_slope": 0.1},
    ):
        super().__init__()

        self._dims = dims
        self._channels = channels
        self._groups = groups
        self._init_kernel_size = init_kernel_size
        self._act_func = act_func
        self._act_kwargs = act_kwargs

        dim_pairs = zip(self._dims[:-1], self._dims[1:])

        self.layers = ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self._channels, self._dims[0], self._init_kernel_size, padding = self._init_kernel_size // 2), 
                    getattr(nn, self._act_func)(
                        **default(self._act_kwargs, {})
                    )
                )
            ]
        )

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                nn.GroupNorm(self._groups, dim_out),
                getattr(nn, self._act_func)(
                    **default(self._act_kwargs, {})
                )
            ))

        dim = self._dims[-1]
        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(dim, dim, 1),
            getattr(nn, self._act_func)(
                **default(self._act_kwargs, {})
            ),
            nn.Conv2d(dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)