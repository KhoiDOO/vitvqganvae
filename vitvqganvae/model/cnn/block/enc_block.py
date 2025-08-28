from ....utils.model.layer_map import cnn_mapping

from torch import nn, Tensor

from beartype import beartype


@beartype
class EncoderBlock(nn.Module):
    def __init__(
            self, 
            dim: int, 
            group: int, 
            conv_type: str = "conv2d",
            act_func: str = "LeakyReLU",
            act_kwargs: dict = {"negative_slope": 0.1}
        ) -> None:
        super().__init__()

        self._dim = dim
        self._group = group
        self._conv_type = conv_type
        self._act_func = act_func
        self._act_kwargs = act_kwargs if act_kwargs is not None else {}

        self.net = nn.Sequential(
            cnn_mapping[conv_type](dim, dim, 3, padding = 1),
            nn.GroupNorm(group, dim),
            getattr(nn, act_func)(**self._act_kwargs),
            cnn_mapping[conv_type](dim, dim, 3, padding = 1),
            nn.GroupNorm(group, dim),
            getattr(nn, act_func)(**self._act_kwargs),
            cnn_mapping[conv_type](dim, dim, 1)
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def group(self) -> int:
        return self._group

    @property
    def conv_type(self) -> str:
        return self._conv_type

    @property
    def act_func(self) -> str:
        return self._act_func

    @property
    def act_kwargs(self) -> dict:
        return self._act_kwargs

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x