from .block import DecoderBlock
from ..layers.layer_map import cnn_mapping, cnn_transpose_mapping
from ...utils.helpers import default

from torch import nn, Tensor

from beartype.typing import Union
from beartype import beartype

@beartype
class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        out_channel: int = 3,
        layers: int = 4,
        layer_mults: Union[list[int], None] = None,
        num_res_blocks: Union[int, tuple[int, ...]] = 1,
        group: int = 16,
        conv_type: str = "conv2d",
        act_func: str = "GLU",
        act_kwargs: dict = {"dim": 1}
    ):
        super().__init__()
        
        self._dim = dim
        self._out_channel = out_channel
        self._layers = layers
        self._group = group
        self._conv_type = conv_type
        self._act_func = act_func
        self._act_kwargs = act_kwargs
        self._num_res_blocks = num_res_blocks

        assert dim % group == 0, f'dimension {dim} must be divisible by {group} (groups for the groupnorm)'

        self.blocks = nn.ModuleList()

        if layer_mults:
            assert len(layer_mults) == layers, 'length of layer_mults must be equal to number of layers'

        self._layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(layers))))

        layer_dims = [dim * mult for mult in self._layer_mults]
        dims = (dim, *layer_dims)
        dim_pairs = zip(dims[:-1], dims[1:])

        if not isinstance(num_res_blocks, (list, tuple)):
            num_res_blocks = (*((0,) * (layers - 1)), num_res_blocks)

        assert len(num_res_blocks) == layers, 'number of resnet blocks config must be equal to number of layers'

        for layer_index, (dim_in, dim_out), layer_num_res_blocks in zip(range(layers), dim_pairs, num_res_blocks):
            self.blocks.insert(
                0,
                nn.Sequential(
                    cnn_transpose_mapping[self._conv_type](
                        in_channels=dim_out,
                        out_channels=dim_in,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.LeakyReLU(0.1)
                )
            )

            for _ in range(layer_num_res_blocks):
                self.blocks.insert(
                    0,
                    DecoderBlock(
                        dim=dim_out,
                        group=self._group,
                        conv_type=self._conv_type,
                        act_func=self._act_func,
                        act_kwargs=self._act_kwargs
                    )
                )

        self.blocks.append(cnn_mapping[self._conv_type](
            self._dim,
            self._out_channel,
            kernel_size=1
            )
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def out_channel(self) -> int:
        return self._out_channel

    @property
    def layers(self) -> int:
        return self._layers

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

    @property
    def layer_mults(self) -> list[int]:
        return self._layer_mults

    @property
    def num_res_blocks(self) -> int | tuple[int, ...]:
        return self._num_res_blocks

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x