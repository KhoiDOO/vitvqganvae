import unittest
import torch
from vitvqganvae.model.cnn.decoder import Decoder
from vitvqganvae.utils.model.layer_map import cnn_mapping

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.dim=64
        self.out_channel=3
        self.layers=2
        self.image_size=32
        self.layer_mults=None
        self.num_res_blocks=1
        self.group=8
        self.act_func="GLU"
        self.act_kwargs={"dim": 1}
        self.conv_types=list(cnn_mapping.keys())

    def test_forward_output_shape(self):
        # Use conv2d for decoder and 4D input
        from vitvqganvae.model.cnn.decoder import Decoder
        dim=64
        out_channel=3
        layers=2
        layer_mults=None
        num_res_blocks=1
        group=8
        conv_type="conv2d"
        act_func="GLU"
        act_kwargs={"dim": 1}
        image_size=32
        decoder = Decoder(
            dim=dim,
            out_channel=out_channel,
            layers=layers,
            layer_mults=layer_mults,
            num_res_blocks=num_res_blocks,
            group=group,
            conv_type=conv_type,
            act_func=act_func,
            act_kwargs=act_kwargs
        )
        x = torch.randn(2, dim * (2**(layers-1)), image_size // (2 ** layers), image_size // (2 ** layers))
        print(x.shape)
        out = decoder(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], out_channel)
        self.assertEqual(out.shape[2], image_size)
        self.assertEqual(out.shape[3], image_size)

    def test_properties(self):
        from vitvqganvae.model.cnn.decoder import Decoder
        dim = 8
        out_channel = 3
        layers = 3
        layer_mults = [1, 2, 4]
        num_res_blocks = 1
        group = 2
        conv_type = "conv2d"
        act_func = "LeakyReLU"
        act_kwargs = {"negative_slope": 0.1}
        decoder = Decoder(
            dim=dim,
            out_channel=out_channel,
            layers=layers,
            layer_mults=layer_mults,
            num_res_blocks=num_res_blocks,
            group=group,
            conv_type=conv_type,
            act_func=act_func,
            act_kwargs=act_kwargs
        )
        self.assertEqual(decoder.dim, dim)
        self.assertEqual(decoder.out_channel, out_channel)
        self.assertEqual(decoder.layers, layers)
        self.assertEqual(decoder.group, group)
        self.assertEqual(decoder.conv_type, conv_type)
        self.assertEqual(decoder.act_func, act_func)
        self.assertEqual(decoder.act_kwargs, act_kwargs)
        self.assertEqual(decoder.layer_mults, layer_mults)
        self.assertEqual(decoder.num_res_blocks, num_res_blocks)

if __name__ == "__main__":
    unittest.main()
