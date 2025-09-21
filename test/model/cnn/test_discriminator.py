import unittest
import torch

from vitvqganvae.model.cnn.discriminator import Discriminator


class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        # choose dims length 3 so final spatial size -> 5x5 output for 32x32 input
        self.dims = [64, 128, 256]
        self.channels = 3
        self.groups = 16
        self.init_kernel_size = 5
        self.act_func = "LeakyReLU"
        self.act_kwargs = {"negative_slope": 0.1}

        self.net = Discriminator(
            dims=self.dims,
            channels=self.channels,
            groups=self.groups,
            init_kernel_size=self.init_kernel_size,
            act_func=self.act_func,
            act_kwargs=self.act_kwargs,
        )

    def test_properties(self):
        self.assertEqual(self.net._dims, self.dims)
        self.assertEqual(self.net._channels, self.channels)
        self.assertEqual(self.net._groups, self.groups)
        self.assertEqual(self.net._init_kernel_size, self.init_kernel_size)
        self.assertEqual(self.net._act_func, self.act_func)
        self.assertEqual(self.net._act_kwargs, self.act_kwargs)

    def test_num_layers(self):
        # initial layer + one per dim pair -> total == len(dims)
        self.assertEqual(len(self.net.layers), len(self.dims))

    def test_forward_shape(self):
        batch = 2
        H = W = 32
        x = torch.randn(batch, self.channels, H, W)
        out = self.net(x)
        # compute expected final spatial size
        num_downsamples = len(self.dims) - 1
        final_spatial = H // (2 ** num_downsamples)
        # last conv in to_logits has kernel size 4 and no padding, so out = final_spatial - 4 + 1
        expected_spatial = final_spatial - 4 + 1
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], batch)
        self.assertEqual(out.shape[1], 1)
        self.assertEqual(out.shape[2], expected_spatial)
        self.assertEqual(out.shape[3], expected_spatial)

    def test_backward_flow(self):
        batch = 1
        H = W = 32
        x = torch.randn(batch, self.channels, H, W, requires_grad=False)
        out = self.net(x)
        loss = out.mean()
        loss.backward()

        # make sure some parameter has grad
        grads = [p.grad for p in self.net.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)

    def test_alternative_activation(self):
        # test with ReLU activation to ensure getattr path works
        net_relu = Discriminator(dims=self.dims, channels=self.channels, act_func="ReLU", act_kwargs={})
        x = torch.randn(1, self.channels, 32, 32)
        out = net_relu(x)
        self.assertIsInstance(out, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
