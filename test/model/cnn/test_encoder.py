import unittest
import torch
from vitvqganvae.model.cnn.encoder import Encoder
from torch import Tensor

class TestEncoder(unittest.TestCase):
	def setUp(self):
		self.dim = 8
		self.in_channel = 3
		self.layers = 3
		self.layer_mults = [1, 2, 4]
		self.num_res_blocks = 1
		self.group = 2
		self.conv_type = "conv2d"
		self.act_func = "LeakyReLU"
		self.act_kwargs = {"negative_slope": 0.1}
		self.first_conv_kernel_size = 5
		self.image_size = 32
		self.encoder = Encoder(
			dim=self.dim,
			in_channel=self.in_channel,
			layers=self.layers,
			layer_mults=self.layer_mults,
			num_res_blocks=self.num_res_blocks,
			group=self.group,
			conv_type=self.conv_type,
			act_func=self.act_func,
			act_kwargs=self.act_kwargs,
			first_conv_kernel_size=self.first_conv_kernel_size
		)

	def test_different_conv_types(self):
		# conv1d
		encoder_1d = Encoder(
			dim=self.dim,
			in_channel=self.in_channel,
			layers=self.layers,
			layer_mults=self.layer_mults,
			num_res_blocks=self.num_res_blocks,
			group=self.group,
			conv_type="conv1d",
			act_func=self.act_func,
			act_kwargs=self.act_kwargs,
			first_conv_kernel_size=self.first_conv_kernel_size
		)
		x1d = torch.randn(2, self.in_channel, self.image_size)
		out1d = encoder_1d(x1d)
		self.assertEqual(out1d.shape[0], 2)
		self.assertEqual(out1d.shape[1], encoder_1d.encoded_dim)

		# conv2d
		encoder_2d = Encoder(
			dim=self.dim,
			in_channel=self.in_channel,
			layers=self.layers,
			layer_mults=self.layer_mults,
			num_res_blocks=self.num_res_blocks,
			group=self.group,
			conv_type="conv2d",
			act_func=self.act_func,
			act_kwargs=self.act_kwargs,
			first_conv_kernel_size=self.first_conv_kernel_size
		)
		x2d = torch.randn(2, self.in_channel, self.image_size, self.image_size)
		out2d = encoder_2d(x2d)
		self.assertEqual(out2d.shape[0], 2)
		self.assertEqual(out2d.shape[1], encoder_2d.encoded_dim)

		# conv3d
		encoder_3d = Encoder(
			dim=self.dim,
			in_channel=self.in_channel,
			layers=self.layers,
			layer_mults=self.layer_mults,
			num_res_blocks=self.num_res_blocks,
			group=self.group,
			conv_type="conv3d",
			act_func=self.act_func,
			act_kwargs=self.act_kwargs,
			first_conv_kernel_size=2  # smaller kernel size to avoid input size error
		)
		x3d = torch.randn(2, self.in_channel, 8, self.image_size, self.image_size)  # increase depth to 8
		out3d = encoder_3d(x3d)
		self.assertEqual(out3d.shape[0], 2)
		self.assertEqual(out3d.shape[1], encoder_3d.encoded_dim)

	def test_properties(self):
		self.assertEqual(self.encoder.dim, self.dim)
		self.assertEqual(self.encoder.in_channel, self.in_channel)
		self.assertEqual(self.encoder.layers, self.layers)
		self.assertEqual(self.encoder.group, self.group)
		self.assertEqual(self.encoder.conv_type, self.conv_type)
		self.assertEqual(self.encoder.act_func, self.act_func)
		self.assertEqual(self.encoder.act_kwargs, self.act_kwargs)
		self.assertEqual(self.encoder.first_conv_kernel_size, self.first_conv_kernel_size)
		self.assertEqual(self.encoder.layer_mults, self.layer_mults)
		self.assertEqual(self.encoder.num_res_blocks, self.num_res_blocks)

	def test_encoded_dim(self):
		expected_dim = self.dim * self.layer_mults[-1]
		self.assertEqual(self.encoder.encoded_dim, expected_dim)

	def test_fmap_size(self):
		fmap = self.encoder.fmap_size(self.image_size)
		self.assertEqual(fmap, self.image_size // (2 ** self.layers))

	def test_forward_shape(self):
		x: Tensor = torch.randn(2, self.in_channel, self.image_size, self.image_size)
		out: Tensor = self.encoder(x)
		self.assertEqual(out.shape[0], 2)
		self.assertEqual(out.shape[1], self.encoder.encoded_dim)

	def test_forward_grad(self):
		x: Tensor = torch.randn(1, self.in_channel, self.image_size, self.image_size, requires_grad=True)
		out: Tensor = self.encoder(x)
		loss = out.sum()
		loss.backward()
		self.assertIsNotNone(x.grad)

	def test_repr(self):
		self.assertIsInstance(repr(self.encoder), str)

if __name__ == "__main__":
	unittest.main()
