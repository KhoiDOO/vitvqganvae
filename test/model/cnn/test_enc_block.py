import unittest
import torch
from vitvqganvae.model.cnn.block.enc_block import EncoderBlock

class TestEncoderBlock(unittest.TestCase):
	def setUp(self):
		self.dim = 8
		self.group = 2
		self.conv_type = "conv2d"
		self.act_func = "LeakyReLU"
		self.act_kwargs = {"negative_slope": 0.1}
		self.block = EncoderBlock(
			dim=self.dim,
			group=self.group,
			conv_type=self.conv_type,
			act_func=self.act_func,
			act_kwargs=self.act_kwargs
		)
		
	def test_different_conv_types(self):
		# Test with conv2d
		block_2d = EncoderBlock(
			dim=self.dim,
			group=self.group,
			conv_type="conv2d",
			act_func=self.act_func,
			act_kwargs=self.act_kwargs
		)
		x2d = torch.randn(2, self.dim, 8, 8)
		out2d: torch.Tensor = block_2d(x2d)
		self.assertEqual(out2d.shape, x2d.shape)

		# Test with conv1d
		block_1d = EncoderBlock(
			dim=self.dim,
			group=self.group,
			conv_type="conv1d",
			act_func=self.act_func,
			act_kwargs=self.act_kwargs
		)
		x1d = torch.randn(2, self.dim, 16)
		out1d: torch.Tensor = block_1d(x1d)
		self.assertEqual(out1d.shape, x1d.shape)

		# Test with conv3d
		block_3d = EncoderBlock(
			dim=self.dim,
			group=self.group,
			conv_type="conv3d",
			act_func=self.act_func,
			act_kwargs=self.act_kwargs
		)
		x3d = torch.randn(2, self.dim, 4, 8, 8)
		out3d: torch.Tensor = block_3d(x3d)
		self.assertEqual(out3d.shape, x3d.shape)

	def test_properties(self):
		self.assertEqual(self.block.dim, self.dim)
		self.assertEqual(self.block.group, self.group)
		self.assertEqual(self.block.conv_type, self.conv_type)
		self.assertEqual(self.block.act_func, self.act_func)

	def test_forward_shape(self):
		# Input tensor: batch_size=4, channels=8, height=16, width=16
		x = torch.randn(4, self.dim, 16, 16)
		out: torch.Tensor = self.block(x)
		self.assertEqual(out.shape, x.shape)

	def test_forward_grad(self):
		x = torch.randn(2, self.dim, 8, 8, requires_grad=True)
		out: torch.Tensor = self.block(x)
		loss = out.sum()
		loss.backward()
		self.assertIsNotNone(x.grad)

	def test_repr(self):
		self.assertIsInstance(repr(self.block), str)

if __name__ == "__main__":
	unittest.main()
