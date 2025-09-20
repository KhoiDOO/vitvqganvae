
import unittest
import torch
import numpy as np
from vitvqganvae.model.layers.gaussian import DiagonalGaussianDistribution

class TestDiagonalGaussianDistribution(unittest.TestCase):
	def setUp(self):
		# shape: (batch, channels*2, height, width)
		self.batch = 2
		self.channels = 4
		self.height = 8
		self.width = 8
		self.device = torch.device('cpu')
		# parameters: mean and logvar concatenated along channel dim
		mean = torch.randn(self.batch, self.channels, self.height, self.width)
		logvar = torch.randn(self.batch, self.channels, self.height, self.width)
		self.parameters = torch.cat([mean, logvar], dim=1)
		self.dist = DiagonalGaussianDistribution(self.parameters)
		self.dist_det = DiagonalGaussianDistribution(self.parameters, deterministic=True)

	def test_init_shapes(self):
		self.assertEqual(self.dist.mean.shape, (self.batch, self.channels, self.height, self.width))
		self.assertEqual(self.dist.logvar.shape, (self.batch, self.channels, self.height, self.width))
		self.assertEqual(self.dist.std.shape, (self.batch, self.channels, self.height, self.width))
		self.assertEqual(self.dist.var.shape, (self.batch, self.channels, self.height, self.width))

	def test_deterministic(self):
		self.assertTrue(torch.all(self.dist_det.std == 0))
		self.assertTrue(torch.all(self.dist_det.var == 0))

	def test_sample(self):
		sample = self.dist.sample()
		self.assertEqual(sample.shape, self.dist.mean.shape)
		# deterministic should return mean
		sample_det = self.dist_det.sample()
		self.assertEqual(sample_det.shape, self.dist_det.mean.shape)

	def test_kl_self(self):
		kl = self.dist.kl()
		self.assertIsInstance(kl, torch.Tensor)
		self.assertEqual(kl.shape, (self.batch,))
		# deterministic should return tensor([0.0])
		kl_det = self.dist_det.kl()
		self.assertTrue(torch.allclose(kl_det, torch.tensor([0.0])))

	def test_kl_other(self):
		other = DiagonalGaussianDistribution(self.parameters)
		kl = self.dist.kl(other)
		self.assertIsInstance(kl, torch.Tensor)
		self.assertEqual(kl.shape, (self.batch,))

	def test_nll(self):
		sample = self.dist.sample()
		nll = self.dist.nll(sample)
		self.assertIsInstance(nll, torch.Tensor)
		self.assertEqual(nll.shape, (self.batch,))
		# deterministic should return tensor([0.0])
		nll_det = self.dist_det.nll(sample)
		self.assertTrue(torch.allclose(nll_det, torch.tensor([0.0])))

	def test_mode(self):
		mode = self.dist.mode()
		self.assertTrue(torch.allclose(mode, self.dist.mean))

if __name__ == "__main__":
	unittest.main()
