
import unittest
import torch
import os
import tempfile
from vitvqganvae.model.cnn.vae import VAE


class TestVAE(unittest.TestCase):
	def setUp(self):
		self.dim = 64
		self.in_channel = 3
		self.out_channel = 3
		self.layers = 2
		self.layer_mults = [1, 2]
		self.num_res_blocks = 1
		self.group = 16
		self.enc_act_func = "LeakyReLU"
		self.dec_act_func = "GLU"
		self.enc_act_kwargs = {"negative_slope": 0.1}
		self.dec_act_kwargs = {"dim": 1}
		self.first_conv_kernel_size = 3
		self.l2_recon_loss = True
		self.use_variational = True
		self.conv_types = [
			"conv1d", 
			"conv2d", 
			"conv3d"
		]
		self.imgs = {
			"conv1d": torch.randn(2, self.in_channel, 32),
			"conv2d": torch.randn(2, self.in_channel, 32, 32),
			"conv3d": torch.randn(2, self.in_channel, 32, 32, 32),
		}

	def make_model(self, conv_type):
		return VAE(
			dim=self.dim,
			in_channel=self.in_channel,
			out_channel=self.out_channel,
			layers=self.layers,
			layer_mults=self.layer_mults,
			num_res_blocks=self.num_res_blocks,
			group=self.group,
			conv_type=conv_type,
			enc_act_func=self.enc_act_func,
			dec_act_func=self.dec_act_func,
			enc_act_kwargs=self.enc_act_kwargs,
			dec_act_kwargs=self.dec_act_kwargs,
			first_conv_kernel_size=self.first_conv_kernel_size,
			l2_recon_loss=self.l2_recon_loss,
			use_variational=self.use_variational,
		)

	def test_encode(self):
		for conv_type in self.conv_types:
			model = self.make_model(conv_type)
			img = self.imgs[conv_type]
			posterior = model.encode(img)
			self.assertTrue(hasattr(posterior, "mean"))
			self.assertTrue(hasattr(posterior, "logvar"))

	def test_decode(self):
		for conv_type in self.conv_types:
			model = self.make_model(conv_type)
			img = self.imgs[conv_type]
			posterior = model.encode(img)
			z = posterior.sample()
			out = model.decode(z)
			self.assertEqual(out.shape, img.shape)

	def test_forward(self):
		for conv_type in self.conv_types:
			model = self.make_model(conv_type)
			img = self.imgs[conv_type]
			out = model(img)
			self.assertEqual(out.shape, img.shape)
			result = model(img, return_loss=True)
			self.assertIn("recon_loss", result)
			self.assertIn("kl_loss", result)
			loss, recons = model(img, return_loss=True, return_recons=True)
			self.assertTrue(torch.is_tensor(loss))
			self.assertEqual(recons.shape, img.shape)

	def test_properties(self):
		for conv_type in self.conv_types:
			model = self.make_model(conv_type)
			self.assertEqual(model.dim, self.dim)
			self.assertEqual(model.in_channel, self.in_channel)
			self.assertEqual(model.out_channel, self.out_channel)
			self.assertEqual(model.layers, self.layers)
			self.assertEqual(model.layer_mults, self.layer_mults)
			self.assertEqual(model.num_res_blocks, self.num_res_blocks)
			self.assertEqual(model.group, self.group)
			self.assertEqual(model.conv_type, conv_type)
			self.assertEqual(model.enc_act_func, self.enc_act_func)
			self.assertEqual(model.dec_act_func, self.dec_act_func)
			self.assertEqual(model.enc_act_kwargs, self.enc_act_kwargs)
			self.assertEqual(model.dec_act_kwargs, self.dec_act_kwargs)
			self.assertEqual(model.first_conv_kernel_size, self.first_conv_kernel_size)
			self.assertIsInstance(model.device, torch.device)
			self.assertIsInstance(model.encoded_dim, int)
			self.assertIsInstance(model.fmap_size(16), int)
			self.assertIsInstance(model.copy_for_eval(), VAE)

	def test_state_dict_and_load_state_dict(self):
		for conv_type in self.conv_types:
			model = self.make_model(conv_type)
			state = model.state_dict()
			model2 = self.make_model(conv_type)
			model2.load_state_dict(state)
			for p1, p2 in zip(model.parameters(), model2.parameters()):
				self.assertTrue(torch.allclose(p1, p2))

	def test_save_and_load(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			path = os.path.join(tmpdir, "vae.pth")
			for conv_type in self.conv_types:
				model1 = self.make_model(conv_type)
				model1.save(path)
				self.assertTrue(os.path.exists(path))
				model2 = self.make_model(conv_type)
				model2.load(path)
				for p1, p2 in zip(model1.parameters(), model2.parameters()):
					self.assertTrue(torch.allclose(p1, p2))

if __name__ == "__main__":
	unittest.main()
