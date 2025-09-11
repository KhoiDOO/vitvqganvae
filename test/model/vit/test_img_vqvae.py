
import unittest
import torch
import os
import tempfile
from vitvqganvae.model.vit.imgvqvae import ImgVQVAE

class TestImgVQVAE(unittest.TestCase):
	def setUp(self):
		self.image_size = 16
		self.patch_size = 4
		self.in_channel = 3
		self.out_channel = 3
		self.dim = 32
		self.depth = 2
		self.heads = 2
		self.codebook_size = 8
		self.model = ImgVQVAE(
			image_size=self.image_size,
			patch_size=self.patch_size,
			in_channel=self.in_channel,
			out_channel=self.out_channel,
			dim=self.dim,
			depth=self.depth,
			heads=self.heads,
			codebook_size=self.codebook_size,
			quantizer_kwargs={"codebook_dim": 8, "decay": 0.99, "commitment_weight": 0.25, "kmeans_init": True, "use_cosine_sim": True},
		)
		self.model.eval()
		self.img = torch.randn(2, self.in_channel, self.image_size, self.image_size)

	def test_encode(self):
		fmap, indices, vq_aux_loss = self.model.encode(self.img)
		self.assertEqual(fmap.shape[0], self.img.shape[0])
		self.assertEqual(fmap.shape[2], self.dim)
		self.assertEqual(indices.shape[0], self.img.shape[0])

	def test_decode(self):
		fmap, _, _ = self.model.encode(self.img)
		out = self.model.decode(fmap)
		self.assertEqual(out.shape, self.img.shape)

	def test_decode_from_ids(self):
		fmap, indices, _ = self.model.encode(self.img)
		out = self.model.decode_from_ids(indices)
		self.assertEqual(out.shape, self.img.shape)

	def test_forward(self):
		out = self.model(self.img)
		self.assertEqual(out.shape, self.img.shape)
		# test with return_loss
		result = self.model(self.img, return_loss=True)
		self.assertIn("recon_loss", result)
		self.assertIn("quantizer_loss", result)
		# test with return_loss and return_recons
		loss, recons = self.model(self.img, return_loss=True, return_recons=True)
		self.assertTrue(torch.is_tensor(loss))
		self.assertEqual(recons.shape, self.img.shape)

	def test_save_and_load(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			path = os.path.join(tmpdir, "vqvae.pth")
			self.model.save(path)
			self.assertTrue(os.path.exists(path))
			model2 = ImgVQVAE(
				image_size=self.image_size,
				patch_size=self.patch_size,
				in_channel=self.in_channel,
				out_channel=self.out_channel,
				dim=self.dim,
				depth=self.depth,
				heads=self.heads,
				codebook_size=self.codebook_size,
				quantizer_kwargs={"codebook_dim": 8, "decay": 0.99, "commitment_weight": 0.25, "kmeans_init": True, "use_cosine_sim": True},
			)
			model2.load(path)
			for p1, p2 in zip(self.model.parameters(), model2.parameters()):
				self.assertTrue(torch.allclose(p1, p2))

	def test_properties(self):
		self.assertEqual(self.model.image_size, self.image_size)
		self.assertEqual(self.model.patch_size, self.patch_size)
		self.assertEqual(self.model.in_channel, self.in_channel)
		self.assertEqual(self.model.out_channel, self.out_channel)
		self.assertEqual(self.model.dim, self.dim)
		self.assertEqual(self.model.depth, self.depth)
		self.assertEqual(self.model.heads, self.heads)
		self.assertIsInstance(self.model.quantizer, torch.nn.Module)
		self.assertEqual(self.model.codebook_size, self.codebook_size)
		self.assertIsInstance(self.model.quantizer_kwargs, dict)
		self.assertIsInstance(self.model.l2_recon_loss, bool)
		self.assertIsInstance(self.model.encoder_attn_kwargs, dict)
		self.assertIsInstance(self.model.decoder_attn_kwargs, dict)
		self.assertIsInstance(self.model.device, torch.device)
		self.assertIsInstance(self.model.fmap, int)

	def test_copy_for_eval(self):
		model_copy = self.model.copy_for_eval()
		self.assertFalse(self.model.training)
		self.assertFalse(model_copy.training)
		self.assertEqual(type(model_copy), type(self.model))

	def test_state_dict_and_load_state_dict(self):
		state = self.model.state_dict()
		model2 = ImgVQVAE(
			image_size=self.image_size,
			patch_size=self.patch_size,
			in_channel=self.in_channel,
			out_channel=self.out_channel,
			dim=self.dim,
			depth=self.depth,
			heads=self.heads,
			codebook_size=self.codebook_size,
			quantizer_kwargs={"codebook_dim": 8, "decay": 0.99, "commitment_weight": 0.25, "kmeans_init": True, "use_cosine_sim": True},
		)
		model2.load_state_dict(state)
		for p1, p2 in zip(self.model.parameters(), model2.parameters()):
			self.assertTrue(torch.allclose(p1, p2))

if __name__ == "__main__":
	unittest.main()
