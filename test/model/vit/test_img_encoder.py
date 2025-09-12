import unittest
import torch
from vitvqganvae.model.vit.encoder import ImgVITEncoder

class TestImgVITEncoder(unittest.TestCase):
    def setUp(self):
        self.image_size = (32, 32)
        self.patch_size = (8, 8)
        self.in_channel = 3
        self.dim = 16
        self.depth = 2
        self.heads = 2
        self.encoder = ImgVITEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channel=self.in_channel,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads
        )

    def test_patch_embedding_shape(self):
        batch_size = 4
        img = torch.randn(batch_size, self.in_channel, *self.image_size)
        x = self.encoder.to_patch_embedding(img)
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.assertEqual(x.shape, (batch_size, num_patches, self.dim))

    def test_forward_shape(self):
        batch_size = 2
        img = torch.randn(batch_size, self.in_channel, *self.image_size)
        out = self.encoder(img)
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.assertEqual(out.shape, (batch_size, num_patches, self.dim))

    def test_properties(self):
        self.assertEqual(self.encoder.image_size, self.image_size)
        self.assertEqual(self.encoder.patch_size, self.patch_size)
        self.assertEqual(self.encoder.in_channel, self.in_channel)
        self.assertEqual(self.encoder.dim, self.dim)
        self.assertEqual(self.encoder.depth, self.depth)
        self.assertEqual(self.encoder.heads, self.heads)
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        patch_dim = self.in_channel * self.patch_size[0] * self.patch_size[1]
        self.assertEqual(self.encoder.num_patches, num_patches)
        self.assertEqual(self.encoder.patch_dim, patch_dim)

    def test_str(self):
        s = str(self.encoder)
        self.assertIn('ImgEncoder', s)
        self.assertIn('image_size', s)
        self.assertIn('patch_size', s)
        self.assertIn('in_channel', s)
        self.assertIn('dim', s)
        self.assertIn('depth', s)
        self.assertIn('heads', s)

    def test_invalid_patch_size(self):
        with self.assertRaises(AssertionError):
            ImgVITEncoder(image_size=(32, 32), patch_size=(7, 7))

    def test_causal_kwarg(self):
        with self.assertRaises(AssertionError):
            ImgVITEncoder(image_size=(32, 32), patch_size=(8, 8), causal=True)

if __name__ == '__main__':
    unittest.main()
