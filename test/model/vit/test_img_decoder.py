import unittest
import torch
from vitvqganvae.model.vit.decoder import ImgDecoder

class TestImgDecoder(unittest.TestCase):
    def setUp(self):
        self.image_size = (32, 32)
        self.patch_size = (8, 8)
        self.in_channel = 3
        self.dim = 16
        self.depth = 2
        self.heads = 2
        self.decoder = ImgDecoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channel=self.in_channel,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads
        )

    def test_forward_shape(self):
        batch_size = 2
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        token = torch.randn(batch_size, num_patches, self.dim)
        out = self.decoder(token)
        self.assertEqual(out.shape, (batch_size, self.in_channel, *self.image_size))

    def test_get_last_layer(self):
        last_layer = self.decoder.get_last_layer()
        self.assertTrue(isinstance(last_layer, torch.nn.Parameter))
        # ConvTranspose2d weight shape: (in_channel, dim, kernel_h, kernel_w)
        self.assertEqual(last_layer.shape[0], self.dim)
        self.assertEqual(last_layer.shape[1], self.in_channel)

    def test_properties(self):
        self.assertEqual(self.decoder.image_size, self.image_size)
        self.assertEqual(self.decoder.patch_size, self.patch_size)
        self.assertEqual(self.decoder.in_channel, self.in_channel)
        self.assertEqual(self.decoder.dim, self.dim)
        self.assertEqual(self.decoder.depth, self.depth)
        self.assertEqual(self.decoder.heads, self.heads)
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        patch_dim = self.in_channel * self.patch_size[0] * self.patch_size[1]
        self.assertEqual(self.decoder.num_patches, num_patches)
        self.assertEqual(self.decoder.patch_dim, patch_dim)

    def test_str(self):
        s = str(self.decoder)
        self.assertIn('ImgEncoder', s)
        self.assertIn('image_size', s)
        self.assertIn('patch_size', s)
        self.assertIn('in_channel', s)
        self.assertIn('dim', s)
        self.assertIn('depth', s)
        self.assertIn('heads', s)

    def test_invalid_patch_size(self):
        with self.assertRaises(AssertionError):
            ImgDecoder(image_size=(32, 32), patch_size=(7, 7))

    def test_causal_kwarg(self):
        with self.assertRaises(AssertionError):
            ImgDecoder(image_size=(32, 32), patch_size=(8, 8), causal=True)

if __name__ == '__main__':
    unittest.main()
