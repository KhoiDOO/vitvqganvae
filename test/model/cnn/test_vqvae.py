import unittest
import torch
import os
import tempfile
from vitvqganvae.model.cnn.vqvae import VQVAE

class TestVQVAE(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.in_channel = 3
        self.out_channel = 3
        self.layers = 2
        self.layer_mults = None
        self.num_res_blocks = 1
        self.group = 16
        self.codebook_size = 16
        self.image_size = 32
        self.batch_size = 2
        self.device = torch.device('cpu')
        self.vqvae = VQVAE(
            dim=self.dim,
            in_channel=self.in_channel,
            out_channel=self.out_channel,
            layers=self.layers,
            layer_mults=self.layer_mults,
            num_res_blocks=self.num_res_blocks,
            group=self.group,
            codebook_size=self.codebook_size,
            l2_recon_loss=True,
            quantizer_kwargs={
                "codebook_dim": 64,
                "decay": 0.99,
                "commitment_weight": 0.25,
                "kmeans_init": True,
                "use_cosine_sim": True
            },
        ).to(self.device)
        self.img = torch.randn(self.batch_size, self.in_channel, self.image_size, self.image_size, device=self.device)

        self.vqvae1d = VQVAE(
            dim=self.dim,
            in_channel=self.in_channel,
            out_channel=self.out_channel,
            layers=self.layers,
            layer_mults=self.layer_mults,
            num_res_blocks=self.num_res_blocks,
            group=self.group,
            conv_type="conv1d",
            codebook_size=self.codebook_size,
            l2_recon_loss=True,
            quantizer_kwargs={
                "codebook_dim": 64,
                "decay": 0.99,
                "commitment_weight": 0.25,
                "kmeans_init": True,
                "use_cosine_sim": True
            },
        ).to(self.device)
        self.img1d = torch.randn(self.batch_size, self.in_channel, self.image_size, device=self.device)

        self.vqvae3d = VQVAE(
            dim=self.dim,
            in_channel=self.in_channel,
            out_channel=self.out_channel,
            layers=self.layers,
            layer_mults=self.layer_mults,
            num_res_blocks=self.num_res_blocks,
            group=self.group,
            conv_type="conv3d",
            codebook_size=self.codebook_size,
            l2_recon_loss=True,
            quantizer_kwargs={          
                "codebook_dim": 64,
                "decay": 0.99,
                "commitment_weight": 0.25,
                "kmeans_init": True,
                "use_cosine_sim": True
            },
        ).to(self.device)
        self.img3d = torch.randn(self.batch_size, self.in_channel, self.image_size, self.image_size, self.image_size, device=self.device)

    def test_encode(self):
        fmap, indices, vq_aux_loss = self.vqvae.encode(self.img)
        self.assertIsInstance(fmap, torch.Tensor)
        self.assertIsInstance(indices, torch.Tensor)
        self.assertIsInstance(vq_aux_loss, torch.Tensor)

        fmap1d, indices1d, vq_aux_loss1d = self.vqvae1d.encode(self.img1d)
        self.assertIsInstance(fmap1d, torch.Tensor)
        self.assertIsInstance(indices1d, torch.Tensor)
        self.assertIsInstance(vq_aux_loss1d, torch.Tensor)

        fmap3d, indices3d, vq_aux_loss3d = self.vqvae3d.encode(self.img3d)
        self.assertIsInstance(fmap3d, torch.Tensor)
        self.assertIsInstance(indices3d, torch.Tensor)
        self.assertIsInstance(vq_aux_loss3d, torch.Tensor)

    def test_decode(self):
        fmap, _, _ = self.vqvae.encode(self.img)
        out = self.vqvae.decode(fmap)
        self.assertEqual(out.shape, self.img.shape)

        fmap1d, _, _ = self.vqvae1d.encode(self.img1d)
        out1d = self.vqvae1d.decode(fmap1d)
        self.assertEqual(out1d.shape, self.img1d.shape)

        fmap3d, _, _ = self.vqvae3d.encode(self.img3d)
        out3d = self.vqvae3d.decode(fmap3d)
        self.assertEqual(out3d.shape, self.img3d.shape)

    def test_decode_from_ids(self):
        fmap, indices, _ = self.vqvae.encode(self.img)
        out = self.vqvae.decode_from_ids(indices)
        self.assertEqual(out.shape, self.img.shape)

        fmap1d, indices1d, _ = self.vqvae1d.encode(self.img1d)
        out1d = self.vqvae1d.decode_from_ids(indices1d)
        self.assertEqual(out1d.shape, self.img1d.shape)

        fmap3d, indices3d, _ = self.vqvae3d.encode(self.img3d)
        out3d = self.vqvae3d.decode_from_ids(indices3d)
        self.assertEqual(out3d.shape, self.img3d.shape)

    def test_forward(self):
        out = self.vqvae(self.img)
        self.assertEqual(out.shape, self.img.shape)
        loss = self.vqvae(self.img, return_loss=True)
        self.assertIsInstance(loss, dict)
        self.assertIn('recon_loss', loss)
        self.assertIn('quantizer_loss', loss)
        recon_loss, recons = self.vqvae(self.img, return_loss=True, return_recons=True)
        self.assertIsInstance(recon_loss, torch.Tensor)
        self.assertEqual(recons.shape, self.img.shape)

        out1d = self.vqvae1d(self.img1d)
        self.assertEqual(out1d.shape, self.img1d.shape)
        loss1d = self.vqvae1d(self.img1d, return_loss=True)
        self.assertIsInstance(loss1d, dict)
        self.assertIn('recon_loss', loss1d)
        self.assertIn('quantizer_loss', loss1d)
        recon_loss1d, recons1d = self.vqvae1d(self.img1d, return_loss=True, return_recons=True)
        self.assertIsInstance(recon_loss1d, torch.Tensor)
        self.assertEqual(recons1d.shape, self.img1d.shape)

        out3d = self.vqvae3d(self.img3d)
        self.assertEqual(out3d.shape, self.img3d.shape)
        loss3d = self.vqvae3d(self.img3d, return_loss=True)
        self.assertIsInstance(loss3d, dict)
        self.assertIn('recon_loss', loss3d)
        self.assertIn('quantizer_loss', loss3d)
        recon_loss3d, recons3d = self.vqvae3d(self.img3d, return_loss=True, return_recons=True)
        self.assertIsInstance(recon_loss3d, torch.Tensor)
        self.assertEqual(recons3d.shape, self.img3d.shape)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'vqvae.pt')
            self.vqvae.save(path)
            self.assertTrue(os.path.exists(path))
            vqvae2 = VQVAE(
                dim=self.dim,
                in_channel=self.in_channel,
                out_channel=self.out_channel,
                layers=self.layers,
                layer_mults=self.layer_mults,
                num_res_blocks=self.num_res_blocks,
                group=self.group,
                codebook_size=self.codebook_size,
                l2_recon_loss=True
            )
            vqvae2.load(path)
            for p1, p2 in zip(self.vqvae.parameters(), vqvae2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_properties(self):
        self.assertEqual(self.vqvae.dim, self.dim)
        self.assertEqual(self.vqvae.in_channel, self.in_channel)
        self.assertEqual(self.vqvae.out_channel, self.out_channel)
        self.assertEqual(self.vqvae.layers, self.layers)
        self.assertEqual(self.vqvae.num_res_blocks, self.num_res_blocks)
        self.assertEqual(self.vqvae.group, self.group)
        self.assertEqual(self.vqvae.codebook_size, self.codebook_size)
        self.assertIsNotNone(self.vqvae.quantizer)
        self.assertIsInstance(self.vqvae.device, torch.device)
        self.assertIsInstance(self.vqvae.encoded_dim, int)
        self.assertIsInstance(self.vqvae.fmap_size(self.image_size), int)
        self.assertIsInstance(self.vqvae.copy_for_eval(), VQVAE)

    def test_multiple_layers(self):
        for layers, image_size in zip([2, 4, 6], [32, 64, 128]):
            vqvae = VQVAE(
                dim=self.dim,
                in_channel=self.in_channel,
                out_channel=self.out_channel,
                layers=layers,
                layer_mults=None,
                num_res_blocks=self.num_res_blocks,
                group=self.group,
                codebook_size=self.codebook_size,
                l2_recon_loss=True,
                quantizer_kwargs={
                    "codebook_dim": 64,
                    "decay": 0.99,
                    "commitment_weight": 0.25,
                    "kmeans_init": True,
                    "use_cosine_sim": True
                },
            ).to(self.device)
            img = torch.randn(self.batch_size, self.in_channel, image_size, image_size, device=self.device)
            out = vqvae(img)
            self.assertEqual(out.shape, img.shape)
        
        for layers, image_size in zip([2, 4, 6], [32, 64, 128]):
            vqvae3d = VQVAE(
                dim=self.dim,
                in_channel=self.in_channel,
                out_channel=self.out_channel,
                layers=layers,
                layer_mults=None,
                num_res_blocks=self.num_res_blocks,
                group=self.group,
                conv_type="conv3d",
                codebook_size=self.codebook_size,
                l2_recon_loss=True,
                quantizer_kwargs={
                    "codebook_dim": 64,
                    "decay": 0.99,
                    "commitment_weight": 0.25,
                    "kmeans_init": True,
                    "use_cosine_sim": True
                },
            ).to(self.device)
            img3d = torch.randn(self.batch_size, self.in_channel, image_size, image_size, image_size, device=self.device)
            out3d = vqvae3d(img3d)
            self.assertEqual(out3d.shape, img3d.shape) 
        
        for layers, image_size in zip([2, 4, 6], [32, 64, 128]):
            vqvae1d = VQVAE(
                dim=self.dim,
                in_channel=self.in_channel,
                out_channel=self.out_channel,
                layers=layers,
                layer_mults=None,
                num_res_blocks=self.num_res_blocks,
                group=self.group,
                conv_type="conv1d",
                codebook_size=self.codebook_size,
                l2_recon_loss=True,
                quantizer_kwargs={
                    "codebook_dim": 64,
                    "decay": 0.99,
                    "commitment_weight": 0.25,
                    "kmeans_init": True,
                    "use_cosine_sim": True
                },
            ).to(self.device)
            img1d = torch.randn(self.batch_size, self.in_channel, image_size, device=self.device)
            out1d = vqvae1d(img1d)
            self.assertEqual(out1d.shape, img1d.shape)

if __name__ == "__main__":
    unittest.main()
