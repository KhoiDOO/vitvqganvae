import unittest
import torch
import importlib.util
import sys
from pathlib import Path

# Import the LPIPSVAEGAN module directly to avoid importing the full package
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
from vitvqganvae.model.cnn.lpipsvaegan import LPIPSVAEGAN

class TestLPIPSVAEGAN(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.in_channel = 3
        self.out_channel = 3
        # minimal settings for fast test
        self.model = LPIPSVAEGAN(
            dim=self.dim,
            in_channel=self.in_channel,
            out_channel=self.out_channel,
            layers=2,
            discr_layers=2,
            use_gan_loss=False,
            use_perceptual_loss=False,
            use_variational=True,
        )

    def test_forward_returns_tensor_when_not_return_loss(self):
        x = torch.randn(2, self.in_channel, 32, 32)
        out = self.model(x, return_loss=False)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, x.shape)

    def test_forward_returns_loss_dict_when_return_loss(self):
        x = torch.randn(2, self.in_channel, 32, 32)
        losses = self.model(x, return_loss=True)
        self.assertIsInstance(losses, dict)
        self.assertIn('recon_loss', losses)
        self.assertIn('kl_loss', losses)

    def test_forward_recons_flag(self):
        x = torch.randn(1, self.in_channel, 32, 32)
        recon_loss, recons = self.model(x, return_loss=True, return_recons=True)
        self.assertIsInstance(recon_loss, torch.Tensor)
        self.assertIsInstance(recons, torch.Tensor)
        self.assertEqual(recons.shape, x.shape)

    def test_properties_return_expected(self):
        # Check the accessor properties added to LPIPSVAEGAN
        self.assertEqual(self.model.discr_layers, 2)
        self.assertIsNone(self.model.discr_dims)
        self.assertIsNone(self.model.discr_groups)
        self.assertIsNone(self.model.discr_init_kernel_size)
        self.assertIsNone(self.model.discr_act_func)
        self.assertIsNone(self.model.discr_act_kwargs)
        self.assertEqual(self.model.use_perceptual_loss, False)
        self.assertEqual(self.model.use_gan_loss, False)
        self.assertIsNone(self.model.use_hinge_loss)
        self.assertIsNone(self.model.perceptual_model)

    def test_forward_discr(self):
        # instantiate a GAN-enabled model and run forward_discr
        model_gan = LPIPSVAEGAN(
            dim=self.dim,
            in_channel=self.in_channel,
            out_channel=self.out_channel,
            layers=2,
            discr_layers=2,
            use_gan_loss=True,
            use_perceptual_loss=False,
            use_variational=True,
        )

        # basic property checks for GAN-enabled model
        self.assertTrue(model_gan.use_gan_loss)
        self.assertEqual(model_gan.discr_layers, 2)
        # internal discriminator should be created when GAN loss is enabled
        self.assertIsNotNone(getattr(model_gan, '_discr', None))

        x = torch.randn(2, self.in_channel, 32, 32)
        out = model_gan.forward_discr(x, add_gradient_penalty=False)
        self.assertIsInstance(out, dict)
        self.assertIn('discr_loss', out)
        self.assertIsInstance(out['discr_loss'], torch.Tensor)

if __name__ == '__main__':
    unittest.main()
