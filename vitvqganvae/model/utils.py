import pickle

from functools import wraps
from pathlib import Path
from packaging import version as packaging_version

from torch import nn
from torch.nn import Module

from beartype import beartype
from beartype.typing import Optional

from pytorch_custom_utils import save_load

import torch

def exists(v):
    return v is not None

def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    else:
        pass

@beartype
def rebuild_save_load(
    save_method_name = 'save',
    load_method_name = 'load',
    config_instance_var_name = '_config',
    init_and_load_classmethod_name = 'init_and_load',
    version: Optional[str] = None
):
    def _save_load(klass):
        assert issubclass(klass, Module), 'save_load should decorate a subclass of torch.nn.Module'

        _orig_init = klass.__init__

        @wraps(_orig_init)
        def __init__(self, *args, **kwargs):
            _config = pickle.dumps((args, kwargs))

            setattr(self, config_instance_var_name, _config)
            _orig_init(self, *args, **kwargs)

        def _save(self, path, overwrite = True):
            path = Path(path)
            assert overwrite or not path.exists()

            pkg = dict(
                model = self.state_dict(),
                config = getattr(self, config_instance_var_name),
                version = version,
            )

            torch.save(pkg, str(path))

        def _load(self, path, strict = True):
            path = Path(path)
            assert path.exists()

            pkg = torch.load(str(path), map_location = 'cpu', weights_only=False)

            if exists(version) and exists(pkg['version']) and packaging_version.parse(version) != packaging_version.parse(pkg['version']):
                print(f'loading saved model at version {pkg["version"]}, but current package version is {version}')

            self.load_state_dict(pkg['model'], strict = strict)

        # init and load from
        # looks for a `config` key in the stored checkpoint, instantiating the model as well as loading the state dict

        @classmethod
        def _init_and_load_from(cls, path, strict = True):
            path = Path(path)
            assert path.exists()
            pkg = torch.load(str(path), map_location = 'cpu')

            assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

            config = pickle.loads(pkg['config'])
            args, kwargs = config
            model = cls(*args, **kwargs)

            _load(model, path, strict = strict)
            return model

        # set decorated init as well as save, load, and init_and_load

        klass.__init__ = __init__
        setattr(klass, save_method_name, _save)
        setattr(klass, load_method_name, _load)
        setattr(klass, init_and_load_classmethod_name, _init_and_load_from)

        return klass

    return _save_load