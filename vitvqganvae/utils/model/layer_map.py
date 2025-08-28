from torch import nn

cnn_mapping = {
    'conv1d': nn.Conv1d,
    'conv2d': nn.Conv2d,
    'conv3d': nn.Conv3d,
}

cnn_transpose_mapping = {
    'conv1d': nn.ConvTranspose1d,
    'conv2d': nn.ConvTranspose2d,
    'conv3d': nn.ConvTranspose3d,
}

cnn_2_ndim = {
    'conv1d': 3,
    'conv2d': 4,
    'conv3d': 5,
}

rearrange_map = {
    3: 'b c h -> b h c',
    4: 'b c h w -> b (h w) c',
    5: 'b c d h w -> b (d h w) c'
}