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