from torch import nn

cnn_mapping = {
    'conv1d': nn.Conv1d,
    'conv2d': nn.Conv2d,
    'conv3d': nn.Conv3d,
}