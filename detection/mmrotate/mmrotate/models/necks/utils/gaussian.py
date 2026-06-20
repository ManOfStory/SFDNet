import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur(nn.Module):
    def __init__(self, channels, sigma):
        super().__init__()
        radius = int(3 * sigma)
        kernel_size = 2 * radius + 1

        x = torch.arange(-radius, radius + 1)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        kernel2d = kernel[:, None] * kernel[None, :]
        kernel2d = kernel2d[None, None, :, :].repeat(channels, 1, 1, 1)

        self.register_buffer('weight', kernel2d)
        self.groups = channels
        self.padding = radius

    def forward(self, x):
        return F.conv2d(
            x, self.weight,
            padding=self.padding,
            groups=self.groups
        )