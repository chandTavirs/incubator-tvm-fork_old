import torch.nn as nn
import torch
from .dynamic_conv import DynamicConv2D

class DynamicResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, max_residual_depth=2, use_1x1_conv=False):
        super(DynamicResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_residual_depth = max_residual_depth
        self.use_1x1_conv = use_1x1_conv

        self.convs = nn.ModuleList()

        for i in range(max_residual_depth):
            if i == 0:
                self.convs.append(DynamicConv2D(in_channels, out_channels, kernel_size, stride, padding))
            else:
                self.convs.append(DynamicConv2D(out_channels, out_channels, kernel_size, 1, padding))

        if self.use_1x1_conv:
            self.downsample = DynamicConv2D(in_channels, out_channels, 1, stride, 0)

        self.bns = nn.ModuleList(nn.BatchNorm2d(out_channels) for i in range(max_residual_depth))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, depth, decomp_type_list, downsample_decomp_type=0):
        if depth == 0:
            # just pass the input through
            return x

        residual = x
        for i in range(depth):
            residual = self.convs[i](residual, decomp_type_list[i])
            residual = self.bns[i](residual)
            if i < depth - 1:
                residual = self.relu(residual)

        if self.use_1x1_conv:
            x = self.downsample(x, downsample_decomp_type)

        return self.relu(x + residual)

