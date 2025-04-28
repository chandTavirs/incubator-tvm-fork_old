import math

import numpy as np
import torch.nn as nn

from .ofa_ops import DynamicResidual

__all__ = ["OFADynamicResnet"]

RESNET_FILTERS = [64, 128, 256, 512, 1024, 2048]
class OFADynamicResnet(nn.Module):
    def __init__(self, num_blocks=4, max_block_depth=4, max_residual_depth=2, num_classes=10):
        super(OFADynamicResnet, self).__init__()
        self.num_blocks = num_blocks
        self.max_residual_depth = max_residual_depth
        self.num_classes = num_classes
        self.max_block_depth = max_block_depth
        self.max_num_residuals_per_block = math.ceil(self.max_block_depth / self.max_residual_depth)

        # first layer of resnet includes 7x7 convolution, batchnorm, relu and maxpool
        self.first_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # last layer of resnet includes adaptive avgpool, flatten and linear layer
        self.last_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                        nn.Linear(RESNET_FILTERS[num_blocks - 1], num_classes))

        self.resnet_blocks = self.get_resnet_blocks()

        self.default_arch = self.default_arch()

        self.sampled_arch = None


    def get_resnet_blocks(self):
        blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            in_channels = RESNET_FILTERS[i] if i == 0 else RESNET_FILTERS[i - 1]
            out_channels = RESNET_FILTERS[i] if i == 0 else RESNET_FILTERS[i]
            stride = 1 if i == 0 else 2
            blocks.append(ResNetBlock(in_channels, out_channels, 3, stride, 1, self.max_residual_depth, self.max_block_depth, i > 0))
        return blocks

    def default_arch(self):
        # residual depth list is number of convs in each residual block. for example, if max_residual_depth=2,
        # and max_block_depth=4, and num_blocks = 4, then residual_depth_list=[[2, 2], [2, 2], [2, 2], [2, 2]]
        # default decomp_type_list will be [[[0,0], [0,0]], [[0,0], [0,0]], [[0,0], [0,0]], [[0,0], [0,0]]] since there are only 2 convs in each residual block.
        # default downsample_decomp_type_list will be [0, 0, 0] since there are only 3 downsample convs.

        arch = {"residual_depth_list": [], "decomp_type_list": [], "downsample_decomp_type_list": []}
        for i in range(self.num_blocks):
            arch["residual_depth_list"].append([])
            arch["decomp_type_list"].append([])
            for j in range(self.max_num_residuals_per_block):
                arch["decomp_type_list"][i].append([])
                arch["residual_depth_list"][i].append(self.max_residual_depth)
                for k in range(self.max_residual_depth):
                    arch["decomp_type_list"][i][j].append(0)

        arch["downsample_decomp_type_list"] = [0] * (self.num_blocks - 1)

        return arch

    def sample_arch(self):
        # sample residual depth list, decomp_type_list and downsample_decomp_type_list
        # rules:
        arch = {"residual_depth_list": [], "decomp_type_list": [], "downsample_decomp_type_list": []}
        for i in range(self.num_blocks):
            arch["residual_depth_list"].append([])
            arch["decomp_type_list"].append([])
            for j in range(self.max_num_residuals_per_block):
                arch["decomp_type_list"][i].append([])
                if j == 0:
                    # first residual block of each block needs to have a depth of at least 1.
                    arch["residual_depth_list"][i].append(np.random.randint(1, self.max_residual_depth + 1))
                else:
                    arch["residual_depth_list"][i].append(np.random.randint(0, self.max_residual_depth + 1))
                for k in range(self.max_residual_depth):
                    arch["decomp_type_list"][i][j].append(np.random.randint(0, 5))
        arch["downsample_decomp_type_list"] = np.random.randint(0, 5, self.num_blocks - 1)

        return arch

    def set_active_subnet(self, arch):
        self.sampled_arch = arch

    def get_active_subnet(self):
        return self.sampled_arch if self.sampled_arch is not None else self.default_arch

    def forward(self, x):
        if self.sampled_arch is None:
            arch = self.default_arch
        else:
            arch = self.sampled_arch

        x = x.reshape(-1, 3, 224, 224)
        x = self.first_layer(x)
        for i in range(self.num_blocks):
            if i == 0:
                x = self.resnet_blocks[i](x, arch["residual_depth_list"][i], arch["decomp_type_list"][i], 0)
            else:
                x = self.resnet_blocks[i](x, arch["residual_depth_list"][i], arch["decomp_type_list"][i], arch["downsample_decomp_type_list"][i-1])

        x = self.last_layer(x)



        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, max_residual_depth=2, max_block_depth=4, use_1x1_conv=False):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_residual_depth = max_residual_depth
        self.use_1x1_conv = use_1x1_conv
        self.max_block_depth = max_block_depth

        self.num_residuals = math.ceil(self.max_block_depth/self.max_residual_depth)
        self.residuals = nn.ModuleList()

        for i in range(self.num_residuals):
            if i == 0:
                self.residuals.append(DynamicResidual(in_channels, out_channels, kernel_size, stride, padding, max_residual_depth, use_1x1_conv))
            else:
                self.residuals.append(DynamicResidual(out_channels, out_channels, kernel_size, 1, padding, max_residual_depth, False))


    def forward(self, x, residual_depth_list, decomp_type_list, downsample_decomp_type):
        for i in range(self.num_residuals):
            x = self.residuals[i](x, residual_depth_list[i], decomp_type_list[i], downsample_decomp_type)
        return x

# test sample_arch generation
# def test_sample_arch():
#     model = OFADynamicResnet()
#     print(model.get_active_subnet())
#
#     arch = model.sample_arch()
#     print(arch)


#test the model and the forward pass

# model = OFADynamicResnet()
# x = torch.randn(1, 3, 224, 224)
# print("OFA Main Model:", model)
#
# print("OFA Main Model Summary:", summary(model, (3, 224, 224)))
#
# for i in range(20):
#     arch = model.sample_arch()
#     model.set_active_subnet(arch)
#     print(f"Active subnet of attempt {i} :: ", model.get_active_subnet())
#
#     print(f"Summary of active subnet {i} :: ", summary(model, (3, 224, 224), batch_size=1))

# manually set default arch and get summary

# model.set_active_subnet(model.default_arch)
# print(f"Default arch :: ", model.get_active_subnet())
# print(f"Summary of default arch :: ", summary(model, (3, 224, 224), batch_size=1))

# manual_arch = {}
# manual_arch["residual_depth_list"] = [[2, 2], [2, 2], [2, 2], [2, 2]]
#
# # set all decomp types to 4
# manual_arch["decomp_type_list"] = [[[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 4]]]
# manual_arch["downsample_decomp_type_list"] = [4, 4, 4]
#
# model.set_active_subnet(manual_arch)
# print(f"Manual arch :: ", model.get_active_subnet())
# print(f"Summary of manual arch :: ", summary(model, (3, 224, 224), batch_size=1))







