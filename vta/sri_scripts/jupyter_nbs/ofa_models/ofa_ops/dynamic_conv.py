import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2D(nn.Module):
    def __init__ (self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DynamicConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.dilation = dilation
        # self.groups = groups
        # self.bias = bias

        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    # get active weights from the base_conv.
    # active_in_channels and active_out_channels are lists of tuples, where each tuple contains the start and end indices of the active channels.
    def get_active_weights(self, active_in_channels, active_out_channels):
        active_in_channels = [(0, self.in_channels)] if active_in_channels is None else active_in_channels
        active_out_channels = [(0, self.out_channels)] if active_out_channels is None else active_out_channels

        active_weights = self.base_conv.weight[active_out_channels[0][0]:active_out_channels[0][1], active_in_channels[0][0]:active_in_channels[0][1], :, :]
        return active_weights

    def forward(self, x, decompose_type):
        if decompose_type is None or decompose_type == 0:
            return self.base_conv(x)
        elif decompose_type == 1:
            # split the output channels into two groups
            active_out_channels = [(0, self.out_channels // 2), (self.out_channels // 2, self.out_channels)]
            active_in_channels = [(0, self.in_channels), (0, self.in_channels)]
            input_slice_indices = active_in_channels
        elif decompose_type == 2:
            # split the output channels into 4 groups
            active_out_channels = [(0, self.out_channels // 4), (self.out_channels // 4, self.out_channels // 2), (self.out_channels // 2, 3 * self.out_channels // 4), (3 * self.out_channels // 4, self.out_channels)]
            active_in_channels = [(0, self.in_channels), (0, self.in_channels), (0, self.in_channels), (0, self.in_channels)]
        elif decompose_type == 3:
            # split the input channels into two groups
            active_in_channels = [(0, self.in_channels // 2), (self.in_channels // 2, self.in_channels)]
            active_out_channels = [(0, self.out_channels), (0, self.out_channels)]
        elif decompose_type == 4:
            # split the input channels into 4 groups
            active_in_channels = [(0, self.in_channels // 4), (self.in_channels // 4, self.in_channels // 2), (self.in_channels // 2, 3 * self.in_channels // 4), (3 * self.in_channels // 4, self.in_channels)]
            active_out_channels = [(0, self.out_channels), (0, self.out_channels), (0, self.out_channels), (0, self.out_channels)]
        else:
            raise ValueError('Unknown decompose_type: {:}'.format(decompose_type))

        convs_completed = 0
        y = None
        for (i_x, i_y), (o_x, o_y) in zip(active_in_channels, active_out_channels):
            filters = self.get_active_weights([(i_x, i_y)], [(o_x, o_y)]).contiguous()
            x_slice = x[:, i_x:i_y, :, :]
            y_slice = F.conv2d(x_slice, filters, None, self.base_conv.stride, self.base_conv.padding, self.base_conv.dilation, self.base_conv.groups)

            if convs_completed == 0:
                y = y_slice
            elif decompose_type == 3 or decompose_type == 4:
                y = y + y_slice
            else:
                y = torch.cat((y, y_slice), 1)
            convs_completed += 1

        return y
