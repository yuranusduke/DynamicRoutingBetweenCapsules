"""
This file builds CapsNet according to the paper

Created by Kunhong Yu
Date: 2021/07/05
"""
import torch as t
from capsules.CapsuleLayer import CapsuleLayer
from torch.nn.functional import one_hot

class CapsNet(t.nn.Module):
    """Define CapsNet"""

    def __init__(self, input_channels = 1):
        """
        Args :
            --input_channel: input channel, default is 1 for MNIST
        """
        super(CapsNet, self).__init__()

        # 1. Conv1 9 x 9 kernel size, 256 kernels, with stride of 1 and relu
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = input_channels, out_channels = 256, kernel_size = 9, stride = 1),
            t.nn.BatchNorm2d(256), # we add one batch norm layer
            t.nn.ReLU(inplace = True)
        )

        # 2. Primary capsule layer, also conv layer 32 * 8 channels, stride 2, 9 x 9 kernel size
        self.primary_capsule = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 256, out_channels = 32 * 8, kernel_size = 9, stride = 2) # group conv
        )

        # 3. DigitCaps layer
        self.digitcaps = t.nn.Sequential(
            CapsuleLayer(in_dim = 8, out_dim = 16, num_in_caps = 1152, num_out_caps = 10, routing_iter = 3)
        )

        # 4. Reconstruction
        self.reconstruction = t.nn.Sequential(
            t.nn.Linear(16, 512),
            t.nn.ReLU(inplace = True),

            t.nn.Linear(512, 1024),
            t.nn.ReLU(inplace = True),

            t.nn.Linear(1024, 784 * input_channels),
            t.nn.Sigmoid()
        )

    def reconstruction_module(self, x, y = None):
        """x has shape [m, 10, 16], y is label
        return :
            --x_: reconstructed input
        """
        if y is not None:
            mask = one_hot(y, 10) # [m, 10]
            mask = mask.unsqueeze(dim = -1) # [m, 10, 1]
            x = x * mask # [m, 10, 16]
            x = t.sum(x, dim = 1) # [m, 16]

        else:
            x_ = t.norm(x, 2, dim = -1) # [m, 10]
            x_index = t.argmax(x_, dim = -1) # [m, ]
            mask = one_hot(x_index, 10).unsqueeze(dim = -1) # [m, 10, 1]
            x = x * mask # [m, 10, 16]
            x = t.sum(x, dim = 1) # [m, 16]

        x_ = self.reconstruction(x)

        return x_

    def forward(self, x, y = None):
        # x has shape [m, img_channel, img_height, img_width]
        # use MNIST as example in the comment
        # 1. Conv1
        x = self.conv1(x) # [m, 20, 20, 256]

        # 2. Primary capsule
        x = self.primary_capsule(x) # [m, 6, 6, 256]
        x = CapsuleLayer.squash(x)

        # 3. DigitCaps
        m, c, h, w = x.size()
        x = x.view(x.size(0), h * w, c)
        x = x.reshape(x.size(0), h * w * 32, 8)
        # groups = [8] * 32
        # x = t.split(x, groups, dim = -1) # 32 * [m, h * w, 8]
        # x = t.cat(x, dim = 1) # [m, h * w * 32, 8]

        x = self.digitcaps(x) # [m, 10, 16]
        x_original = x

        # 4. Reconstruction
        x_ = self.reconstruction_module(x, y)

        x = t.norm(x, 2, dim = -1)

        return x, x_, x_original