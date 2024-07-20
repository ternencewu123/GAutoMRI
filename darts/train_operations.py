""" Operations """
import math
import sys
import torch.nn as nn


OPS = {
    'std_conv_3x3': lambda C, stride: StdConv(C, C, 3, stride, 1),
    'std_conv_5x5': lambda C, stride: StdConv(C, C, 5, stride, 2),
    'std_conv_7x7': lambda C, stride: StdConv(C, C, 7, stride, 3),
    'dil_2_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2, 2),  # 5x5
    'dil_3_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 3, 3),  # 7x7
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride, 2),
    'sep_conv_7x7': lambda C, stride: SepConv(C, C, 7, stride, 3),
}


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
     Conv(Kx1) - Conv(1xK) - ReLU
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride=(stride, 1), padding=(padding, 0)),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride=(1, stride), padding=(0, padding)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class CosineDecayScheduler(object):
    def __init__(self, base_lr=1.0, last_iter=0, T_max=50):
        self.base_lr = base_lr
        self.last_iter = last_iter
        self.T_max = T_max
        self.cnt = 0

    def decay_rate(self, step):
        self.last_iter = step
        decay_rate = self.base_lr * (1+ math.cos(math.pi * self.last_iter / self.T_max)) / 2.0 if self.last_iter <=self.T_max else 0
        return decay_rate






