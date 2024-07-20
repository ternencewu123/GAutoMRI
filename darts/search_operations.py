""" Operations """
import math
import sys
import torch
import torch.nn as nn
from darts import genotypes as gt


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


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class StdConv(nn.Module):
    """ Standard conv
    Conv - ReLU
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding),
            nn.ReLU(),
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
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    (Dilated) depthwise separable - Pointwise - ReLU

    If dilation == 2, 3x3 conv => 5x5 receptive field
    If dilation == 3, 3x3 conv => 7x7 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2   depthwise + pointwise + ReLU
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, factor=0.):
        if factor > 0.:
            std = x.std() * factor
            means = 0. + torch.zeros_like(x, device=torch.device('cuda'), requires_grad=False)
            noise = torch.normal(means, std).cuda()
            x = x + noise
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


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


class NoiseOp(nn.Module):
    def __init__(self, factor, mean, add_noise=True, args=None):
        super(NoiseOp).__init__()
        self.factor = factor  # factor for std
        self.mean = mean
        self.add_noise = add_noise
        self.args = args

    def forward(self, x):
        if self.add_noise:
            # normal distribution
            std = x.std() * self.factor
            means = self.mean + torch.zeros_like(x, device=torch.device('cuda'), requires_grad=False)
            noise = torch.normal(means, std, out=None).cuda()
            x = x + noise

        return x


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        # temp = []
        # for i, (w, primitive, op) in enumerate(zip(weights, gt.PRIMITIVES, self._ops)):
        #     if 'skip' in primitive:
        #         temp.append(w * op(x, self.factor))
        #     else:
        #         temp.append(w * op(x))
        # res = sum(temp)
        # return res
        return sum(w * op(x) for w, op in zip(weights, self._ops))


