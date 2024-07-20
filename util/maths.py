import torch
import numpy as np


def softmax_loss(data):
    total = 0.
    res = []
    for i in range(len(data)):
        total += np.exp(data[i])
    for i in range(len(data)):
        res.append(np.exp(data[i])/total)
    return res


def rss(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS).
    :param data:
    :param dim:
    :return: torch.Tensor: The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def complex_mul(x, y):
    """
    Complex multiplication
    :param x: A PyTorch tensor with the last dimension of size 2.
    :param y: A PyTorch tensor with the last dimension of size 2.
    :return: torch.Tensor: A PyTorch tensor with the last dimension of size 2.
    """
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x):
    """
    Complex conjugate.
    :param x: x (torch.Tensor): A PyTorch tensor with the last dimension of size 2.
    :return: torch.Tensor: A PyTorch tensor with the last dimension of size 2.
    """
    assert x.shape[-1] == 2

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs_sq(data):
    """
    Compute the squared absolute value of a complex tensor
    :param data:  data (torch.Tensor): A complex valued tensor, where the size of the
            final dimension should be 2.
    :return: torch.Tensor: Squared absolute value of data.
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1)


def fft2c(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    :param data:
    :return:  torch.Tensor: The FFT of the input.
    """
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))

    return data


def ifft2c(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    :param data:
    :return: torch.Tensor: The FFT of the input.
    """
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))

    return data
