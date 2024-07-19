import numpy as np
import torch


def to_tensor(data):
    data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def mask_center(x, mask_from, mask_to):
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    :param data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
    :param shape: The output shape. The shape should be smaller than
            the corresponding dimensions of data.
    :return: The center cropped image.
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    :param data: The complex input tensor to be center cropped. It
            should have at least 3 dimensions and the cropping is applied along
            dimensions -3 and -2 and the last dimensions should have a size of
            2.
    :param shape: The output shape. The shape should be smaller than
            the corresponding dimensions of data.
    :return: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def zero_one_norm(data, eps=1e-11):
    """
    0-1 normalize
    :param data: real value
    :param eps:
    :return:
    """
    data = (data - data.min()) / (data.max() - data.min() + eps)

    return data


def mean_std_norm(data, eps=0.0):
    """
    Normalize the given tensor  with mean std norm
    :param data: Input data to be normalized
    :param eps: Added to stddev to prevent dividing by zero
    :return: Normalized tensor
    """
    mean = data.mean()
    std = data.std()
    data = (data - mean) / (std + eps)

    return data