import numpy
import numpy as np
import scipy.signal
import scipy.ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import torch
import csv
import pandas as pd


def mse_slice(gt, pred):
    """
    Compute Normalized Mean Squared Error (NMSE)
    :param gt:
    :param pred:
    :return:
    """
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    MSE = 0.0
    for i in range(batch_size):
        MSE += mean_squared_error(gt[i], pred[i])

    return MSE / batch_size


def psnr_slice(gt, pred, maxval=None):  # [b, 320, 320, 2]b=2
    """
    Compute Peak Signal to Noise Ratio metric (PSNR)
    :param gt:
    :param pred:
    :param maxval:
    :return:
    """
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    PSNR = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        PSNR += peak_signal_noise_ratio(gt[i], pred[i], data_range=max_val)

    return PSNR / batch_size


def ssim_slice(gt, pred, maxval=None):
    """
    Compute Structural Similarity Index Metric (SSIM)
    :param gt:
    :param pred:
    :param maxval:
    :return:
    """
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    SSIM = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        SSIM += structural_similarity(gt[i], pred[i], data_range=max_val)

    return SSIM / batch_size


def vif_slice(gt, pred):
    """
    Compute Visual information quality
    :param gt:
    :param pred:
    :return:
    """
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    VIF = 0.0
    for i in range(batch_size):
        VIF += compare_vifp(gt[i], pred[i])

    return VIF / batch_size


def save_csv(datas, columns, url):
    """
    save result to csv
    :param data:
    :param columns:
    :param url:
    :return:
    """
    try:
        with open(url, 'w')as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for data in datas:
                writer.writerow(data)
    except IOError:
        print('I/O error')


def dict_to_csv(data, columns, url):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(url, index=False)


from collections import OrderedDict
def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v
    return new_state_dict
