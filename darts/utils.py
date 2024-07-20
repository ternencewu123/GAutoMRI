""" Utilities """
import sys
import os
import logging
import shutil
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e3


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_PSNR_SSIM(output, target):
    # """ Computes the precision@k for the specified values of k """

    output = output.to('cpu').squeeze(0).detach().numpy()
    target = target.to('cpu').squeeze(0).detach().numpy()

    return compare_psnr(target, output, data_range=target.max()), compare_ssim(target, output, data_range=target.max())


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_checkpoint(state, ckpt_dir, is_best=False):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def create_exp_dir(path, script_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir: {}'.format(path))

    if script_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in script_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
