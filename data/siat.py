import random
import pathlib
import numpy as np
import h5py
import scipy.io as sio
import torch
import os
from torch.utils.data import Dataset
from data.transforms import to_tensor
from util.maths import fft2c, ifft2c
from data.transforms import mean_std_norm


class SIATData(Dataset):
    def __init__(self, args, data_path, sample_rate, mode):
        super(SIATData, self).__init__()
        self.data_path = data_path
        self.mask_path = args.mask_path
        self.sample_rate = sample_rate
        self.mode = mode

        self.examples = []
        if self.mode == 'train':
            path = os.path.join(self.data_path, 'train/')
        elif self.mode == 'val':
            path = os.path.join(self.data_path, 'val/')
        else:
            path = os.path.join(self.data_path, 'test/')
        files = list(pathlib.Path(path).iterdir())
        if self.sample_rate < 1:
            num_examples = round(int(len(files) * self.sample_rate))
            self.examples = files[:num_examples]
        else:
            random.shuffle(files)
            self.examples = files

        self.mask_under = np.array(sio.loadmat(self.mask_path)['mask'])
        self.mask_under = torch.from_numpy(self.mask_under).float()  # [h, w]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file = self.examples[item]
        with h5py.File(file, 'r') as data:
            coilComb = np.array(data['coilComb'])  # [h, w]

        coilComb = to_tensor(coilComb).float()  # [coils, h, w, 2]

        label = torch.view_as_complex(coilComb)  # [coils, h, w], complex

        kspace = fft2c(label)

        under_image = ifft2c(kspace * self.mask_under)

        under_image = torch.view_as_complex(mean_std_norm(torch.view_as_real(under_image), eps=1e-10))
        label = torch.view_as_complex(mean_std_norm(torch.view_as_real(label), eps=1e-10))

        return under_image, label, self.mask_under, file.name, 0