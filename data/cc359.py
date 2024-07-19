import os
import random
import pathlib
import numpy as np
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from data.transforms import to_tensor
from util.maths import ifft2c
from data.transforms import mean_std_norm


class CC359Data(Dataset):
    def __init__(self, args, data_path,  sample_rate, mode):
        super(CC359Data, self).__init__()
        self.data_path = data_path
        self.mask_path = args.mask_path
        self.sample_rate = sample_rate

        self.mode = mode

        self.examples = []
        if self.mode == 'train':
            path = os.path.join(self.data_path, 'train_/')
        elif self.mode == 'val':
            path = os.path.join(self.data_path, 'val_/')
        else:
            path = os.path.join(self.data_path, 'test_/')
        files = list(pathlib.Path(path).iterdir())
        if self.sample_rate == 1.0:
            random.shuffle(files)
        num_examples = round(int(len(files) * self.sample_rate))
        files = files[:num_examples]
        for file in sorted(files):
            with h5py.File(file, 'r') as data:
                slices = np.array(data['kspace']).shape[0]
            self.examples += [(file, slice_id) for slice_id in range(slices)]

        self.mask_under = np.array(sio.loadmat(self.mask_path)['mask'])
        self.mask_under = torch.from_numpy(self.mask_under).float()  # [h, w]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file, slice_id = self.examples[item]
        with h5py.File(file, 'r') as data:
            full_kspace = np.array(data['kspace'][slice_id, ...])  # [h, w]

        full_kspace = to_tensor(full_kspace).float()  # [h, w, 2]

        full_kspace = torch.view_as_complex(full_kspace)  # [h, w], complex

        under_image = torch.fft.ifftshift(ifft2c(full_kspace*self.mask_under), dim=(0, 1))
        label = torch.fft.ifftshift(ifft2c(full_kspace), dim=(0, 1))  # [h, w],

        under_image = torch.view_as_complex(mean_std_norm(torch.view_as_real(under_image), eps=1e-10))

        label = torch.view_as_complex(mean_std_norm(torch.view_as_real(label), eps=1e-10))

        return under_image, label, self.mask_under, file.name, slice_id