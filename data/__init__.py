import os
import torch
from data.siat import SIATData
from data.cc359 import CC359Data
from data.fastMRI import FastMRIData
from torch.utils.data import DataLoader


def build_dataloader(args,  sample_rate, mode='train'):
    fastmri = FastMRIData(args, args.fast_data_path, sample_rate=sample_rate, mode=mode)
    cc359 = CC359Data(args, args.cc359_data_path, sample_rate=sample_rate, mode=mode)
    siat = SIATData(args, args.siat_data_path, sample_rate=sample_rate, mode=mode)

    datasets = [fastmri, cc359, siat]

    data_loader = []
    dataset_len = []

    for dataset in datasets:
        dataset_len.append(len(dataset))
        if mode == 'train':
            data_loader.append(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.NUM_WORKERS, pin_memory=True))
        elif mode == 'val':
            data_loader.append(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.NUM_WORKERS, pin_memory=True))
        else:
            data_loader.append(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.NUM_WORKERS, pin_memory=True))
    return data_loader, dataset_len
