import argparse
import os
import time
import torch
import torch.nn as nn
import darts.utils as utils
from data import build_dataloader
import numpy as np
import random
import wandb
import copy
from darts import genotypes
from darts.model import NasModel
from fed_zoo.client import train, local_infer, local_test, local_test1
from fed_zoo.server import aggregate_weight, infer
from util.metric import save_csv, dict_to_csv
from fed_zoo.variance_adjust import weight_adjust

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--name', type=str, default='GAUTOMRI')
    parser.add_argument('--gpus', type=str, default='1', help='gpu')
    parser.add_argument('--stage', type=str, default='train',
                        help='stage: search; train')

    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_number', type=int, default=3, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--init_channels', type=int, default=2, help='num of init channels')
    parser.add_argument('--n_nodes', type=int, default=3, help='the amount of node in cell ')
    parser.add_argument('--hidden', type=int, default=16, help='num of hidden channels')  # 16
    parser.add_argument('--layers', type=int, default=3, help='DARTS layers')  # 3
    parser.add_argument('--output', type=int, default=2, help='num of output channels')
    parser.add_argument('--iteration', type=int, default=5, help='the number of iteration')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')

    parser.add_argument('--arch', type=str, default='FedNAS_V1', help='which architecture to use')

    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--mask_path', type=str, default='../mask/1D_Random/mask_1DRandom_4x_acs24_256x256.mat')
    parser.add_argument('--fast_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/'
                                                              'FedNAS/client3/')
    parser.add_argument('--modl_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/'
                                                              'brain_MoDL/')
    parser.add_argument('--cc359_data_path', type=str,
                        default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/'
                                'CC359/Raw-data/Single-channel')
    parser.add_argument('--siat_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/'
                                                              'in_house_brain_data/new/')
    parser.add_argument('--train_sample_rate', type=float, default=1.0)
    parser.add_argument('--valid_sample_rate', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=0.05)

    parser.add_argument('--path', type=str, default='./trains/')
    args = parser.parse_args()
    return args


def main(args, logger):
    logger.info('load datasets')
    logger.info(args)

    # 1. load data
    train_dataloaders, train_len = build_dataloader(args, args.train_sample_rate, mode='train')
    valid_dataloaders, valid_len = build_dataloader(args, args.valid_sample_rate, mode='val')

    logger.info('training data : {}'.format(train_len))
    logger.info('validation data: {}'.format(valid_len))
    logger.info('\n')

    # 2. aggregator
    side = ['fastmri', 'cc359', 'siat']
    w = torch.tensor([1 / len(side) for _, data in enumerate(side)], requires_grad=False)
    genotype = genotypes.FedNAS_V1
    logger.info(genotype)
    server = NasModel(args.init_channels, args.hidden, args.output, args.layers,
                      args.n_nodes, genotype, args, device)
    clients = [copy.deepcopy(server).to(device) for _ in range(args.client_number)]

    # initialize
    criterion = nn.MSELoss().to(device)

    # optimizer and scheduler
    optimizers = [torch.optim.AdamW(params=clients[idx].parameters(), lr=args.learning_rate)
                  for idx in range(args.client_number)]
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[idx], milestones=[80], gamma=0.1, last_epoch=-1)
                  for idx in range(args.client_number)]

    best_client = [{'loss': 0., 'psnr': 0., 'ssim': 0.} for i in range(args.client_number)]
    client_checkpoint = [{} for i in range(args.client_number)]
    best_server = {'loss': 0., 'psnr': 0., 'ssim': 0.}
    server_checkpoint = {}
    server_dict = {'loss': [], 'psnr': [], 'ssim': []}
    train_dict = [
        {'loss': [], 'psnr': [], 'ssim': []} for i in range(args.client_number)]
    val_dict = [
        {'loss': [], 'psnr': [], 'ssim': []} for i in range(args.client_number)]
    after_val_dict = [
        {'loss': [], 'psnr': [], 'ssim': []} for i in range(args.client_number)]
    loss_before_agg = {}
    loss_after_agg = {}

    # 3. start training
    for round in range(args.comm_round):
        logger.info(' communication: {} round'.format(round + 1))

        # client search
        for idx, name in enumerate(side):
            logger.info('######## training ########## {}'.format(name))
            start_time = time.time()
            train_psnr, train_ssim, train_loss = train(name, clients[idx], train_dataloaders[idx], criterion,
                                                       optimizers[idx], args, device, logger)
            loss_before_agg[idx] = train_loss
            end_time = time.time()
            logger.info('{} | local training time cost: {} s'.format(name, end_time - start_time))
            train_dict[idx]['loss'].append(train_loss)
            train_dict[idx]['psnr'].append(train_psnr)
            train_dict[idx]['ssim'].append(train_ssim)

            val_psnr, val_ssim, val_loss = local_infer(name, clients[idx], valid_dataloaders[idx], criterion,
                                                       device, logger)
            val_dict[idx]['loss'].append(val_loss)
            val_dict[idx]['psnr'].append(val_psnr)
            val_dict[idx]['ssim'].append(val_ssim)

            schedulers[idx].step()
            lr = schedulers[idx].get_last_lr()[0]
            logger.info('{} | epoch: {} | lr: {}'.format(name, round + 1, lr))
            wandb.log({'train/lr/{}'.format(name): lr})

        # server aggregate
        logger.info('################## aggregate weights ###########')
        server, clients = aggregate_weight(server, clients, w)
        # after train
        for idx, name in enumerate(side):
            st_psnr, st_ssim, st_loss = local_test1(name, clients[idx], train_dataloaders[idx], criterion, device, logger)
            loss_after_agg[idx] = st_loss

        # after test
        for idx, name in enumerate(side):
            val_psnr, val_ssim, val_loss = local_test(name, clients[idx], valid_dataloaders[idx], criterion, device,
                                                      logger)
            after_val_dict[idx]['loss'].append(val_loss)
            after_val_dict[idx]['psnr'].append(val_psnr)
            after_val_dict[idx]['ssim'].append(val_ssim)

            if val_psnr > best_client[idx]['psnr']:
                best_client[idx]['loss'] = val_loss
                best_client[idx]['psnr'] = val_psnr
                best_client[idx]['ssim'] = val_ssim
                client_checkpoint[idx] = {
                    '{}'.format(name): clients[idx],
                    'epoch': round + 1,
                    'psnr': val_psnr,
                    'ssim': val_ssim
                }
            # save client model
            torch.save(client_checkpoint[idx], os.path.join(args.checkpoint, '{}.pth'.format(name)))

        # server test
        avg_psnr, avg_ssim, avg_loss = infer(server, valid_dataloaders, criterion, device, logger)
        server_dict['loss'].append(avg_loss)
        server_dict['psnr'].append(avg_psnr)
        server_dict['ssim'].append(avg_ssim)
        # server checkpoint
        if avg_psnr > best_server['psnr']:
            best_server['psnr'] = avg_psnr
            best_server['ssim'] = avg_ssim
            best_server['loss'] = avg_loss
            server_checkpoint = {
                'server': server,
                'epoch': round + 1,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }
        # save server model
        torch.save(server_checkpoint, os.path.join(args.checkpoint, 'server.pth'))

        # weight adjust
        w = weight_adjust(w, loss_before_agg, loss_after_agg, args.scale)

        logger.info('weight: {}'.format(w))

    # logger best metric
    logger.info('the best epoch for server is {}'.format(server_checkpoint['epoch']))
    logger.info('psnr: {:.4f}'.format(server_checkpoint['psnr']))
    logger.info('ssim: {:.4f}'.format(server_checkpoint['ssim']))

    for idx, name in enumerate(side):
        logger.info('the best epoch for {} is {}'.format(name, client_checkpoint[idx]['epoch']))
        logger.info('psnr: {:.4f}'.format(client_checkpoint[idx]['psnr']))
        logger.info('ssim: {:.4f}'.format(client_checkpoint[idx]['ssim']))

    # save metric
    columns = ['loss', 'psnr', 'ssim']
    save_csv(train_dict, columns, os.path.join(args.path, 'train.csv'))
    save_csv(val_dict, columns, os.path.join(args.path, 'val.csv'))
    save_csv(after_val_dict, columns, os.path.join(args.path, 'after_val.csv'))
    dict_to_csv(server_dict, columns, os.path.join(args.path, 'server.csv'))


if __name__ == '__main__':

    seed=22
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    # start a new wandn run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project='GAUTOMRI',

        # track hyperparameters and run metadata
        config=args

    )
    print('current gpu: {}'.format(torch.cuda.current_device()))
    args.path = os.path.join(args.path, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    if not os.path.exists(os.path.join(args.path, 'checkpoint')):
        os.mkdir(os.path.join(args.path, 'checkpoint'))
    args.checkpoint = os.path.join(args.path, 'checkpoint')
    logger = utils.get_logger(os.path.join(args.path, "{}.log".format(args.name)))
    logger.info('seed: {}'.format(seed))

    start_time = time.time()
    main(args, logger)
    end_time = time.time()
    logger.info('total time cost: {} h'.format((end_time - start_time) / 3600.))
    # mark the run as finished
    run.finish()
