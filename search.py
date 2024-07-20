import argparse
import os
import time
import torch
import numpy as np
from models.FedNASAggregator import FedNASAggregator
from models.FedNASTrainer import FedNASTrainer
import darts.utils as utils
from data import build_dataloader
from tensorboardX import SummaryWriter
import random
import wandb

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--name', type=str, default='GAUTOMRI')
    parser.add_argument('--gpus', type=str, default='0', help='gpu')
    parser.add_argument('--stage', type=str, default='search',
                        help='stage: search; train')

    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')  # 5

    parser.add_argument('--client_number', type=int, default=3, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many round of communications we shoud use')  # 50

    parser.add_argument('--init_channels', type=int, default=2, help='num of init channels')
    parser.add_argument('--n_nodes', type=int, default=3, help='the amount of node in cell ')  # 4
    parser.add_argument('--hidden', type=int, default=16, help='num of hidden channels')
    parser.add_argument('--layers', type=int, default=3, help='DARTS layers')  # 4
    parser.add_argument('--output', type=int, default=2, help='num of output channels')
    parser.add_argument('--iteration', type=int, default=5, help='the number of iteration modl')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')

    parser.add_argument('--arch_learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--lambda_valid_regularizer', type=float, default=0.01, help='validation regularizer parameter')

    parser.add_argument('--arch', type=str, default='FedNAS_V1', help='which architecture to use')

    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--mask_path', type=str, default='./mask/1D_Random/mask_1DRandom_4x_acs24_256x256.mat')
    parser.add_argument('--fast_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/'
                                                              'FedNAS/client3/')
    parser.add_argument('--cc359_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/'
                                'CC359/Raw-data/Single-channel')
    parser.add_argument('--siat_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/'
                                                              'in_house_brain_data/new/')
    parser.add_argument('--train_sample_rate', type=float, default=1.0)
    parser.add_argument('--valid_sample_rate', type=float, default=1.0)
    parser.add_argument('--test_sample_rate', type=float, default=1.0)

    parser.add_argument('--path', type=str, default='./searchs/')

    args = parser.parse_args()
    return args


def main(args, writer, logger):

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
    server = FedNASAggregator(side, valid_dataloaders, w, device, args, writer, logger)

    clients = []
    for idx in range(args.client_number):
        clients.append(FedNASTrainer(side[idx], train_dataloaders[idx], valid_dataloaders[idx], device, args, logger))

    # initialize
    global_model = server.get_model()  # get model
    global_model_params = global_model.state_dict()  # get model_parameters
    global_arch_params = []
    if args.stage == 'search':
        global_arch_params = global_model.arch_parameters()  # get architecture parameters

    print(global_model.genotype())
    # 3. start training
    for round in range(args.comm_round):
        logger.info(' communication: {} round'.format(round+1))
        # client search
        for idx, name in enumerate(side):
            logger.info('######## training ########## {}'.format(name))
            start_time = time.time()
            clients[idx].update_model(global_model_params)
            if args.stage == 'search':
                clients[idx].update_arch(global_arch_params)
            if args.stage == 'search':
                weights, alphas, train_psnr, train_ssim, train_loss = clients[idx].search(round)
            else:
                weights, train_psnr, train_ssim, train_loss = clients[idx].train(round)
                alphas = []
            end_time = time.time()
            logger.info('{} | local searching time cost: {} s'.format(name, end_time - start_time))
            val_psnr, val_ssim, val_loss = clients[idx].local_infer()
            server.add_local_trained_result(name, weights, alphas, train_psnr, train_ssim, train_loss,
                                            val_psnr, val_ssim, val_loss)
        all_received = server.check_whether_all_receive()
        logger.info('whether receive all model = {}'.format(all_received))
        if all_received:
            if args.stage == 'search':
                global_model_params, global_arch_params = server.aggregate()
            else:
                global_model_params = server.aggregate()
                global_arch_params = []
            server.infer()  # global model validation
            server.statistics(round+1)
            if args.stage == 'search':
                server.record_model_global_architecture(round+1)
        logger.info('\n')

    np.savetxt(os.path.join(args.path, 'current_structure_size.txt'), server.current_structure_size, fmt='%.2f', delimiter=' ')
    np.savetxt(os.path.join(args.path, 'select_structure_size.txt'), server.select_structure_size, fmt='%.2f', delimiter=' ')
    np.savetxt(os.path.join(args.path, 'test_psnr.txt'), server.test_psnr, fmt='%.2f', delimiter=' ')


if __name__ == '__main__':
    seed = 42
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
    device = torch.device("cuda:{}".format(args.gpus[0]))
    print(device)
    args.path = os.path.join(args.path, time.strftime("%Y%m%d-%H%M%S")+'_automri')
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    if not os.path.exists(os.path.join(args.path, 'plot_path')):
        os.mkdir(os.path.join(args.path, 'plot_path'))
    args.plot_path = os.path.join(args.path, 'plot_path')
    logger = utils.get_logger(os.path.join(args.path, "{}.log".format(args.name)))

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.path, "tb"))
    # writer.add_text('args', args.as_markdown(), 0)
    start_time = time.time()

    main(args, writer, logger)

    end_time = time.time()
    logger.info('search time cost: {} h'.format((end_time - start_time)/3600.))

    # mark the run as finished
    run.finish()
