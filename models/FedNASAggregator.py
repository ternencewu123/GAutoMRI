import time

import torch
import os

import wandb

from darts import genotypes
from darts.search_cnn import Network
from darts.model import NasModel
from torch import nn as nn
from util.metric import psnr_slice, ssim_slice
from darts.visualize import plot
import darts.utils as utils


class FedNASAggregator(object):
    def __init__(self, side, valid_data, w, device, args, writer, logger):
        self.valid_global = valid_data
        self.w = w
        self.clients = side
        self.device = device
        self.args = args
        self.writer = writer
        self.logger = logger
        self.criterion = nn.MSELoss().to(self.device)
        self.model = self.init_model()
        self.model.to(self.device)
        self.model_dict = dict()
        self.arch_dict = dict()
        self.train_psnr_dict = dict()
        self.train_ssim_dict = dict()
        self.train_loss_dict = dict()
        self.train_psnr_avg = 0.
        self.train_ssim_avg = 0.
        self.train_loss_avg = 0.

        self.val_psnr_dict = dict()
        self.val_ssim_dict = dict()
        self.val_loss_dict = dict()

        self.test_psnr_avg = 0.
        self.test_ssim_avg = 0.
        self.test_loss_avg = 0.

        self.flag_client_model_uploaded_dict = dict()
        for idx, name in enumerate(self.clients):
            self.flag_client_model_uploaded_dict[name] = False

        self.best_psnr = 0.
        self.best_epoch = 0
        self.best_genotype = None
        self.best_psnr_different_architecture = dict()
        self.best_psnr_different_architecture_round = dict()

        self.current_structure_size = []  # all edges for every node
        self.select_structure_size = []  # two edges for every node
        self.test_psnr = []

    def init_model(self):
        if self.args.stage == 'search':
            model = Network(self.args.init_channels, self.args.hidden, self.args.output, self.args.layers, self.criterion,
                            self.device, self.args.n_nodes, self.args)
        else:
            genotype = genotypes.FedNAS_V1
            self.logger.info(genotype)
            model = NasModel(self.args.init_channels, self.args.hidden, self.args.output, self.args.layers,
                             self.args.n_nodes, genotype, self.args, self.device)

        return model

    def get_model(self):
        return self.model

    def add_local_trained_result(self, name, model_params, arch_parms, train_psnr, train_ssim, train_loss, val_psnr, val_ssim, val_loss):
        self.logger.info('add_model. index = {}'.format(name))
        self.model_dict[name] = model_params
        self.arch_dict[name] = arch_parms
        self.train_psnr_dict[name] = train_psnr
        self.train_ssim_dict[name] = train_ssim
        self.train_loss_dict[name] = train_loss
        self.val_psnr_dict[name] = val_psnr
        self.val_ssim_dict[name] = val_ssim
        self.val_loss_dict[name] = val_loss
        self.flag_client_model_uploaded_dict[name] = True

    def check_whether_all_receive(self):
        for idx, name in enumerate(self.clients):
            if not self.flag_client_model_uploaded_dict[name]:
                return False
        for idx, name in enumerate(self.clients):
            self.flag_client_model_uploaded_dict[name] = False
        return True

    def aggregate(self):
        averaged_weights = self.__aggregate_weight()
        self.model.load_state_dict(averaged_weights)
        if self.args.stage == 'search':
            averaged_alphas = self.__aggregate_alpha()
            self.__update_arch(averaged_alphas)
            return averaged_weights, averaged_alphas
        else:
            return averaged_weights

    def __aggregate_weight(self):
        self.logger.info('################## aggregate weights ###########')
        start_time = time.time()

        # averaged_params = self.model_dict[0]
        # for k in averaged_params.keys():
        #     for i in range(self.client_num):
        #         if i == 0:
        #             averaged_params[k] = self.model_dict[i][k]*self.w[i]
        #         else:
        #             averaged_params[k] += self.model_dict[i][k] * self.w[i]
        model_list = []
        for idx, name in enumerate(self.clients):
            model_list.append((self.w[idx], self.model_dict[name]))
        (_, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                w, local_model_params = model_list[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        end_time = time.time()
        self.logger.info('aggregate weights time cost: {} s'.format(end_time-start_time))
        return averaged_params

    def __aggregate_alpha(self):
        self.logger.info('################# aggregate alphas ############')
        start_time = time.time()
        # averaged_alphas = self.arch_dict[0]
        # for k, alpha in enumerate(averaged_alphas):
        #     for i in range(self.client_num):
        #         if i == 0:
        #             averaged_alphas[k] = self.arch_dict[i][k]*self.w[i]
        #         else:
        #             averaged_alphas[k] += self.arch_dict[i][k] * self.w[i]
        alpha_list = []
        # v.grad.data.copy_(g.data)
        for idx, name in enumerate(self.clients):
            alpha_list.append((self.w[idx], self.arch_dict[name]))
        (_, averaged_alphas) = alpha_list[0]
        for index, alpha in enumerate(averaged_alphas):
            for i in range(0, len(alpha_list)):
                w, local_alpha = alpha_list[i]
                if i == 0:
                    averaged_alphas[index].grad.data = local_alpha[index].grad.data * w
                else:
                    averaged_alphas[index].grad.data += local_alpha[index].grad.data * w
        alpha_list.clear()
        del alpha_list
        self.arch_dict.clear()
        end_time = time.time()
        self.logger.info('average alphas time cost: {} s'.format(end_time-start_time))

        return averaged_alphas

    def __update_arch(self, alphas):
        self.logger.info('update architecture. server.')
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    def infer(self):
        self.model.eval()
        self.model.to(self.device)
        start_time = time.time()
        psnrs = []
        ssims = []
        losss = []
        for _, test_data in enumerate(self.valid_global):
            test_psnr = []
            test_ssim = []
            test_loss = []
            # loss
            criterion = nn.MSELoss().to(self.device)
            with torch.no_grad():
                for batch_idx, (x, target, mask, _, _) in enumerate(test_data):
                    x = x.to(self.device, non_blocking=True)
                    y = target.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True)
                    x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
                    y = torch.abs(y)

                    pred = self.model(x, mask)
                    pred = torch.abs(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()))
                    loss = criterion(y, pred)
                    test_psnr.append(psnr_slice(y, pred))
                    test_ssim.append(ssim_slice(y, pred))
                    test_loss.append(loss)

            psnrs.append(sum(test_psnr) / len(test_psnr))
            ssims.append(sum(test_ssim) / len(test_ssim))
            losss.append(sum(test_loss) / len(test_loss))

        self.test_psnr_avg = sum(psnrs) / len(psnrs)
        self.test_ssim_avg = sum(ssims) / len(ssims)
        self.test_loss_avg = sum(losss) / len(losss)

        end_time = time.time()
        self.logger.info('server_infer time cost: {} s'.format(end_time-start_time))

    def statistics(self, round_idx):
        if self.args.stage == 'search':
            # client train psnr
            train_psnr_list = self.train_psnr_dict.values()
            print(train_psnr_list)
            self.train_psnr_avg = sum(train_psnr_list) / len(train_psnr_list)
            self.logger.info('Round: {} | client_train_psnr: {:.4f}'.format(round_idx, self.train_psnr_avg))

            # train ssim
            train_ssim_list = self.train_ssim_dict.values()
            self.train_ssim_avg = sum(train_ssim_list) / len(train_ssim_list)
            self.logger.info('Round: {} | client_train_ssim: {:.4f}'.format(round_idx, self.train_ssim_avg))

            # train loss
            train_loss_list = self.train_loss_dict.values()
            self.train_loss_avg = sum(train_loss_list) / len(train_loss_list)
            self.logger.info('Round: {} | average train loss: {:.6f}'.format(round_idx, self.train_loss_avg))

            # server valid psnr
            self.logger.info('Round: {} | server_test_psnr: {:.4f}'.format(round_idx, self.test_psnr_avg))
            # self.writer.add_scalar('search/server_test_psnr', self.test_psnr_avg, round_idx)
            wandb.log({'search/server_test_psnr': self.test_psnr_avg})
            # valid ssim
            self.logger.info('Round: {} | server_test_ssim: {:.4f}'.format(round_idx, self.test_ssim_avg))
            # self.writer.add_scalar('search/server_test_ssim', self.test_ssim_avg, round_idx)
            wandb.log({'search/server_test_ssim': self.test_ssim_avg})
            # valid loss
            self.logger.info('Round: {} | server_test_loss: {:.6f}'.format(round_idx, self.test_loss_avg))
            # self.writer.add_scalar('search/server_test_loss', self.test_loss_avg, round_idx)
            wandb.log({'search/server_test_loss': self.test_loss_avg})

            # gap
            self.logger.info('Round: {} | search_gap_psnr: {:.4f}'.format(round_idx, self.train_psnr_avg-self.test_psnr_avg))
            self.logger.info('Round: {} | search_gap_ssim: {:.4f}'.format(round_idx, self.train_ssim_avg-self.test_ssim_avg))
            self.logger.info('Round: {} | search_gap_loss: {:.6f}'.format(round_idx, self.train_loss_avg-self.test_loss_avg))

            # clear
            self.train_psnr_dict.clear()
            self.train_ssim_dict.clear()
            self.train_loss_dict.clear()
            self.flag_client_model_uploaded_dict.clear()

    def record_model_global_architecture(self, round_idx):
        # genotype as image
        genotype, _ = self.model.genotype()  # select architecture
        plot_path = os.path.join(self.args.plot_path, "EP{:02d}".format(round_idx))
        caption = "Epoch-{}".format(round_idx)
        plot(genotype.normal, plot_path, caption)

        current_size = self.model.get_current_model_size(self.model.alpha_normal)
        select_size = self.model.get_select_model_size(genotype)
        self.current_structure_size.append(current_size)
        self.select_structure_size.append(select_size)
        self.test_psnr.append(self.test_psnr_avg)

        wandb.log({'search/current_size': current_size})
        wandb.log({'search/select_size': select_size})

        self.logger.info('genotype = {}'.format(genotype))
        self.logger.info('current_size: {} K | select_size: {} K | test_psnr: {} | round: {}'.format(
            current_size, select_size, self.test_psnr_avg, round_idx))

        # save the cnn architecture according to the cnn count
        if select_size not in self.best_psnr_different_architecture.keys():
            self.best_psnr_different_architecture[select_size] = self.test_psnr_avg
            self.best_psnr_different_architecture_round[select_size] = round_idx

        else:
            if self.test_psnr_avg > self.best_psnr_different_architecture[select_size]:
                self.best_psnr_different_architecture[select_size] = self.test_psnr_avg
                self.best_psnr_different_architecture_round[select_size] = round_idx

        if self.test_psnr_avg > self.best_psnr:
            self.best_psnr = self.test_psnr_avg
            self.best_epoch = round_idx
            self.best_genotype = genotype
            utils.save(self.model, os.path.join(self.args.path, 'weight.pth'))

        self.logger.info("Final best PSNR = {:.4f} | round:{: d}".format(self.best_psnr, self.best_epoch))
        self.logger.info("Best Genotype = {}".format(self.best_genotype))


















