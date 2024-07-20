import torch
import torch.nn as nn

from darts import genotypes
from darts.architect import Architect
from darts.model import NasModel
from darts.search_cnn import Network
from util.metric import psnr_slice, ssim_slice


class FedNASTrainer(object):
    def __init__(self, client_index, train_local, test_local, device, args, logger):
        self.client_name = client_index
        self.train_local = train_local
        self.test_local = test_local
        self.device = device
        self.args = args
        self.logger = logger
        self.criterion = nn.MSELoss().to(self.device)
        self.model = self.init_model()
        self.model.to(self.device)

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

    def update_model(self, weights):
        self.logger.info('update_model. client= {}'.format(self.client_name))
        self.model.load_state_dict(weights)

    def update_arch(self, alphas):
        self.logger.info('update_arch. client= {}'.format(self.client_name))
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    # local search
    def search(self, round):
        self.model.to(self.device)
        self.model.train()

        arch_parameters = self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        parameters = self.model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params, parameters)

        optimizer = torch.optim.Adam(
            weight_params,  # model.parameters()
            self.args.learning_rate,
            # self.args.momentum,
            # weight_decay=self.args.weight_decay
        )

        architect = Architect(self.model, self.criterion, self.args, self.device)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 40], gamma=0.1, last_epoch=-1
        )

        local_avg_train_psnr = []
        local_avg_train_ssim = []
        local_avg_train_loss = []
        for epoch in range(self.args.epochs):
            # training
            train_psnr, train_ssim, train_loss = self.local_search(
                self.train_local, self.test_local, self.model, architect, self.criterion, optimizer, self.device)
            self.logger.info('{} | epoch : {} | local_search_psnr: {:.4f} | local_search_ssim: {:.4f} | local_search_loss'
                             ': {:.4f}'.format(self.client_name, epoch+1, train_psnr, train_ssim, train_loss))
            local_avg_train_psnr.append(train_psnr)
            local_avg_train_ssim.append(train_ssim)
            local_avg_train_loss.append(train_loss)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        self.logger.info('{} | epoch: {} | lr: {}'.format(self.client_name, round+1, lr))
        weights = self.model.state_dict()
        alphas = self.model.arch_parameters()

        return weights, alphas, sum(local_avg_train_psnr)/len(local_avg_train_psnr), \
                sum(local_avg_train_ssim)/len(local_avg_train_ssim), sum(local_avg_train_loss)/len(local_avg_train_loss)

    def local_search(self, train_loader, valid_loader, model, architect, criterion, optimizer, device):
        psnrs = []
        ssims = []
        losss = []
        for step, ((trn_X, trn_y, t_mask, _, _), (val_X, val_y, v_mask, _, _)) in enumerate(zip(train_loader, valid_loader)):
            trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
            val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
            t_mask, v_mask = t_mask.to(device, non_blocking=True), v_mask.to(device, non_blocking=True)

            trn_X = torch.view_as_real(trn_X).permute(0, 3, 1, 2).contiguous()
            val_X = torch.view_as_real(val_X).permute(0, 3, 1, 2).contiguous()
            trn_y = torch.view_as_real(trn_y).permute(0, 3, 1, 2).contiguous()
            val_y = torch.view_as_real(val_y).permute(0, 3, 1, 2).contiguous()

            # phase 2. architect step (alpha)
            architect.step_milenas_1storder(trn_X, trn_y, val_X, val_y, t_mask, v_mask, self.args.lambda_valid_regularizer)

            # phase 1. child network step (w)
            optimizer.zero_grad()
            logits = model(trn_X, t_mask)
            logits = torch.abs(torch.view_as_complex(logits.permute(0, 2, 3, 1).contiguous()))
            trn_y = torch.abs(torch.view_as_complex(trn_y.permute(0, 2, 3, 1).contiguous()))

            loss = criterion(logits, trn_y)

            loss.backward()
            parameters = model.arch_parameters()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()

            psnr = psnr_slice(trn_y, logits)
            ssim = ssim_slice(trn_y, logits)

            psnrs.append(psnr)
            ssims.append(ssim)
            losss.append(loss)

            # if step % self.args.report_freq == 0:
            #     self.logger.info('client= {} | search: {} | loss: {:.4f} | psnr: {:.4f} | ssim: {:.4f}'.format(
            #         self.client_name, step+1, loss, psnr, ssim
            #     ))
        return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)

    def local_infer(self):
        self.model.eval()
        self.model.to(self.device)
        psnrs = []
        ssims = []
        losss = []
        with torch.no_grad():
            for step, (X, y, mask, _, _) in enumerate(self.test_local):
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                val_X = torch.view_as_real(X).permute(0, 3, 1, 2).contiguous()
                y = torch.abs(y)

                logits = self.model(val_X, mask)
                logits = torch.abs(torch.view_as_complex(logits.permute(0, 2, 3, 1).contiguous()))
                loss = self.criterion(logits, y)

                psnr = psnr_slice(y, logits)
                ssim = ssim_slice(y, logits)
                psnrs.append(psnr)
                ssims.append(ssim)
                losss.append(loss)

            return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)

# ############################################## train ################################

    def train(self, round):
        self.model.to(self.device)
        self.model.train()
        parameters = self.model.parameters()
        optimizer = torch.optim.AdamW(parameters, self.args.learning_rate)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100, 150], gamma=0.1, last_epoch=-1
        )

        local_avg_train_psnr = []
        local_avg_train_ssim = []
        local_avg_train_loss = []
        for epoch in range(self.args.epochs):
            # training
            train_psnr, train_ssim, train_loss = self.local_train(
                self.train_local, self.model, self.criterion, optimizer
            )
            self.logger.info('{} | epoch: {} | local_train_psnr: {:.4f} | local_train_ssim: {:.4f} | local_train_loss: '
                             '{:.4f}'.format(self.client_name, epoch + 1, train_psnr, train_ssim, train_loss))
            local_avg_train_psnr.append(train_psnr)
            local_avg_train_ssim.append(train_ssim)
            local_avg_train_loss.append(train_loss)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        self.logger.info('{} | epoch= {} | lr= {}'.format(self.client_name, round+1, lr))

        weights = self.model.state_dict()

        return weights, sum(local_avg_train_psnr)/len(local_avg_train_psnr),\
                sum(local_avg_train_ssim) / len(local_avg_train_ssim), sum(local_avg_train_loss) / len(local_avg_train_loss)

    def local_train(self, train_queue, model, criterion, optimizer):

        psnrs = []
        ssims = []
        losss = []
        for step, (trn_X, trn_y, mask, _, _) in enumerate(train_queue):
            trn_X, trn_y = trn_X.to(self.device, non_blocking=True), trn_y.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            trn_X = torch.view_as_real(trn_X).permute(0, 3, 1, 2).contiguous()
            trn_y = torch.abs(trn_y)

            optimizer.zero_grad()
            logits = model(trn_X, mask)
            logits = torch.abs(torch.view_as_complex(logits.permute(0, 2, 3, 1).contiguous()))
            loss = criterion(logits, trn_y)
            loss.backward()
            optimizer.step()

            psnr = psnr_slice(trn_y, logits)
            ssim = ssim_slice(trn_y, logits)
            psnrs.append(psnr)
            ssims.append(ssim)
            losss.append(loss)

        return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)

    def infer(self):
        self.model.to(self.device)
        self.model.eval()

        psnrs = []
        ssims = []
        losss = []
        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, mask, _, _) in enumerate(self.test_local):
                trn_X, trn_y = trn_X.to(self.device, non_blocking=True), trn_y.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                trn_X = torch.view_as_real(trn_X).permute(0, 3, 1, 2).contiguous()
                trn_y = torch.abs(trn_y)

                pred = self.model(trn_X, mask)
                pred = torch.abs(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()))
                loss = self.criterion(pred, trn_y)

                psnr = psnr_slice(trn_y, pred)
                ssim = ssim_slice(trn_y, pred)

                psnrs.append(psnr)
                ssims.append(ssim)
                losss.append(loss)

        self.logger.info('{} | local_valid_psnr: {:.4f} | local_valid_ssim: {:.4f} | local_valid_loss: '
                         '{:.4f}'.format(self.client_name, sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)))
        return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)







