""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from torch.autograd import Variable
import torch.nn as nn


class Architect(object):
    """ Compute gradients of alphas """
    def __init__(self, model, criterion, args, device):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        param = model.arch_parameters()

        self.arch_optimizer = torch.optim.Adam(
            param,
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay)

    def step_milenas_1storder(self, input_train, target_train, input_valid, target_valid, t_mask, v_mask, lambda_valid_regularizer):
        self.arch_optimizer.zero_grad()
        # grads_alpha_with_train_dataset
        logits = self.model(input_train, t_mask)
        loss_train = self.criterion(logits, target_train)
        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)

        self.arch_optimizer.zero_grad()
        # grads_alpha_with_val_dataset
        logits = self.model(input_valid, v_mask)
        loss_val = self.criterion(logits, target_valid)
        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)

        for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
            g_val.data.copy_(lambda_valid_regularizer * g_val.data)
            g_val.data.add_(g_train.data)

        arch_parameters = self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                # print('hello')
                v.grad = nn.Parameter(g.data).to(self.device)
                # v.grad = Variable(g.data, requires_grad=True)
            else:
                # print(v)
                v.grad.data.copy_(g.data)

        self.arch_optimizer.step()


