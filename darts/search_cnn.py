""" CNN for architecture search """
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import darts.genotypes as gt
import logging
from darts import search_operations as ops
from darts.utils import param_size
from util.maths import fft2c, ifft2c
from darts.model import NasModel


def AtA(data, mask):
    data = fft2c(data)
    data = data * mask
    data = ifft2c(data)

    return data


class ConjugatedGrad(nn.Module):
    def __init__(self):
        super(ConjugatedGrad, self).__init__()

    def forward(self, rhs, mask, lam):
        rhs = torch.view_as_complex(rhs.permute(0, 2, 3, 1).contiguous())
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(torch.conj(r)*r, dim=(-2, -1)).real
        num_iter, epsilon = 10, 1e-10
        for i in range(num_iter):
            Ap = AtA(p, mask) + lam*p
            alpha = rTr / torch.sum(torch.conj(p)*Ap, dim=(-2, -1)).real
            x = x + alpha[:, None, None]*p
            r = r - alpha[:, None, None]*Ap
            rTrNew = torch.sum(torch.conj(r)*r, dim=(-2, -1)).real
            if rTrNew.max() < epsilon:
                break
            beta = rTrNew/rTr
            rTr = rTrNew
            p = r + beta[:, None, None]*p
        return x


class InnerCell(nn.Module):
    def __init__(self, C, alpha):
        """
        Args
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not  false
            reduction: flag for whether the current cell is reduction cell or not   false
        """
        super(InnerCell, self).__init__()

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        self.preproc0 = ops.StdConv(C, C, 1, 1, 0)
        self.preproc1 = ops.StdConv(C, C, 1, 1, 0)

        # generate dag  (Directed Acyclic Graph)
        self.dag = nn.ModuleList()
        for node, edge in enumerate(alpha):  # 2
            self.dag.append(nn.ModuleList())
            for j in range(2+node):  # include 2 input nodes
                stride = 1 if j < 2 else 1

                choice = gt.PRIMITIVES[edge[j].argmax()]

                op = ops.OPS[choice](C, stride)   # sets for all operation
                self.dag[node].append(op)  # [[all operation], [all operation], [all operation]]

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for node, edges in enumerate(self.dag):
            s_cur = sum(edges[i](s) for i, s in enumerate(states))
            states.append(s_cur)

        s_out = 0
        for i in range(2, len(states)):
            s_out += states[i]
        return s_out


class ModelForModelSizeMeasure(nn.Module):
    def __init__(self, C_in, C_hidden, c_out, n_layers, n_node, alpha, args, device):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super(ModelForModelSizeMeasure, self).__init__()
        self.C_in = C_in  # 2
        self.C_hidden = C_hidden  # 64
        self.c_out = c_out  # 2
        self.n_layers = n_layers  # 4
        self.n_nodes = n_node  # 4
        self.alpha = alpha
        self.args = args
        self.device = device

        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hidden, 3, 1, 1),
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.

        self.cells = nn.ModuleList()

        for i in range(n_layers):  # 4
            cell = InnerCell(C_hidden, self.alpha)
            self.cells.append(cell)  # get all cells for each layer

        self.linear = nn.Sequential(
            nn.Conv2d(C_hidden, c_out, 3, 1, 1),
            nn.ReLU()
        )

        self.cg = ConjugatedGrad()
        self.iteration = self.args.iteration  # layer number of modl
        self.lam = nn.Parameter(torch.FloatTensor([0.05]).to(self.device), requires_grad=True)

    def Dw(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        logits = self.linear(s1)
        return x + logits

    def forward(self, under_img, under_mask):
        x = under_img
        for i in range(self.iteration):
            x = self.Dw(x)
            x = under_img + self.lam * x
            x = self.cg(x, under_mask, self.lam)
            x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        x_final = x
        return x_final


#  search model
class Cell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not  false
            reduction: flag for whether the current cell is reduction cell or not   false
        """
        super(Cell, self).__init__()
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.

        self.preproc0 = ops.StdConv(C, C, 1, 1, 0)
        self.preproc1 = ops.StdConv(C, C, 1, 1, 0)

        # generate dag  (Directed Acyclic Graph)
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):  # 4
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 1 if j < 2 else 1
                op = ops.MixedOp(C, stride)  # sets for all operation
                self.dag[i].append(op)  # [[all operation], [all operation], [all operation]]

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):  # [[5,5], [5, 5, 5]], [2x5, 3x5]
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)
        s_out = 0
        for i in range(2, len(states)):
            s_out += states[i]
        return s_out


class Network(nn.Module):
    """ SearchCNN controller supporting multi-gpu
    c_in: ; c: ; n_classes: ; n_layers: ; criterion: l1; n_nodes:2; stem_multiplier:1; device_id=[0,1]
    """

    def __init__(self, C_in, C_hidden, c_out, n_layers, criterion, device, n_nodes, args):
        super(Network, self).__init__()
        self.C_in = C_in
        self.C_hidden = C_hidden
        self.c_out = c_out
        self.n_layers = n_layers
        self.criterion = criterion
        self.device = device
        self.n_nodes = n_nodes

        self.args = args

        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hidden, 3, 1, 1),
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.

        self.cells = nn.ModuleList()
        for i in range(n_layers):  # 4
            cell = Cell(n_nodes, C_hidden)
            self.cells.append(cell)  # get all cells for each layer

        self.linear = nn.Sequential(
            nn.Conv2d(C_hidden, c_out, 3, 1, 1),
            nn.ReLU()
        )

        # initialize architect parameters: alphas
        self._initialize_alphas()
        self.cg = ConjugatedGrad()
        self.iteration = args.iteration  # layer number of modl
        self.lam = nn.Parameter(torch.FloatTensor([0.05]).to(device), requires_grad=True)

    # calculate weight of edge for each node
    def forward(self, under_img, under_mask):
        x = under_img
        for i in range(self.iteration):
            x = self.Dw(x)
            x = under_img + self.lam * x
            x = self.cg(x, under_mask, self.lam)
            x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        x_final = x
        return x_final

    def Dw(self, x):
        s0 = s1 = self.stem(x)
        alpha_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]  # calculate weight of edge for each node

        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, alpha_normal)
        logits = self.linear(s1)

        return x + logits

    def new(self):
        model_new = Network(self.C_in, self.C_hidden, self.c_out, self.n_layers, self.criterion).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_alphas(self):
        n_ops = len(gt.PRIMITIVES)  # the number of operation  5
        self.alpha_normal = nn.ParameterList()
        for i in range(self.n_nodes):  # n_nodes=4
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))

        # setup alphas list
        self._arch_parameters = []
        for n, p in self.named_parameters():  # name, parameter
            if 'alpha' in n:
                self._arch_parameters.append(p)

    def new_arch_parameters(self):
        n_ops = len(gt.PRIMITIVES)  # the number of operation  8
        alpha_normal = nn.ParameterList()
        for i in range(self.n_nodes):  # n_nodes=4
            alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops))).to(self.device)
        # setup alphas list
        _arch_parameters = []
        for an in alpha_normal:
            _arch_parameters.append(an)

        return _arch_parameters

    def loss(self, X, y):
        logits = self.forward(X)

        return self.criterion(logits, y)

    def arch_parameters(self):

        return self._arch_parameters

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal, normal_cnn_count = gt.parse(self.alpha_normal, k=2)  # operation list
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes
        genotype = gt.Genotype(normal=gene_normal, normal_concat=concat)  # [[(node1_ops_1, edge_idx), (node1_ops_2, edge_idx)],
                                                                          # [(node1_ops_1, edge_idx), (node1_ops_2, edge_idx)]]
                                                                          # (2, 4)

        return genotype, normal_cnn_count

    def get_current_model_size(self, normal):
        # C_in, C_hidden, n_classes, n_layers, n_nodes, stem_multiplier
        model = ModelForModelSizeMeasure(self.C_in, self.C_hidden, self.c_out, self.n_layers, self.n_nodes, normal,
                                         self.args, self.device)
        size = param_size(model)
        del model
        return size

    def get_select_model_size(self, genotype):
        s_model = NasModel(self.C_in, self.C_hidden, self.c_out, self.n_layers, self.n_nodes, genotype, self.args, self.device)
        size = param_size(s_model)
        del s_model
        return size



