import torch
import torch.nn as nn
from darts import train_operations as ops
from darts.utils import*
from util.maths import fft2c, ifft2c
from darts import genotypes


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


class Cell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, genotype, C):
        super(Cell, self).__init__()
        # print(C_pp, C_p, C)

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.

        self.preproc0 = ops.StdConv(C, C, 1, 1, 0)
        self.preproc1 = ops.StdConv(C, C, 1, 1, 0)

        op_names, indices = self._transfer(genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _transfer(self, normal):
        ops, indices = [], []
        for i, node in enumerate(normal):
            for j, (op, index) in enumerate(node):
                ops.append(op)
                indices.append(index)
        return ops, indices

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = ops.OPS[name](C, stride)
            self._ops += [op]
        self._indices = indices

        # self._ops = to_dag(C, genotype.normal)  # each node have two edge

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        s_out = 0
        for i in self._concat:
            s_out += states[i]
        return s_out
        # return torch.cat([states[i] for i in self._concat], dim=1)


class NasModel(nn.Module):
    # self.C_in, self.C_hidden, self.c_out, self.n_layers, self.n_nodes, alpha
    def __init__(self, C_in, C_hidden, c_out, n_layers, n_nodes, genotype, args, device):
        super(NasModel, self).__init__()
        self.C_in = C_in
        self.C_hidden = C_hidden
        self.c_out = c_out
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.args = args
        self.device = device

        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hidden, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.cells = nn.ModuleList()
        for i in range(n_layers):  # 4
            cell = Cell(genotype, C_hidden)
            self.cells.append(cell)  # get all cells for each layer

        self.linear = nn.Sequential(
            nn.Conv2d(C_hidden, c_out, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.cg = ConjugatedGrad()
        self.iteration = args.iteration  # layer number of modl
        self.lam = nn.Parameter(torch.FloatTensor([0.05]).to(self.device), requires_grad=True)

    def Dw(self, x):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        logits = self.linear(s1)

        return x + logits

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
