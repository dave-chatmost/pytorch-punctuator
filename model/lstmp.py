import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMP(nn.Module):
    """Multi-layer LSTM with Projection.
    Support optional peephole connection and projection connection.
    """

    def __init__(self, input_size, hidden_size, num_layers=1,
                 use_peepholes=False, proj_size=None):
        """
        Inputs:
        - input_size: D
        - hidden_size: H
        - num_layers: L
        NOTE: right now, only support time first. not support batch first
        """
        super(LSTMP, self).__init__()
        # remember hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_peepholes = use_peepholes
        self.proj_size = proj_size
        # model parameters
        self.cell_lst = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = input_size
            elif proj_size is not None:
                layer_input_size = proj_size
            else:
                layer_input_size = hidden_size
            # layer_input_size = input_size if layer == 0 else hidden_size
            self.cell_lst.append(LSTMPCell(layer_input_size,
                                           hidden_size,
                                           use_peepholes=use_peepholes,
                                           proj_size=proj_size))
            # self.cell_lst.append(LSTMCell(layer_input_size, hidden_size))

    def forward(self, input, prev):
        """
        Inputs:
        - input: (T, N, D)
        - prev: (h0, c0), each is (L, N, H)

        Returns a tuple of:
        - output: (T, N, H)
        - next: (hT, cT), each is (L, N, H)
        """
        T = input.size(0)
        layer_input = input
        next_h, next_c = [], []
        for layer, cell in enumerate(self.cell_lst):
            layer_prev = prev[0][layer], prev[1][layer]  # (h0, c0)
            layer_output = []
            for t in range(T):
                layer_prev = cell(layer_input[t], layer_prev)
                layer_output.append(layer_prev[0])
            next_h.append(layer_prev[0])  # hT, (N, D)
            next_c.append(layer_prev[1])  # cT, (N, D)
            layer_input = torch.stack(layer_output, 0)  # (T, N, D)
        return layer_input, (torch.stack(next_h, 0), torch.stack(next_c, 0))


class LSTMPCell(nn.Module):
    """LSTM cell with peephole connection and Projection connection.

    This implmentation is based on: 
        https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.

    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """

    def __init__(self, input_size, hidden_size, use_peepholes=False,
                 proj_size=None, bias=True):
        super(LSTMPCell, self).__init__()
        # remember hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_peepholes = use_peepholes
        self.proj_size = proj_size
        self.bias = bias
        # model parameters
        self.weight_ih = nn.Parameter(torch.Tensor(input_size,
                                                   4 * hidden_size))
        if proj_size is not None:
            self.weight_hh = nn.Parameter(torch.Tensor(proj_size,
                                                       4 * hidden_size))
        else:
            self.weight_hh = nn.Parameter(torch.Tensor(hidden_size,
                                                       4 * hidden_size))
        if use_peepholes:
            self.weight_ic_diag = nn.Parameter(torch.Tensor(hidden_size))
            self.weight_fc_diag = nn.Parameter(torch.Tensor(hidden_size))
            self.weight_oc_diag = nn.Parameter(torch.Tensor(hidden_size))
        if proj_size is not None:
            self.weight_hm = nn.Parameter(torch.Tensor(hidden_size,
                                                       proj_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, prev):
        """
        Inputs:
        - input: (N, D)
        - prev: (prev_h, prev_c), each is (N, H)
        """
        prev_h, prev_c = prev
        affine = input.mm(self.weight_ih) + prev_h.mm(self.weight_hh)  # N x 4H
        if self.bias:
            affine += self.bias_ih + self.bias_hh
        ai, af, ag, ao = torch.split(affine, self.hidden_size, dim=1)
        if self.use_peepholes:
            i = torch.sigmoid(ai + prev_c * self.weight_ic_diag)
            f = torch.sigmoid(af + prev_c * self.weight_fc_diag)
            g = torch.tanh(ag)
            next_c = f * prev_c + i * g
            o = torch.sigmoid(ao + next_c * self.weight_oc_diag)
        else:
            i = torch.sigmoid(ai)
            f = torch.sigmoid(af)
            g = torch.tanh(ag)
            next_c = f * prev_c + i * g
            o = torch.sigmoid(ao)
        next_h = o * torch.tanh(next_c)
        if self.proj_size is not None:
            next_h = next_h.mm(self.weight_hm)
        return next_h, next_c


class LSTMCell(nn.Module):
    """Basic LSTM Cell.
    The implementation is based on: https://arxiv.org/pdf/1409.2329.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        # remember hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # model parameters
        self.weight_ih = nn.Parameter(torch.Tensor(input_size,
                                                   4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size,
                                                   4 * hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, prev):
        """
        Inputs:
        - input: (N, D)
        - prev: (prev_h, prev_c), each is (N, H)
        """
        prev_h, prev_c = prev
        affine = input.mm(self.weight_ih) + prev_h.mm(self.weight_hh)  # N x 4H
        if self.bias:
            affine += self.bias_ih + self.bias_hh
        ai, af, ag, ao = torch.split(affine, self.hidden_size, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        g = torch.tanh(ag)
        o = torch.sigmoid(ao)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c
