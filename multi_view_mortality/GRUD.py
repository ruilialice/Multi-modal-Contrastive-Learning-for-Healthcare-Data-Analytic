import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        #         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


class GRUD(nn.Module):
    def __init__(self, input_size, args, X_mean, device,
                 leavesList, ancestorList, mapInfo, ini_embds,
                 output_last=False):
        """
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.

        Implemented based on the paper:
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }

        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """

        super(GRUD, self).__init__()

        self.hidden_size = args.nz
        self.delta_size = input_size
        self.mask_size = input_size

        # self.identity = torch.eye(input_size).to(device)
        self.zeros = Variable(torch.zeros(input_size)).to(device)
        self.zeros_1 = Variable(torch.zeros(self.hidden_size)).to(device)
        self.X_mean = Variable(torch.Tensor(X_mean)).to(device)

        self.zl = nn.Linear(input_size + self.hidden_size + self.mask_size, self.hidden_size)
        self.rl = nn.Linear(input_size + self.hidden_size + self.mask_size, self.hidden_size)
        self.hl = nn.Linear(input_size + self.hidden_size + self.mask_size, self.hidden_size)

        self.classify = nn.Linear(self.hidden_size + 32, 1)

        # self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        self.gamma_x_l = nn.Linear(self.delta_size, self.delta_size)

        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size)

        self.output_last = output_last
        self.device = device

        self.leavesList = leavesList
        self.ancesterList = ancestorList
        self.mapInfo = mapInfo
        self.ini_embd = nn.Parameter(ini_embds)  # using given ini
        self.embd_size = 128
        self.Wa = nn.Linear(2 * 128, 128)
        self.Ua = nn.Linear(128, 1, bias=False)
        self.code_embd_trans = nn.Linear(128, 32)

        self.clinic_embd_common = nn.Linear(32, 32)
        self.code_embd_common = nn.Linear(32, 32)

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros_1, self.gamma_h_l(delta)))
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        combined = torch.cat((x, h, mask), 1)
        z = torch.sigmoid(self.zl(combined))
        r = torch.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = torch.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde

        return h

    def forward(self, X, Mask, Delta, X_last_obsv, seq_len, medical_code, profiles):
        batch_size = X.size(0)
        step_size = X.size(1)
        # compute clinic embd
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size)).to(self.device)
        outputs = torch.zeros((batch_size, 1, self.hidden_size)).to(self.device)
        for i in range(step_size):
            Hidden_State = self.step(torch.squeeze(X[:, i:i + 1, :]) \
                                     , torch.squeeze(X_last_obsv[:, i:i + 1, :]) \
                                     , self.X_mean \
                                     , Hidden_State \
                                     , torch.squeeze(Mask[:, i:i + 1, :]) \
                                     , torch.squeeze(Delta[:, i:i + 1, :]))
            outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
        clinic_embd = torch.cat([temp1[idx, :].unsqueeze(0) for idx, temp1 in zip(seq_len, outputs)])

        # compute diagnosis embd
        embedList = []
        for leaves, ancestors in zip(self.leavesList, self.ancesterList):
            sampleNum, _ = leaves.size()
            leavesEmbd = self.ini_embd[leaves.view(1, -1).squeeze(dim=0)].view(
                sampleNum, -1, self.embd_size)
            ancestorsEmbd = self.ini_embd[ancestors.view(
                1, -1).squeeze(dim=0)].view(sampleNum, -1, self.embd_size)
            concated = torch.cat([leavesEmbd, ancestorsEmbd], dim=2)  # nodeNum * len* 2 embd_size
            weights1 = self.Ua(torch.tanh(self.Wa(concated)))
            weights = F.softmax(weights1, dim=1).transpose(1, 2)
            embedList.append(
                weights.bmm(self.ini_embd[ancestors]).squeeze(dim=1))
        embedMat = torch.cat(embedList, dim=0)
        embedMat = embedMat[self.mapInfo]
        code_embd = torch.einsum('bj,jk->bk', [medical_code, embedMat])  # b * 128
        code_embd = self.code_embd_trans(code_embd)

        clinic_embd_new = self.clinic_embd_common(torch.tanh(clinic_embd))
        code_embd_new = self.code_embd_common(torch.tanh(code_embd))

        finalPre = torch.cat([clinic_embd, code_embd], dim=1)

        pred = torch.sigmoid(self.classify(finalPre).squeeze(dim=-1))

        return pred, clinic_embd_new, code_embd_new

    def initHidden(self, batch_size, device):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
        return Hidden_State