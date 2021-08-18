import torch.nn as nn
import torch.nn.functional as F
#from esm.EGNN_Layer import GraphConvolution
#from esm.EGNN_Layer import InnerProductDecoder
import torch
from torch.nn.parameter import Parameter

class Linear_layer(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""
    def __init__(self, in_features, bias=True,act=F.sigmoid,act_true = False):
        super().__init__()
        self.regression = nn.Linear(in_features, 1, bias)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        # 将n张map乘以w+b，经过线性回归变成1张map
        output = self.regression(features).squeeze(-1)
        if self.act_true:
            output = self.act(output)
        return output# squeeze去掉维数为1的的维度
class Conv_layer3(nn.Module):
    def __init__(self, in_channel,act=F.sigmoid,act_true = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        features = features.unsqueeze(0).permute(0,3,1,2)
        output = self.conv2d(features).squeeze(0).squeeze(0)
        if self.act_true:
            output = self.act(output)

        return output# squeeze去掉维数为1的的维度
class Conv_layer5(nn.Module):
    def __init__(self, in_channel,act=F.sigmoid,act_true = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, 1, 5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        features = features.unsqueeze(0).permute(0,3,1,2)
        output = self.conv2d(features).squeeze(0).squeeze(0)
        if self.act_true:
            output = self.act(output)

        return output# squeeze去掉维数为1的的维度
class Conv_layer3_linear(nn.Module):
    def __init__(self, in_channel,hidden_size,act=F.sigmoid,act_true = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, hidden_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.regression = nn.Linear(hidden_size, 1, True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        output = self.conv2d(features).squeeze(0).squeeze(0)
        output1 = output.permute(1, 2, 0)
        output2 = self.regression(output1).squeeze(-1)
        if self.act_true:
            output2 = self.act(output2)
        return output2
class Conv_layer5_linear(nn.Module):
    def __init__(self, in_channel,hidden_size,act=F.sigmoid,act_true = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, hidden_size, 5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.regression = nn.Linear(hidden_size, 1, True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        output = self.conv2d(features).squeeze(0).squeeze(0)
        output1 = output.permute(1, 2, 0)
        output2 = self.regression(output1).squeeze(-1)
        if self.act_true:
            output2 = self.act(output2)
        return output2
class BiLSTM(nn.Module):
    def __init__(self, in_channel,hidden_size,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm = nn.LSTM(in_channel, hidden_size, num_layers=nl,bidirectional=True)
        self.regression = nn.Linear(hidden_size*2, 1, True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        output1,(h_n,c_n) = self.bilstm(features)

        output = self.regression(output1).squeeze(-1)


        if self.act_true:
            output = self.act(output)

        return output# squeeze去掉维数为1的的维度
class TurnBiLSTM(nn.Module):
    def __init__(self, in_channel,hidden_size,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm = nn.LSTM(in_channel, hidden_size, num_layers=nl,bidirectional=True)
        self.regression = nn.Linear(hidden_size*4, 1, True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        f2 = features.permute(1,0,2)
        output1,(h_n,c_n) = self.bilstm(features)
        output2, (h_n, c_n) = self.bilstm(f2)
        output3 = torch.cat((output1,output2.permute(1,0,2)),-1)
        output = self.regression(output3).squeeze(-1)


        if self.act_true:
            output = self.act(output)

        return output# squeeze去掉维数为1的的维度
class CrossBiLSTM(nn.Module):
    def __init__(self, in_channel,hidden_size1,hidden_size2,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm_h = nn.LSTM(in_channel, hidden_size1, num_layers=nl,bidirectional=True)
        self.bilstm_l = nn.LSTM(hidden_size1*2, hidden_size2, num_layers=nl, bidirectional=True)
        self.regression = nn.Linear(hidden_size2*2, 1, True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):

        output1,(h_n,c_n) = self.bilstm_h(features)
        output11 = output1.permute(1,0,2)
        output2, (h_n, c_n) = self.bilstm_l(output11)
        output3 = output2.permute(1,0,2)
        output = self.regression(output3).squeeze(-1)


        if self.act_true:
            output = self.act(output)

        return output# squeeze去掉维数为1的的维度