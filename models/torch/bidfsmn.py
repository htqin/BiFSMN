import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
# from .utils import weight_init
import argparse

class BiDfsmnLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 backbone_memory_size,
                 left_kernel_size,
                 right_kernel_size,
                 dilation=1,
                 dropout=0.0):
        super().__init__()
        self.fc_trans = nn.Sequential(*[
            nn.Linear(backbone_memory_size, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, backbone_memory_size),
            nn.Dropout(dropout)
        ])
        self.memory = nn.Conv1d(backbone_memory_size,
                                backbone_memory_size,
                                kernel_size=left_kernel_size +
                                right_kernel_size + 1,
                                padding=0,
                                stride=1,
                                dilation=dilation,
                                groups=backbone_memory_size)

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        # input (B, N, T)
        residual = input_feat
        # dfsmn-memory
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B,N,T+(l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        # fc-transform
        fc_output = self.fc_trans(memory_out.transpose(1, 2))  # (B, T, N)
        output = fc_output.transpose(1, 2) + residual  # (B, N, T)
        return output


class BiDfsmnLayerBN(nn.Module):
    def __init__(self,
                 hidden_size,
                 backbone_memory_size,
                 left_kernel_size,
                 right_kernel_size,
                 dilation=1,
                 dropout=0.0):
        super().__init__()
        self.fc_trans = nn.Sequential(*[
            nn.Conv1d(backbone_memory_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, backbone_memory_size, 1),
            nn.BatchNorm1d(backbone_memory_size),
            nn.PReLU(),
            nn.Dropout(dropout, ),
        ])
        self.memory = nn.Sequential(*[
            nn.Conv1d(backbone_memory_size,
                      backbone_memory_size,
                      kernel_size=left_kernel_size + right_kernel_size + 1,
                      padding=0,
                      stride=1,
                      dilation=dilation,
                      groups=backbone_memory_size),
            nn.BatchNorm1d(backbone_memory_size),
            nn.PReLU(),
        ])

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        # input (B, N, T)
        residual = input_feat
        # dfsmn-memory
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B,N,T+(l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        # fc-transform
        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        output = fc_output + residual  # (B, N, T)
        return output


class BiDfsmnModel(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_mels=32,
                 num_layer=8,
                 frondend_channels=16,
                 frondend_kernel_size=5,
                 hidden_size=256,
                 backbone_memory_size=128,
                 left_kernel_size=2,
                 right_kernel_size=2,
                 dilation=1,
                 dropout=0.0,
                 dfsmn_with_bn=True,
                 distill=False,
                 **kwargs):
        super().__init__()
        self.front_end = nn.Sequential(*[
            nn.Conv2d(in_channels,
                      out_channels=frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2,
                               frondend_kernel_size // 2)),
            nn.BatchNorm2d(frondend_channels),
            nn.PReLU(),
            nn.Conv2d(frondend_channels,
                      out_channels=2 * frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2,
                               frondend_kernel_size // 2)),
            nn.BatchNorm2d(2 * frondend_channels),
            nn.PReLU()
        ])
        self.n_mels = n_mels
        self.fc1 = nn.Sequential(*[
            nn.Linear(in_features=2 * frondend_channels * self.n_mels // 4,
                      out_features=backbone_memory_size),
            nn.PReLU(),
        ])
        backbone = list()
        for idx in range(num_layer):
            if dfsmn_with_bn:
                backbone.append(
                    BiDfsmnLayerBN(hidden_size, backbone_memory_size,
                                 left_kernel_size, right_kernel_size, dilation,
                                 dropout))
            else:
                backbone.append(
                    BiDfsmnLayer(hidden_size, backbone_memory_size,
                               left_kernel_size, right_kernel_size, dilation,
                               dropout))
        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(*[
            nn.Dropout(p=dropout),
            # nn.Linear(backbone_memory_size * self.n_mels // 4, num_classes),
            nn.Linear(backbone_memory_size * 32 // 4, num_classes),
        ])
        self.distill = distill
        # self.apply(weight_init)

    def forward(self, input_feat):
        # input_feat: B, 1, N, T
        batch = input_feat.shape[0]
        out = self.front_end(input_feat)  # B, C, N//4, T//4
        out = out.view(batch, -1,
                       out.shape[3]).transpose(1, 2).contiguous()  # B, T, N1
        out = self.fc1(out).transpose(1, 2).contiguous()  # B, N, T
        features = []
        for layer in self.backbone:
            out = layer(out)
            features.append(out)
        out = out.contiguous().view(batch, -1)
        out = self.classifier(out)
        if self.distill:
            return out, features
        else:
            return out


class BiDfsmnLayerBN_thinnable(nn.Module):
    def __init__(self,
                 hidden_size,
                 backbone_memory_size,
                 left_kernel_size,
                 right_kernel_size,
                 dilation=1,
                 dropout=0.0):
        super().__init__()
        self.fc_trans = nn.Sequential(*[
            nn.Conv1d(backbone_memory_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, backbone_memory_size, 1),
        ])
        self.bn0 = nn.BatchNorm1d(backbone_memory_size)
        self.act0 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(backbone_memory_size)
        self.act1 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(backbone_memory_size)
        self.act2 = nn.PReLU()
        self.bn3 = nn.BatchNorm1d(backbone_memory_size)
        self.act3 = nn.PReLU()
        self.memory = nn.Sequential(*[
            nn.Conv1d(backbone_memory_size,
                      backbone_memory_size,
                      kernel_size=left_kernel_size + right_kernel_size + 1,
                      padding=0,
                      stride=1,
                      dilation=dilation,
                      groups=backbone_memory_size),
            nn.BatchNorm1d(backbone_memory_size),
            nn.PReLU(),
        ])

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat, opt):
        # input (B, N, T)
        residual = input_feat
        # dfsmn-memory
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B,N,T+(l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        # fc-transform
        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        if opt == 0:
            fc_output = self.bn0(fc_output)
            fc_output = self.act0(fc_output)
        elif opt == 1:
            fc_output = self.bn1(fc_output)
            fc_output = self.act1(fc_output)
        elif opt == 2:
            fc_output = self.bn2(fc_output)
            fc_output = self.act2(fc_output)
        elif opt == 3:
            fc_output = self.bn3(fc_output)
            fc_output = self.act3(fc_output)
        else:
            raise Exception('opt should in [0, 1, 2, 3] but opt = {}'.format(opt))
        output = fc_output + residual  # (B, N, T)
        return output


class BiDfsmnModel_thinnable(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_mels=32,
                 num_layer=8,
                 frondend_channels=16,
                 frondend_kernel_size=5,
                 hidden_size=256,
                 backbone_memory_size=128,
                 left_kernel_size=2,
                 right_kernel_size=2,
                 dilation=1,
                 dropout=0.0,
                 dfsmn_with_bn=True,
                 thin_n=3,
                 distill=False,
                 **kwargs):
        super().__init__()
        self.front_end = nn.Sequential(*[
            nn.Conv2d(in_channels,
                      out_channels=frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2,
                               frondend_kernel_size // 2)),
            nn.BatchNorm2d(frondend_channels),
            nn.PReLU(),
            nn.Conv2d(frondend_channels,
                      out_channels=2 * frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2,
                               frondend_kernel_size // 2)),
            nn.BatchNorm2d(2 * frondend_channels),
            nn.PReLU()
        ])
        self.n_mels = n_mels
        self.fc1 = nn.Sequential(*[
            nn.Linear(in_features=2 * frondend_channels * self.n_mels // 4,
                      out_features=backbone_memory_size),
            nn.PReLU(),
        ])
        backbone = list()
        for idx in range(num_layer):
            backbone.append(
                BiDfsmnLayerBN_thinnable(hidden_size, backbone_memory_size,
                             left_kernel_size, right_kernel_size, dilation,
                             dropout))
        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(*[
            nn.Dropout(p=dropout),
            # nn.Linear(backbone_memory_size * self.n_mels // 4, num_classes),
            nn.Linear(backbone_memory_size * 32 // 4, num_classes),
        ])
        self.distill = distill
        self.thin_n = thin_n
        # self.apply(weight_init)

    def forward(self, input_feat, opt):
        # input_feat: B, 1, N, T
        batch = input_feat.shape[0]
        out = self.front_end(input_feat)  # B, C, N//4, T//4
        out = out.view(batch, -1,
                       out.shape[3]).transpose(1, 2).contiguous()  # B, T, N1
        out = self.fc1(out).transpose(1, 2).contiguous()  # B, N, T
        features = []
        if opt == 0:
            for idx in [0, 1, 2, 3, 4, 5, 6, 7]:
                out = self.backbone[idx](out, opt)
                features.append(out)
        elif opt == 1:
            for idx in [1, 3, 5, 7]:
                out = self.backbone[idx](out, opt)
                features.append(out)
        elif opt == 2:
            for idx in [3, 7]:
                out = self.backbone[idx](out, opt)
                features.append(out)
        elif opt == 3:
            for idx in [7]:
                out = self.backbone[idx](out, opt)
                features.append(out)

        out = out.contiguous().view(batch, -1)
        out = self.classifier(out)
        if self.distill:
            return out, features
        else:
            return out


class DfsmnLayerBN_pre(nn.Module):
    def __init__(self,
                 hidden_size,
                 backbone_memory_size,
                 left_kernel_size,
                 right_kernel_size,
                 dilation=1,
                 dropout=0.0):
        super().__init__()
        self.fc_trans = nn.Sequential(*[
            nn.Conv1d(backbone_memory_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, backbone_memory_size, 1),
        ])
        self.bn0 = nn.BatchNorm1d(backbone_memory_size)
        self.act0 = nn.PReLU()
        self.memory = nn.Sequential(*[
            nn.Conv1d(backbone_memory_size,
                      backbone_memory_size,
                      kernel_size=left_kernel_size + right_kernel_size + 1,
                      padding=0,
                      stride=1,
                      dilation=dilation,
                      groups=backbone_memory_size),
            nn.BatchNorm1d(backbone_memory_size),
            nn.PReLU(),
        ])
        
        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        # input (B, N, T)
        residual = input_feat
        # dfsmn-memory
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B,N,T+(l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        # fc-transform
        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        fc_output = self.bn0(fc_output)
        fc_output = self.act0(fc_output)
        output = fc_output + residual  # (B, N, T)
        return output


class DfsmnModel_pre(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_mels=32,
                 num_layer=8,
                 frondend_channels=16,
                 frondend_kernel_size=5,
                 hidden_size=256,
                 backbone_memory_size=128,
                 left_kernel_size=2,
                 right_kernel_size=2,
                 dilation=1,
                 dropout=0.2,
                 dfsmn_with_bn=True,
                 distill=False,
                 **kwargs):
        super().__init__()
        self.front_end = nn.Sequential(*[
            nn.Conv2d(in_channels,
                      out_channels=frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2,
                               frondend_kernel_size // 2)),
            nn.BatchNorm2d(frondend_channels),
            nn.ReLU(),
            nn.Conv2d(frondend_channels,
                      out_channels=2 * frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2,
                               frondend_kernel_size // 2)),
            nn.BatchNorm2d(2 * frondend_channels),
            nn.ReLU()
        ])
        self.n_mels = n_mels
        self.fc1 = nn.Sequential(*[
            nn.Linear(in_features=2 * frondend_channels * self.n_mels // 4,
                      out_features=backbone_memory_size),
            nn.ReLU(),
        ])
        backbone = list()
        for idx in range(num_layer):
            backbone.append(
                DfsmnLayerBN_pre(hidden_size, backbone_memory_size,
                             left_kernel_size, right_kernel_size, dilation,
                             dropout))
        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(*[
            nn.Dropout(p=dropout),
            nn.Linear(backbone_memory_size * self.n_mels // 4, num_classes),
        ])
        self.distill = distill
        # self.apply(weight_init)

    def forward(self, input_feat):
        # input_feat: B, 1, N, T
        batch = input_feat.shape[0]
        out = self.front_end(input_feat)  # B, C, N//4, T//4
        out = out.view(batch, -1,
                       out.shape[3]).transpose(1, 2).contiguous()  # B, T, N1
        out = self.fc1(out).transpose(1, 2).contiguous()  # B, N, T
        features = []
        for layer in self.backbone:
            out = layer(out)
            features.append(out)
        out = out.contiguous().view(batch, -1)
        out = self.classifier(out)
        if self.distill:
            return out, features
        else:
            return out


