from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn


#############  Two differentiable layers  #####################
class RDC(nn.Module):
    def __init__(self):
        super(RDC, self).__init__()

    def forward(self, k, size):
        # assert type(size) == torch.Size
        N, C, H, W = size
        N = k.shape[0]
        base_grid = k.new(N, H, W, 2).cuda()
        linear_points = torch.linspace(-1, 1, W).cuda() if W > 1 else torch.Tensor([-1]).cuda()
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H).cuda(), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H).cuda() if H > 1 else torch.Tensor([-1]).cuda()
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W).cuda()).expand_as(base_grid[:, :, :, 1])

        base_grid = base_grid.view(N, H * W, 2)
        # crop in advance
        # base_grid[:, :, 0] = base_grid[:, :, 0] * (1 / (1 + k * 2))
        # base_grid[:, :, 1] = base_grid[:, :, 1] * (1 / (1 + k * 2))
        # in_place operation don't allow gradient backpropagation
        b1 = (base_grid[:, :, 0] * (1 / (1 + k * 2))).unsqueeze(2)
        b2 = (base_grid[:, :, 1] * (1 / (1 + k * 2))).unsqueeze(2)
        base_grid2 = torch.cat((b1, b2), 2)
        # rectify
        ru = torch.norm(base_grid2, 2, 2)
        ru_2 = ru.mul(ru)
        sin_theta = base_grid2[:, :, 0] / ru
        cos_theta = base_grid2[:, :, 1] / ru
        rd = (1 - (1 - 4 * k * ru_2).sqrt()) / (2 * k * ru - 1e-5)
        xd = rd.mul(sin_theta).unsqueeze(2)
        yd = rd.mul(cos_theta).unsqueeze(2)
        grid = torch.cat((xd, yd), 2)
        grid = grid.view(N, H, W, 2)
        return grid


class RDS(nn.Module):
    def __init__(self):
        super(RDS, self).__init__()

    def forward(self, k, size):
        # assert type(size) == torch.Size
        k = k - 1e-5
        N, C, H, W = size
        N = k.shape[0]
        base_grid = k.new(N, H, W, 2).cuda()
        linear_points = torch.linspace(-1, 1, W).cuda() if W > 1 else torch.Tensor([-1]).cuda()
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H).cuda(), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H).cuda() if H > 1 else torch.Tensor([-1]).cuda()
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W).cuda()).expand_as(base_grid[:, :, :, 1])

        base_grid = base_grid.view(N, H * W, 2)
        # crop in advance
        base_grid[:, :, 0] = base_grid[:, :, 0] * ((1 - (1 - 8 * k).sqrt()) / (4 * k))
        base_grid[:, :, 1] = base_grid[:, :, 1] * ((1 - (1 - 8 * k).sqrt()) / (4 * k))
        # distort
        rd = torch.norm(base_grid, 2, 2)
        rd_2 = rd.mul(rd)
        grid = base_grid.mul((1 / ((1 + rd_2 * k) + 1e-5)).unsqueeze(2).repeat(1, 1, 2))
        grid = grid.view(N, H, W, 2)
        return grid


########################## ResNet ########################################################
def conv3_3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False):
        super(residual_block, self).__init__()
        self.downsample = downsample
        stride = 2 if self.downsample else 1  # 判断是否下采样

        self.forwardPath = nn.Sequential(
            conv3_3(in_channel, out_channel, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            conv3_3(out_channel, out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        if self.downsample:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.forwardPath(x)
        if self.downsample:
            x = self.conv3(x)
        return nn.functional.relu(x + out, True)


class ResNet18(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(ResNet18, self).__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=2, padding=1)  # 256->128,64
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),  # 128->64,64
            residual_block(64, 64),
            residual_block(64, 64)
        )
        self.conv3_x = nn.Sequential(
            residual_block(64, 128, downsample=True),  # 64->32,128
            residual_block(128, 128)
        )
        self.conv4_x = nn.Sequential(
            residual_block(128, 256, downsample=True),  # 32->16,256
            residual_block(256, 256)
        )
        self.conv5_x = nn.Sequential(
            residual_block(256, 512, downsample=True),  # 16->8,512
            residual_block(512, 512),
            nn.AvgPool2d(8)  # 8->1,512
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            # nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        if self.verbose:
            print('conv1 output:{}'.format(x.shape))
        x = self.conv2_x(x)
        if self.verbose:
            print('conv2_x output:{}'.format(x.shape))
        x = self.conv3_x(x)
        if self.verbose:
            print('conv3_x output:{}'.format(x.shape))
        x = self.conv4_x(x)
        if self.verbose:
            print('conv4_x output:{}'.format(x.shape))
        x = self.conv5_x(x)
        if self.verbose:
            print('conv5_x output:{}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        k_pred = -x * 0.35
        # avoid nan
        k_pred[torch.abs(k_pred) < 0.001] = k_pred[torch.abs(k_pred) < 0.001] - k_pred[
            torch.abs(k_pred) < 0.001] - 0.001
        return k_pred
