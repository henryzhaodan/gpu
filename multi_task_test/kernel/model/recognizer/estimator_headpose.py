from kernel.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from kernel.model.utils import l2_norm


class LineLayer(nn.Module):
    def __init__(self, num_input, num_bins, num_middle=1024):
        super(LineLayer, self).__init__()
        self.conv1 = nn.Conv2d(num_input, num_middle, 1)
        self.adtavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(num_middle, num_bins, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.adtavgpool(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x


class EstimatorHeadPose(nn.Module):

    def __init__(self, opt, criterion=None, is_train=False):
        super(EstimatorHeadPose, self).__init__()
        input_channel = opt.head_conv
        num_bins = opt.num_bins_p
        num_middle = opt.num_middle_headpose
        self.is_attation = opt.is_attation_headpose

        self.conv = nn.Conv2d(input_channel,
                              1,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.fc_yaw = LineLayer(input_channel, num_bins, num_middle)
        self.fc_pitch = LineLayer(input_channel, num_bins, num_middle)
        self.fc_roll = LineLayer(input_channel, num_bins, num_middle)

        self.softmax = nn.Softmax(dim=1)
        idx_tensor = [idx for idx in range(num_bins)]
        self.idx_tensor = torch.from_numpy(np.array(idx_tensor).astype(np.float32))
        self.bin_set = opt.bin_set_p

    def forward(self, x, targets=None):
        if self.is_attation:
            y = F.relu(self.conv(x))
            x = x * y
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)
        out = [pre_yaw, pre_pitch, pre_roll]
        return out

