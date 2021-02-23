#coding=utf-8
import torch
import torch.nn as nn
from kernel.model.shared_net.feature_pyramid_network import FPN_1, FPN_2
from kernel.model.shared_net.shared_backbone import get_backbone


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


class DetectorHM(nn.Module):
    def __init__(self, heads, nFeats=256, _nStack=1):
        """
        输入： 256^2
        """
        super(DetectorHM, self).__init__()
        self.heads = heads
        self._nFeats = nFeats
        self._nStack = _nStack
        # self.backbone = backbone
        self._init_heatmap()

    def _init_heatmap(self):
        ## keypoint heatmaps
        make_heat_layer = make_kp_layer
        make_regr_layer = make_kp_layer
        cnv_dim = self._nFeats
        curr_dim = 256
        for head in self.heads.keys():
            if 'hm' in head:
                module = nn.ModuleList([
                    make_heat_layer(cnv_dim, curr_dim, self.heads[head]) for _ in range(self._nStack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(cnv_dim, curr_dim, self.heads[head]) for _ in range(self._nStack)
                ])
                self.__setattr__(head, module)

    def forward(self, x):
        # x = self.backbone(x)
        out = []

        for i in range(self._nStack):
            ll = x[i]
            # Predicted heatmaps
            tmpOut = {}
            for head in self.heads:
                layer = self.__getattr__(head)[i]
                y = layer(ll)
                tmpOut[head] = y
            out.append(tmpOut)

        return out


class DetectorHMOnnx(nn.Module):
    def __init__(self, heads, nFeats=256, _nStack=1):
        """
        输入： 256^2
        """
        super(DetectorHMOnnx, self).__init__()
        self.heads = heads
        self._nFeats = nFeats
        self._nStack = _nStack
        # self.backbone = backbone
        self._init_heatmap()
        self.heads_dic = {'hm': 0, 'wh': 1, 'reg': 2}

    def _init_heatmap(self):
        ## keypoint heatmaps
        make_heat_layer = make_kp_layer
        make_regr_layer = make_kp_layer
        cnv_dim = self._nFeats
        curr_dim = 256
        for head in self.heads.keys():
            if 'hm' in head:
                module = nn.ModuleList([
                    make_heat_layer(cnv_dim, curr_dim, self.heads[head]) for _ in range(self._nStack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(cnv_dim, curr_dim, self.heads[head]) for _ in range(self._nStack)
                ])
                self.__setattr__(head, module)

    def forward(self, x):
        # x = self.backbone(x)
        out = []

        for i in range(self._nStack):
            ll = x[i]
            # Predicted heatmaps
            tmpOut = list()
            for head in self.heads:
                layer = self.__getattr__(head)[i]
                y = layer(ll)
                # tmpOut[self.heads_dic[head]] = y
                tmpOut.append(y)
            out.append(tmpOut)

        return out


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import torch.optim as optim

    class tempDataset(Dataset):
        def __init__(self, input_size, num_classes):
            self.X = np.random.randn(3, input_size[0], input_size[1]).astype(np.float32)
            y1 = np.random.randn(num_classes, 128, 128).astype(np.float32)
            y2 = np.random.randn(2, 128, 128).astype(np.float32)
            y3 = np.random.randn(2, 128, 128).astype(np.float32)
            self.Y = [y1, y2, y3]

        def __len__(self):
            return 1000

        def __getitem__(self, item):
            inp = self.X
            hm = self.Y[0]
            wh = self.Y[1]
            reg = self.Y[2]
            ret = {'hm': hm, 'wh': wh, 'reg': reg}
            return inp, ret

    backbone_name = "hourglass2"
    input_h, input_w = 512, 512
    num_classes = 21
    batch_size = 2
    nFeats = 256
    up_sample_num = 2
    heads = {'hm': num_classes, 'wh': 2, 'reg': 2}

    dataset = tempDataset(input_size=(input_h, input_w), num_classes=num_classes)
    dataLoader = DataLoader(dataset=dataset, batch_size=batch_size)

    backbone_net = get_backbone(backbone_name,
                            up_sample_num=up_sample_num,
                            input_channels=3,
                            fpn_out_channels=nFeats,
                            num_classes=num_classes,
                            fpn_function=FPN_2,
                            pretrained=True)
    net = DetectorHM(heads=heads, nFeats=nFeats, _nStack=up_sample_num)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    for i, (x, y) in enumerate(dataLoader):
        x = backbone_net(x)
        y_pred = net.forward(x)
        pass
