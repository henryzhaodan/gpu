from ..feature_pyramid_network import LastLevelMaxPool, BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from ..base_net import resnet
from torch import nn


def resnet_fpn_backbone(backbone_name, fpn_function, pretrained,
                        up_sample_num=3, input_channels=3, out_channels=256, is_FrozenNorm=False,
                        is_extra_blocks=False):
    if is_FrozenNorm:
        norm_layer = misc_nn_ops.FrozenBatchNorm2d
    else:
        norm_layer = None
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
    # print(backbone)
    # exit()
    backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)
    return_layers, in_channels_list = get_up_sample_params(backbone_name, up_sample_num)
    if return_layers is None or in_channels_list is None:
        print("error up_sample_num: ", up_sample_num)
        exit()
    if is_extra_blocks:
        extra_blocks = LastLevelMaxPool()
    else:
        extra_blocks = None
    backbone_fpn = BackboneWithFPN(backbone,
                                   fpn_function,
                                   return_layers,
                                   in_channels_list,
                                   out_channels,
                                   extra_blocks)
    return backbone_fpn


def get_up_sample_params(backbone_name, up_sample_num):
    return_layers = None
    in_channels_list = None

    if backbone_name in ['resnet18', 'resnet34']:
        in_channels_stage2 = 64
        if up_sample_num == 3:
            return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2}
            in_channels_list = [
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
            ]
        elif up_sample_num == 2:
            return_layers = {'layer2': 0, 'layer3': 1}
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
            ]
        elif up_sample_num == 1:
            return_layers = {'layer3': 0}
            in_channels_list = [
                in_channels_stage2 * 4,
            ]
    elif backbone_name in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']:
        in_channels_stage2 = 256
        if up_sample_num == 4:
            return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
            in_channels_list = [
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
        if up_sample_num == 3:
            return_layers = {'layer2': 0, 'layer3': 1, 'layer4': 2}
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
        elif up_sample_num == 2:
            return_layers = {'layer1': 0, 'layer2': 1}
            in_channels_list = [
                in_channels_stage2 * 1,
                in_channels_stage2 * 2,
            ]
        elif up_sample_num == 1:
            return_layers = {'layer1': 0}
            in_channels_list = [
                in_channels_stage2 * 1,
            ]
    else:
        print("error backbone_name: ", backbone_name)
        exit()
    return return_layers, in_channels_list