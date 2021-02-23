#coding=utf-8
import torch
import torch.nn as nn
from ..shared_net.feature_pyramid_network import FPN_1, FPN_2
from ..shared_net.fpn.backbone_resnet_fpn import resnet_fpn_backbone
from ..shared_net.base_net.hourglass import StackedHourGlass


def get_backbone(backbone_name, input_channels, up_sample_num, fpn_out_channels, num_classes=1,
                 fpn_function=None, pretrained=True):
    backbone = None
    if fpn_function is None:
        fpn_function = FPN_1
    if backbone_name == "resnet50_fpn":
        backbone = resnet_fpn_backbone('resnet50',
                                       fpn_function=fpn_function,
                                       pretrained=pretrained,
                                       up_sample_num=up_sample_num,
                                       input_channels=input_channels,
                                       out_channels=fpn_out_channels,
                                       is_FrozenNorm=True,
                                       is_extra_blocks=False)
    elif backbone_name == "resnet101_fpn":
        backbone = resnet_fpn_backbone('resnet101',
                                       fpn_function=fpn_function,
                                       pretrained=pretrained,
                                       up_sample_num=up_sample_num,
                                       input_channels=input_channels,
                                       out_channels=fpn_out_channels,
                                       is_FrozenNorm=True,
                                       is_extra_blocks=False)
    elif backbone_name == "resnet152_fpn":
        backbone = resnet_fpn_backbone('resnet152',
                                       fpn_function=fpn_function,
                                       pretrained=pretrained,
                                       up_sample_num=up_sample_num,
                                       input_channels=input_channels,
                                       out_channels=fpn_out_channels,
                                       is_FrozenNorm=True,
                                       is_extra_blocks=False)
    elif backbone_name == "resnext50_fpn":
        backbone = resnet_fpn_backbone('resnext50_32x4d',
                                       fpn_function=fpn_function,
                                       pretrained=pretrained,
                                       up_sample_num=up_sample_num,
                                       input_channels=input_channels,
                                       out_channels=fpn_out_channels,
                                       is_FrozenNorm=False,
                                       is_extra_blocks=False)
    elif backbone_name == "resnext101_fpn":
        backbone = resnet_fpn_backbone('resnext101_32x8d',
                                       fpn_function=fpn_function,
                                       pretrained=pretrained,
                                       up_sample_num=up_sample_num,
                                       input_channels=input_channels,
                                       out_channels=fpn_out_channels,
                                       is_FrozenNorm=False,
                                       is_extra_blocks=False)
    elif backbone_name == "hourglass2":
        backbone = StackedHourGlass(nFeats=fpn_out_channels,
                                    nStack=up_sample_num,
                                    nJoints=num_classes)
    else:
        print('backbone_name is error: ', backbone_name)
        exit()
    return backbone
