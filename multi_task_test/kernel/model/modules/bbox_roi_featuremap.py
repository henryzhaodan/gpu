from torch import nn
import torch
import math
import numpy as np
from torchvision.ops import RoIAlign, roi_align
import torch.nn.functional as F
import cv2


class ROIPatch(nn.Module):

    def __init__(self, num_stacks):
        '''
        用一批数据ROI区域的最大宽高为H和W，补0，得到roi feature map
        '''
        super().__init__()
        self.num_stacks = num_stacks

    def forward(self, feature_maps, boxes):
        '''
        :param feature_map:  N * 128 * 128 * 32
        :param boxes: M * 8
        :return: N * H * W * C
        '''
        cropped_images_padded_stacks = []
        for n in range(self.num_stacks):
            feature_maps_i = feature_maps[n]
            max_width, max_hight = 0, 0
            boxes_w_h = []
            feature_rois = []
            for feature_map, boxes_i in zip(feature_maps_i, boxes[n]):
                for i, box in enumerate(boxes_i):
                    if torch.sum(box) == 0:
                        break
                    x1, y1, x2, y2 = box.cpu().numpy().astype(np.int)  # 128
                    box_w = x2 - x1
                    box_h = y2 - y1
                    max_width = box_w if box_w > max_width else max_width
                    max_hight = box_h if box_h > max_hight else max_hight

                    feature_rois.append(feature_map[:, y1:y2, x1:x2])
                    boxes_w_h.append([box_w, box_h])

            channels = feature_maps_i.shape[1]
            cropped_images_padded = torch.zeros((len(feature_rois), channels, max_hight, max_width),
                                                dtype=feature_maps_i.dtype,
                                                device=feature_maps_i.device)

            for i in range(cropped_images_padded.shape[0]):
                w, h = boxes_w_h[i]
                if max_width == w and max_hight == h:
                    cropped_images_padded[i] = feature_rois[i]
                else:
                    cropped_images_padded[i, :, 0:h, 0:w] = feature_rois[i]

            cropped_images_padded_stacks.append(cropped_images_padded)

        return cropped_images_padded_stacks


class ROIWarp(nn.Module):

    def __init__(self, num_stacks, size):
        '''
        通过插值缩放，得到指定的宽高为H和W的roi feature map
        '''
        super().__init__()
        self.num_stacks = num_stacks
        self.size = size

    def forward(self, feature_maps, boxes, has_box_indx):
        '''
        :param feature_map:  N * 128 * 128 * 32
        :param boxes: M * 8
        :return: N * H * W * C
        '''
        cropped_images_padded_stacks = []
        for n in range(self.num_stacks):
            feature_rois = []
            for index, (feature_map, boxes_i) in enumerate(zip(feature_maps[n], boxes[n])):
                if index in has_box_indx[n]:
                    for i, box in enumerate(boxes_i):
                        if torch.sum(box) == 0:
                            break
                        x1, y1, x2, y2 = box.cpu().numpy().astype(np.int)

                        feature_roi = feature_map[:, y1:y2, x1:x2]
                        feature_roi = feature_roi.unsqueeze(0)
                        feature_roi = F.interpolate(feature_roi, size=self.size, mode='bilinear', align_corners=False)
                        feature_roi = feature_roi.squeeze(0)
                        feature_rois.append(feature_roi)
            if len(feature_rois) > 0:
                feature_rois = torch.stack(feature_rois, dim=0)
            cropped_images_padded_stacks.append(feature_rois)

        return cropped_images_padded_stacks


class RROIMask(nn.Module):

    def __init__(self, num_stacks, size):
        '''
        固定128*128，ROI区域之外补0，得到roi feature map
        '''
        super().__init__()
        self.num_stacks = num_stacks
        self.weight = size[0]
        self.height = size[1]

    def forward(self, feature_maps, boxes):
        '''
        :param feature_map:  N * 128 * 128 * 32
        :param boxes: M * 8
        :return: N * H * W * C
        '''
        mask_images_padded_stacks = []
        for n in range(self.num_stacks):
            feature_maps_i = feature_maps[n]
            mask_images_padded = []
            for feature_map, boxes_i in zip(feature_maps_i, boxes[n]):
                width = feature_map.shape[2]
                height = feature_map.shape[1]
                for i, box in enumerate(boxes_i):
                    if torch.sum(box) == 0:
                        break
                    x1, y1, x3, y3 = box.cpu().numpy().astype(np.float32)
                    x2, y2 = x3, y1
                    x4, y4 = x1, y3
                    mapped_x1, mapped_y1 = (0, 0)
                    mapped_x2, mapped_y2 = (self.weight, 0)
                    mapped_x4, mapped_y4 = (0, self.height)
                    src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
                    dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
                    affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
                    feature_rotated = cv2.warpAffine(feature_map.permute(1, 2, 0).detach().cpu().numpy(), affine_matrix, (width, height))
                    feature_roi_i = feature_rotated.transpose(2, 0, 1)[:, 0:self.height, 0:self.weight]

                    mask_images_padded.append(torch.from_numpy(feature_roi_i).to(feature_maps_i.device))
            mask_images_padded = torch.stack(mask_images_padded, dim=0)
            mask_images_padded_stacks.append(mask_images_padded)

        return mask_images_padded_stacks


class ROIAlign(nn.Module):

    def __init__(self, num_stacks, size):
        '''
        用指定的宽高为H和W，通过roi align，得到roi feature map
        '''
        super().__init__()
        self.num_stacks = num_stacks
        self.size = size

    def forward(self, feature_maps, boxes):
        '''
        :param feature_map:  N * 128 * 128 * 32
        :param boxes: M * 8
        :return: N * H * W * C
        '''
        cropped_images_padded_stacks = []
        for n in range(self.num_stacks):
            feature_maps_i = feature_maps[n]
            boxes_use = []
            box_index = []
            index = 0
            for boxes_i in boxes:
                for i, box in enumerate(boxes_i):
                    if torch.sum(box) == 0:
                        break
                    boxes_use.append(box)
                    # x1, y1, x3, y3 = box.cpu().numpy().astype(np.float32)
                    box_index.append(index)
                    index += 1
            box_index = torch.from_numpy(np.array(box_index).astype(np.float32)).unsqueeze(1).cuda()
            boxes_use = torch.stack(boxes_use, dim=0)
            boxes_use = torch.cat((box_index, boxes_use), dim=1)
            feature_rois = roi_align(input=feature_maps_i, boxes=boxes_use, output_size=self.size, spatial_scale=1, sampling_ratio=-1)
            cropped_images_padded_stacks.append(feature_rois)

        return cropped_images_padded_stacks
