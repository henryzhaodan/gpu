import torch
import cv2
from .modules.roi_featuremap import ROIPatch, ROIResize, ROIMask
import torch.optim as optim
import numpy as np
from .detector.detector_heatmap import DetectorHMOnnx
from .recognizer.estimator_headpose import EstimatorHeadPose
from .recognizer.estimator_landmark_hr import EstimatorLandmark
from .recognizer.estimator_gaze import EstimatorGaze
from .shared_net.shared_backbone import get_backbone
# from .shared_net.feature_pyramid_network import FPN_1, FPN_2
from .detector.utils import flip_tensor
from .detector.decode import ctdet_decode
try:
    from kernel.utils.external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')


class FBPModelOnnx(object):

    def __init__(self, config, is_train=True):
        self.config = config
        # self.mode = config.mode
        self.is_train = is_train
        self.sharedConv = get_backbone(config.arch,
                                       up_sample_num=config.num_stacks,
                                       input_channels=3,
                                       fpn_out_channels=config.head_conv,
                                       num_classes=config.num_classes,
                                       pretrained=False)
        self.boxDetector = DetectorHMOnnx(heads=config.heads, nFeats=config.head_conv, _nStack=config.num_stacks)
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            if self.config.roi_process_headpose == 'ROIPatch':
                self.roi_headpose = ROIPatch(num_stacks=config.num_stacks)
            elif self.config.roi_process_headpose == 'ROIResize':
                self.roi_headpose = ROIResize(num_stacks=config.num_stacks,
                                              size=config.heatmap_size_landmark)
            self.headPoseEvaluator = EstimatorHeadPose(config.head_conv, config.num_bins_p, num_middle=512)
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            if self.config.roi_process_landmark == 'ROIMask':
                self.roi_landmark = ROIMask(num_stacks=config.num_stacks)
            elif self.config.roi_process_landmark == 'ROIResize':
                self.roi_landmark = ROIResize(num_stacks=config.num_stacks,
                                              size=config.heatmap_size_landmark)
            # self.landmarkEvaluator = EstimatorLandmark(input_channel=256,
            #                                            final_conv_kernel=config.final_conv_kernel,
            #                                            nparts=config.nparts)
            self.landmarkEvaluator = EstimatorLandmark(config)
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            if self.config.roi_process_gaze == 'ROIPatch':
                self.roi_gaze = ROIPatch(num_stacks=config.num_stacks)
            elif self.config.roi_process_gaze == 'ROIResize':
                self.roi_gaze = ROIResize(num_stacks=config.num_stacks,
                                          size=config.heatmap_size_landmark)
            self.gazeEvaluator = EstimatorGaze(config.head_conv, config.num_bins_g, num_middle=512)

    def parallelize(self):
        self.sharedConv = torch.nn.DataParallel(self.sharedConv)
        self.boxDetector = torch.nn.DataParallel(self.boxDetector)
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            self.headPoseEvaluator = torch.nn.DataParallel(self.headPoseEvaluator)
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            self.landmarkEvaluator = torch.nn.DataParallel(self.landmarkEvaluator)
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            self.gazeEvaluator = torch.nn.DataParallel(self.gazeEvaluator)

    def to(self, device):
        self.sharedConv = self.sharedConv.to(device)
        self.boxDetector = self.boxDetector.to(device)
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            self.headPoseEvaluator = self.headPoseEvaluator.to(device)
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            self.landmarkEvaluator = self.landmarkEvaluator.to(device)
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            self.gazeEvaluator = self.gazeEvaluator.to(device)

    def summary(self):
        self.sharedConv.summary()
        self.boxDetector.summary()
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            self.headPoseEvaluator.summary()
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            self.landmarkEvaluator.summary()
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            self.gazeEvaluator.summary()

    def optimize(self, optimizer_type, params):
        optimizer_list = [{'params': self.sharedConv.parameters(), 'lr': params['lr'][0]},
                          {'params': self.boxDetector.parameters(), 'lr': params['lr'][1]}]
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            optimizer_list.append({'params': self.headPoseEvaluator.parameters(), 'lr': params['lr'][2]})
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            optimizer_list.append({'params': self.landmarkEvaluator.parameters(), 'lr': params['lr'][3]})
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            optimizer_list.append({'params': self.gazeEvaluator.parameters(), 'lr': params['lr'][4]})
        # optimizer = getattr(optim, optimizer_type)(optimizer_list, **params)
        # if optimizer_type == 'Adam':
        #     optimizer = torch.optim.Adam([
        #         {'params': self.sharedConv.parameters(), 'lr': 0.01},
        #         {'params': self.boxDetector.parameters(), 'lr': 0.1},
        #         {'params': self.gazeEvaluator.parameters()},
        #     ], params['lr'])
        optimizer = getattr(optim, optimizer_type)(optimizer_list)
        return optimizer

    def train(self):
        self.sharedConv.train()
        self.boxDetector.train()
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            self.headPoseEvaluator.train()
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            self.landmarkEvaluator.train()
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            self.gazeEvaluator.train()

    def eval(self):
        self.sharedConv.eval()
        self.boxDetector.eval()
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            self.headPoseEvaluator.eval()
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            self.landmarkEvaluator.eval()
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            self.gazeEvaluator.eval()

    def state_dict(self):
        sd = {
            '0': self.sharedConv.state_dict(),
            '1': self.boxDetector.state_dict()
        }
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
            sd['2'] = self.headPoseEvaluator.state_dict()
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
            sd['3'] = self.landmarkEvaluator.state_dict()
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
            sd['4'] = self.gazeEvaluator.state_dict()
        return sd

    def load_state_dict(self, sd):
        self.sharedConv.load_state_dict(sd['0'])
        self.boxDetector.load_state_dict(sd['1'])
        if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0 and '2' in sd:
            self.headPoseEvaluator.load_state_dict(sd['2'])
        if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0 and '3' in sd:
            self.landmarkEvaluator.load_state_dict(sd['3'])
        if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0 and '4' in sd:
            self.gazeEvaluator.load_state_dict(sd['4'])

    def forward(self, batch):
        '''
        :param input:
        :return:
        '''
        if self.is_train:
            feature_map = self.sharedConv.forward(batch['input'])
        else:
            feature_map = self.sharedConv.forward(batch)
        detector_out = self.boxDetector(feature_map)

        headpose_out, landmark_out, gaze_out = [], [], []
        if len([task for task in self.config.tasks if task in ["united", "headpose", "landmark", "gaze"]]) > 0:
            if self.is_train:
                boxes = []
                for n in range(self.config.num_stacks):
                    boxes.append(batch['box'])
            else:
                boxes_hm, boxes = [], []
                for n in range(self.config.num_stacks):
                    output = detector_out[n]
                    hm = output[0].clone().sigmoid_()
                    wh = output[1].clone()
                    reg = output[2].clone() if self.config.reg_offset else None
                    if self.config.flip_test:
                        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                        reg = reg[0:1] if reg is not None else None
                    dets = ctdet_decode(hm, wh, reg, cat_spec_wh=self.config.cat_spec_wh, K=self.config.K)
                    # post_process
                    dets = dets.detach().cpu().numpy()
                    dets = dets.reshape(1, -1, dets.shape[2])
                    dets_new = []
                    for i in range(dets.shape[0]):
                        top_preds = {}
                        classes = dets[i, :, -1]
                        for j in range(self.config.num_classes):
                            inds = (classes == j)
                            top_preds[j + 1] = np.concatenate([
                                dets[i, inds, :4].astype(np.float32),
                                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
                        dets_new.append(top_preds)
                    for j in range(1, self.config.num_classes + 1):
                        dets_new[0][j] = np.array(dets_new[0][j], dtype=np.float32).reshape(-1, 5)
                    detection = dets_new[0]
                    # merge_outputs
                    results = {}
                    for j in range(1, self.config.num_classes + 1):
                        results[j] = detection[j]
                        if self.config.nms:
                            soft_nms(results[j], Nt=0.5, method=2)
                    scores = np.hstack([results[j][:, 4] for j in range(1, self.config.num_classes + 1)])
                    if len(scores) > self.config.K:
                        kth = len(scores) - self.config.K
                        thresh = np.partition(scores, kth)[kth]
                        for j in range(1, self.config.num_classes + 1):
                            keep_inds = (results[j][:, 4] >= thresh)
                            results[j] = results[j][keep_inds]

                    boxes_hm_n = []
                    hm_h, hm_w = output[0].shape[2:]
                    for result_i in results[1]:  # 目前仅第一类目标face做分析
                        if result_i[-1] > self.config.vis_thresh:
                            result_i[0] = max(0, result_i[0])
                            result_i[1] = max(0, result_i[1])
                            result_i[2] = min(hm_w, result_i[2])
                            result_i[3] = min(hm_h, result_i[3])
                            boxes_hm_n.append(result_i[0:5])
                    boxes_hm_n = np.array(boxes_hm_n)
                    if boxes_hm_n.shape[0] > 0:
                        boxes_n = boxes_hm_n[:, 0:4]
                    else:
                        boxes_n = boxes_hm_n
                    boxes_n = torch.from_numpy(boxes_n).unsqueeze(dim=0).to(self.config.device)
                    boxes_hm.append(boxes_hm_n)
                    boxes.append(boxes_n)
                # if self.config.num_stacks > 1:  # 合并所有堆叠层的输出，如果仅一层就使用第一层堆叠输出
                #     boxes = np.concatenate(boxes_hm)
                #     soft_nms(boxes, Nt=0.5, method=2)
                #     boxes = torch.from_numpy(boxes[:, np.newaxis, :-1])
                # else:
                #     boxes = boxes[0]
            if boxes[0].shape[1] != 0:
                if len([task for task in self.config.tasks if task in ["united", "headpose"]]) > 0:
                    rois_stacks = self.roi_headpose(feature_map, boxes)
                    for n in range(self.config.num_stacks):
                        rois = rois_stacks[n]
                        headpose_out_i = self.headPoseEvaluator(rois)
                        headpose_out.append(headpose_out_i)
                        if self.config.debug == 3:
                            ##################################################
                            target = rois.detach().cpu().numpy()[0]
                            target = target * 255.
                            target = np.clip(target, 0, 255).astype(np.uint8)
                            hm_headpose = np.zeros((target.shape[1], target.shape[2], 1), dtype=np.uint8)
                            for n in range(target.shape[0]):
                                if n == 0:
                                    hm_headpose = target[n, :, :]
                                else:
                                    hm_headpose += target[n, :, :]
                            window_name_1 = '1'
                            cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
                            cv2.imshow(window_name_1, hm_headpose)
                            cv2.waitKey(100)
                            #################################################
                if len([task for task in self.config.tasks if task in ["united", "landmark"]]) > 0:
                    rois_stacks = self.roi_landmark(feature_map, boxes)
                    for n in range(self.config.num_stacks):
                        rois = rois_stacks[n]
                        landmark_out_i = self.landmarkEvaluator(rois)
                        landmark_out.append(landmark_out_i)
                        if self.config.debug == 3:
                            ##################################################
                            target = rois.detach().cpu().numpy()[0]
                            target = target * 255.
                            target = np.clip(target, 0, 255).astype(np.uint8)
                            hm_landmark = np.zeros((target.shape[1], target.shape[2], 1), dtype=np.uint8)
                            for n in range(target.shape[0]):
                                if n == 0:
                                    hm_landmark = target[n, :, :]
                                else:
                                    hm_landmark += target[n, :, :]
                            window_name_2 = '2'
                            cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
                            cv2.imshow(window_name_2, hm_landmark)
                            cv2.waitKey(100)
                            #################################################
                if len([task for task in self.config.tasks if task in ["united", "gaze"]]) > 0:
                    rois_stacks = self.roi_gaze(feature_map, boxes)
                    for n in range(self.config.num_stacks):
                        rois = rois_stacks[n]
                        gaze_out_i = self.gazeEvaluator(rois)
                        gaze_out.append(gaze_out_i)
                        if self.config.debug == 3:
                            ##################################################
                            target = rois.detach().cpu().numpy()[0]
                            target = target * 255.
                            target = np.clip(target, 0, 255).astype(np.uint8)
                            hm_gaze = np.zeros((target.shape[1], target.shape[2], 1), dtype=np.uint8)
                            for n in range(target.shape[0]):
                                if n == 0:
                                    hm_gaze = target[n, :, :]
                                else:
                                    hm_gaze += target[n, :, :]
                            window_name_3 = '3'
                            cv2.namedWindow(window_name_3, cv2.WINDOW_NORMAL)
                            cv2.imshow(window_name_3, hm_gaze)
                            cv2.waitKey(100)
                            #################################################
            else:
                for n in range(self.config.num_stacks):
                    headpose_out.append([])
                    landmark_out.append([])
                    gaze_out.append([])

        if self.is_train:
            return detector_out, headpose_out, landmark_out, gaze_out
        else:
            return detector_out, boxes_hm, headpose_out, landmark_out, gaze_out
