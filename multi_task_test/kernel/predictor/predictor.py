import torch
import numpy as np
import cv2
import time
import torch.nn.functional as F
from kernel.model.fbp_model import FBPModel
from kernel.utils.util import load_model
from kernel.utils.image import get_affine_transform
try:
    from kernel.utils.external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from kernel.model.detector.post_process import ctdet_post_process
from kernel.model.utils import generate_idx_tensor
from kernel.utils.image import transform_preds
from kernel.utils.decode_landmark import decode_landmarkmap, decode_transform_landmarkmap


class ModelInference(object):
    def __init__(self, opt):
        if len(opt.gpus) > 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = FBPModel(opt, is_train=False)
        self.model = load_model(self.model, opt.model_file, opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = opt.K
        self.num_classes = opt.num_classes
        self.opt = opt
        self.pause = True
        self.idx_tensor_p = generate_idx_tensor(opt.num_bins_p, opt.device)
        self.idx_tensor_g = generate_idx_tensor(opt.num_bins_g, opt.device)

    def pre_process(self, image):
        height, width = image.shape[0:2]
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (height | self.opt.pad) + 1  # 保证能被4整除
            inp_width = (width | self.opt.pad) + 1  # 保证能被4整除
            c = np.array([width // 2, height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        # resized_image = cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            if self.opt.device == torch.device('cuda'):
                torch.cuda.synchronize()
            first_time = time.time()
            detector_out, boxes_hm, headpose_out, landmark_out, gaze_out = self.model.forward(images)
            end_time = time.time()
            inference_time = end_time - first_time

        if return_time:
            return boxes_hm, headpose_out, landmark_out, gaze_out, inference_time
        else:
            return boxes_hm, headpose_out, landmark_out, gaze_out

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],
                                  meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detection):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = detection[j]
            if self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def run(self, im):
        images, meta = self.pre_process(im)
        images = images.to(self.opt.device)
        boxes_hm, headpose_out, landmark_out, gaze_out, inference_time = self.process(images, return_time=True)
        results = self.post_process_all(boxes_hm, meta, headpose_out, landmark_out, gaze_out)
        results['time'] = inference_time
        return results

    def post_process_all(self, boxes_hm, meta, headpose_out, landmark_out, gaze_out):
        thresholds = []
        boxes = []
        poses = []
        landmarks = []
        gazes = []
        centers = []
        scales = []
        for i in range(self.opt.num_stacks):
            boxes_hm_i = boxes_hm[i]
            if len([task for task in self.opt.tasks if task in ["united", "headpose"]]) > 0:
                headpose_out_i = headpose_out[i]
            if len([task for task in self.opt.tasks if task in ["united", "landmark"]]) > 0:
                landmark_out_i = landmark_out[i]
            if len([task for task in self.opt.tasks if task in ["united", "gaze"]]) > 0:
                gaze_out_i = gaze_out[i]
            if boxes_hm[i].shape[0] > 0:
                # decode boxes_hm out
                boxes_i = self.post_process_boxes(boxes_hm_i, meta)
                for j, bbox in enumerate(boxes_i):
                    thresholds.append(bbox[4])
                    boxes.append(bbox[0:4])
                    centers.append([(boxes_hm_i[j][0]+boxes_hm_i[j][2])/2, (boxes_hm_i[j][1]+boxes_hm_i[j][3])/2])
                    ratio = (self.opt.input_h//self.opt.down_ratio) / (boxes_hm_i[j][2]-boxes_hm_i[j][0])
                    scales.append(self.opt.scale_factor_landmark / ratio)
                    # decode head pose out
                    if len([task for task in self.opt.tasks if task in ["united", "headpose"]]) > 0:
                        if len(headpose_out[i]) == 3:
                            yaws_out, pitches_out, rolls_out = headpose_out_i
                            pose = self.post_process_pose(yaws_out[j], pitches_out[j], rolls_out[j])
                            poses.append(pose)
                    # decode gaze out
                    if len([task for task in self.opt.tasks if task in ["united", "gaze"]]) > 0:
                        if type(gaze_out[i]) is not list:
                            yaws_out, pitches_out, = gaze_out_i[:, :, 0], gaze_out_i[:, :, 1]
                            gaze = self.post_process_gaze(yaws_out[j], pitches_out[j])
                            gazes.append(gaze)
                # decode landmarks out
                if len([task for task in self.opt.tasks if task in ["united", "landmark"]]) > 0:
                    landmarks_i = self.pose_process_landmarks(landmark_out_i, meta, centers, scales)
                    landmarks.extend(landmarks_i)
                    if self.opt.debug == 3:
                        ##################################################
                        target = landmark_out_i.cpu().numpy()[0]  # 输出-面部特征点gauss特征图
                        target = target * 255.
                        target = np.clip(target, 0, 255).astype(np.uint8)
                        hm_landmark = np.zeros((target.shape[1], target.shape[2], 1), dtype=np.uint8)
                        for n in range(target.shape[0]):
                            if n == 0:
                                hm_landmark = target[n, :, :]
                            else:
                                hm_landmark += target[n, :, :]
                        for j in range(boxes_hm_i.shape[0]):
                            x1, y1, x2, y2 = boxes_hm_i[j][0:4]
                            cv2.rectangle(hm_landmark, (x1, y1), (x2, y2), (255, 255, 255), 1, 4)
                        window_name_1 = 'landmark heatmap'
                        cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
                        cv2.imshow(window_name_1, hm_landmark)
                        #################################################

        results = {'confidence': thresholds, 'box': boxes, 'headpose': poses, 'landmark': landmarks, 'gaze': gazes}
        return results

    def post_process_boxes(self, boxes_hm, meta):
        boxes_raw = boxes_hm.copy()
        boxes_raw[:, :2] = transform_preds(boxes_raw[:, 0:2], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))
        boxes_raw[:, 2:4] = transform_preds(boxes_raw[:, 2:4], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))
        return boxes_raw

    def post_process_pose(self, yaw_out, pitch_out, roll_out):
        yaw_predicted = F.softmax(yaw_out, dim=0)
        pitch_predicted = F.softmax(pitch_out, dim=0)
        roll_predicted = F.softmax(roll_out, dim=0)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor_p).cpu().numpy() * self.opt.bin_set_p[2] + \
                        self.opt.bin_set_p[0]
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor_p).cpu().numpy() * self.opt.bin_set_p[2] + \
                          self.opt.bin_set_p[0]
        roll_predicted = torch.sum(roll_predicted * self.idx_tensor_p).cpu().numpy() * self.opt.bin_set_p[2] + \
                         self.opt.bin_set_p[0]
        pose = [yaw_predicted, pitch_predicted, roll_predicted]

        return pose

    def post_process_gaze(self, yaw_out, pitch_out):
        yaw_predicted = F.softmax(yaw_out, dim=0)
        pitch_predicted = F.softmax(pitch_out, dim=0)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor_g).cpu().numpy() * self.opt.bin_set_g[2] + \
                        self.opt.bin_set_g[0]
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor_g).cpu().numpy() * self.opt.bin_set_g[2] + \
                          self.opt.bin_set_g[0]
        gaze = [yaw_predicted, pitch_predicted]

        return gaze

    def pose_process_landmarks(self, landmark_out, meta, centers, scales):
        score_map = landmark_out.data.cpu()
        landmarks = []
        # decode landmarks and transform to feature map coordinate
        if self.opt.roi_process_landmark == 'ROIMask':
            preds = decode_landmarkmap(score_map, [meta['out_height'], meta['out_width']])
        elif self.opt.roi_process_landmark == 'ROIResize':
            centers = np.array(centers, dtype=np.float32)
            scales = np.array(scales, dtype=np.float32)
            preds = decode_transform_landmarkmap(score_map, centers, scales, self.opt.heatmap_size_landmark)
        # transform landmarks from feature map coordinate
        preds = preds.numpy()
        for preds_i in preds:
            for i in range(self.opt.nparts):
                preds_i[i:i+1, :] = transform_preds(preds_i[i:i+1, :], meta['c'], meta['s'],
                                                    (meta['out_width'], meta['out_height']))
            landmarks.append(preds_i)
        return landmarks
