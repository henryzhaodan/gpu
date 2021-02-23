from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2, time
import torch
import numpy as np
from configs.opts_test_0 import opts
from kernel.predictor.bbox_predictor import ModelInference
from kernel.utils.util import draw_axis, draw_marks, draw_gaze_angle


if __name__ == '__main__':
    opt = opts
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bin_set_p = opt.bin_set_p
    opt.bins_p = np.array(range(bin_set_p[0], bin_set_p[1], bin_set_p[2]))
    opt.num_bins_p = opt.bins_p.shape[0] + 1
    bin_set_g = opt.bin_set_g
    opt.bins_g = np.array(range(bin_set_g[0], bin_set_g[1], bin_set_g[2]))
    opt.num_bins_g = opt.bins_g.shape[0] + 1

    model_inference = ModelInference(opt)

    namewindows = 'face detect and analysis'
    cv2.namedWindow(namewindows, cv2.WINDOW_NORMAL)
    video = cv2.VideoCapture(0)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    print('frame size: %d, %d' % (width, height))
    colors = [(0, 255, 0), (0, 0, 255)]

    num = 0
    fps_list = []
    while True:
        start = time.time()
        ret, frame = video.read()
        if not ret:
            break
        image = frame
        num += 1
        if num % 1 == 0:  # 每多少帧分析一帧
            im_h, im_w, im_c = image.shape
            results = model_inference.run(image)
            confidences = results['confidence'][opt.use_stacks[0][0]]
            boxes = results['box'][opt.use_stacks[0][0]]
            headpose = results['headpose'][opt.use_stacks[1][0]]
            landmarks = results['landmark'][opt.use_stacks[2][0]]
            gaze = results['gaze'][opt.use_stacks[3][0]]
            inference_time = results['time']

            for i in range(len(confidences)):
                inp_bbox = boxes[i]
                ptLeftTop = (inp_bbox[0], inp_bbox[1])
                ptRightBottom = (inp_bbox[2], inp_bbox[3])
                point_color = (0, 255, 0)  # BGR
                thickness = 3
                lineType = 4
                cv2.rectangle(image, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
                if len([task for task in opt.tasks if task in ["united", "headpose"]]) > 0:
                    yaw, pitch, roll = headpose[i]
                    ptCenter_x = (inp_bbox[0] + inp_bbox[2]) // 2
                    ptCenter_y = (inp_bbox[1] + inp_bbox[3]) // 2
                    draw_axis(image, yaw, pitch, roll, tdx=ptCenter_x, tdy=ptCenter_y, size=im_w / 8)
                if len([task for task in opt.tasks if task in ["united", "landmark"]]) > 0:
                    draw_marks(image, landmarks[i], color=(0, 255, 0))
                if len([task for task in opt.tasks if task in ["united", "gaze"]]) > 0:
                    yaw, pitch = gaze[i]
                    ptCenter_x = (inp_bbox[0] + inp_bbox[2]) // 2
                    ptCenter_y = (inp_bbox[1] + inp_bbox[3]) // 2
                    draw_gaze_angle(image, (yaw, pitch), gaze_tdx=ptCenter_x, gaze_tdy=ptCenter_y, gaze_size=im_w / 8,
                                    color=(255, 0, 0))
                    # print('gaze: yaw %d, pitch %d' % (yaw, pitch))

        end = time.time()
        seconds = end - start
        fps_list.append(1 / seconds)
        if len(fps_list) < 100:
            fps = 0.0
        else:
            fps = np.sum(fps_list) / 100
            fps_list = fps_list[1:]
        cv2.putText(frame, '%.2f' % fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0, 0, 255), thickness=2)
        cv2.imshow(namewindows, image)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
