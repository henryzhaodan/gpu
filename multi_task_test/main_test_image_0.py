from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import torch
import numpy as np
from configs.opts_test_0 import opts
from kernel.predictor.bbox_predictor import ModelInference
from kernel.utils.util import draw_axis, draw_marks, draw_gaze_angle

image_ext = ['jpg', 'jpeg', 'png', 'webp']


if __name__ == '__main__':
    opt = opts
    opt.device = torch.device('cuda: %d' % opt.gpus[0] if torch.cuda.is_available() else 'cpu')
    opt.img_dir = '/home/jerry/data/data/cv/gaze/DWK_gaze/images'
    gpus = ''
    for gpu_num in opt.gpus:
        gpus += '%d, ' % gpu_num
    gpus = gpus[:-2]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    bin_set_p = opt.bin_set_p
    opt.bins_p = np.array(range(bin_set_p[0], bin_set_p[1], bin_set_p[2]))
    opt.num_bins_p = opt.bins_p.shape[0] + 1
    bin_set_g = opt.bin_set_g
    opt.bins_g = np.array(range(bin_set_g[0], bin_set_g[1], bin_set_g[2]))
    opt.num_bins_g = opt.bins_g.shape[0] + 1

    model_inference = ModelInference(opt)

    window_name_1 = 'src'
    cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)

    image_files = []
    for main_dir, sub_dir, files in os.walk(opt.img_dir):
        for file in files:
            if file.split('.')[-1] in ['jpg', 'png']:
                image_files.append(os.path.join(main_dir, file))

    for image_file in image_files:
        image = cv2.imread(image_file)
        im_h, im_w, im_c = image.shape
        results = model_inference.run(image)
        confidences = results['confidence'][opt.use_stacks[0][0]]
        boxes = results['box'][opt.use_stacks[0][0]]
        headpose = results['headpose'][opt.use_stacks[1][0]]
        landmarks = results['landmark'][opt.use_stacks[2][0]]
        gaze = results['gaze'][opt.use_stacks[3][0]]
        inference_time = results['time']
        print(inference_time)

        for i in range(len(confidences)):
            inp_bbox = boxes[i].astype(np.int32)
            ptLeftTop = (inp_bbox[0], inp_bbox[1])
            ptRightBottom = (inp_bbox[2], inp_bbox[3])
            point_color = (0, 255, 0)  # BGR
            thickness = 3
            lineType = 4
            cv2.rectangle(image, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            text = "face: {:.3f}".format(confidences[i])
            cv2.putText(image, text, (ptLeftTop[0], ptLeftTop[1] - 5), cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5, color=(0, 0, 255), thickness=1)
            if len([task for task in opt.tasks if task in ["united", "headpose"]]) > 0:
                yaw, pitch, roll = headpose[i]
                ptCenter_x = (inp_bbox[0] + inp_bbox[2])//2
                ptCenter_y = (inp_bbox[1] + inp_bbox[3])//2
                draw_axis(image, yaw, pitch, roll, tdx=ptCenter_x, tdy=ptCenter_y, size=im_w / 8)
            if len([task for task in opt.tasks if task in ["united", "landmark"]]) > 0:
                draw_marks(image, landmarks[i], color=(0, 255, 0))
            if len([task for task in opt.tasks if task in ["united", "gaze"]]) > 0:
                yaw, pitch = gaze[i]
                ptCenter_x = (inp_bbox[0] + inp_bbox[2])//2
                ptCenter_y = (inp_bbox[1] + inp_bbox[3])//2
                draw_gaze_angle(image, (yaw, pitch), gaze_tdx=ptCenter_x, gaze_tdy=ptCenter_y, gaze_size=im_w / 8, color=(255, 0, 0))
                print('gaze: yaw %d, pitch %d' % (yaw, pitch))

        cv2.imshow(window_name_1, image)
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()
