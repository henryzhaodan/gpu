import torch
from torch.autograd import Variable
import os
import numpy as np
from configs.opts_test_0 import opts
from kernel.model.fbp_model_onnx import FBPModelOnnx


if __name__ == '__main__':
    input_model_file = '../checkpoints/hourglass2/headpose_gaze/model_best.pth'
    output_model_dir = '../checkpoints/hourglass2/headpose_gaze/onnx'
    input_size = (640, 480)

    if input_size == (640, 480):
        sharedConv_input_size = (1, 3, 512, 768)
        boxDetector_input_sizes = [(1, 256, 128, 192)]
        headPoseEvaluator_input_size = (1, 256, 64, 64)
        landmarkEvaluator_input_size = (1, 256, 64, 64)
        gazeEvaluator_input_size = (1, 256, 64, 64)

    if not os.path.exists(output_model_dir):
        os.mkdir(output_model_dir)
    opt = opts
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bin_set_p = opt.bin_set_p
    opt.bins_p = np.array(range(bin_set_p[0], bin_set_p[1], bin_set_p[2]))
    opt.num_bins_p = opt.bins_p.shape[0] + 1
    bin_set_g = opt.bin_set_g
    opt.bins_g = np.array(range(bin_set_g[0], bin_set_g[1], bin_set_g[2]))
    opt.num_bins_g = opt.bins_g.shape[0] + 1

    model = FBPModelOnnx(opt, is_train=False)
    print(model)
    state_dict = torch.load(input_model_file)
    model.load_state_dict(state_dict)

    dummy_input = torch.randn(sharedConv_input_size[0], sharedConv_input_size[1],
                              sharedConv_input_size[2], sharedConv_input_size[3])
    output_sharedConv_file = os.path.join(output_model_dir, 'sharedConv.onnx')
    torch.onnx.export(model.sharedConv, dummy_input, output_sharedConv_file)
    dummy_inputs = []
    for i, boxDetector_input_size in enumerate(boxDetector_input_sizes):
        dummy_input = torch.randn(boxDetector_input_size[0], boxDetector_input_size[1],
                                  boxDetector_input_size[2], boxDetector_input_size[3])
        dummy_inputs.append(dummy_input)
    output_boxDetector_file = os.path.join(output_model_dir, 'boxDetector.onnx')
    torch.onnx.export(model.boxDetector, dummy_inputs, output_boxDetector_file)

    if len([task for task in opt.tasks if task in ["united", "headpose"]]) > 0:
        dummy_input = torch.randn(headPoseEvaluator_input_size[0], headPoseEvaluator_input_size[1],
                                  headPoseEvaluator_input_size[2], headPoseEvaluator_input_size[3])
        output_headPoseEvaluator_file = os.path.join(output_model_dir, 'headPoseEvaluator.onnx')
        torch.onnx.export(model.headPoseEvaluator, dummy_input, output_headPoseEvaluator_file)
    if len([task for task in opt.tasks if task in ["united", "landmark"]]) > 0:
        dummy_input = torch.randn(landmarkEvaluator_input_size[0], landmarkEvaluator_input_size[1],
                                  landmarkEvaluator_input_size[2], landmarkEvaluator_input_size[3])
        output_landmarkEvaluator_file = os.path.join(output_model_dir, 'landmarkEvaluator.onnx')
        torch.onnx.export(model.landmarkEvaluator, dummy_input, output_landmarkEvaluator_file)
    if len([task for task in opt.tasks if task in ["united", "gaze"]]) > 0:
        dummy_input = torch.randn(gazeEvaluator_input_size[0], gazeEvaluator_input_size[1],
                                  gazeEvaluator_input_size[2], gazeEvaluator_input_size[3])
        output_gazeEvaluator_file = os.path.join(output_model_dir, 'gazeEvaluator.onnx')
        torch.onnx.export(model.gazeEvaluator, dummy_input, output_gazeEvaluator_file)
