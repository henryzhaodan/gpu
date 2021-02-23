import os
CURRENT = os.path.split(os.path.abspath(__file__))[0]


class Opts(object):
    # -------test
    tasks = ['headpose', 'gaze']  # united, detection, headpose, landmark, gaze
    debug = 1  # level of visualization.
    # 1: only show the final detection results
    # 2: show the network output features
    # 3: show feature roi figure
    gpus = [0]
    classes_file = os.path.join(CURRENT, '../checkpoints/WIDER_classes.csv')
    model_file = os.path.join(CURRENT, '../checkpoints/bbox/headpose_gaze/hourglass2_2/model_best.pth')
    mean = [0.408, 0.447, 0.470]
    std = [0.289, 0.274, 0.278]
    fix_res = False  # fix testing resolution or keep the original resolution
    flip_test = False
    input_h = 512
    input_w = 512
    down_ratio = 4
    vis_thresh = 0.5
    mse_loss = True  # use mse loss or focal loss to train keypoint heatmaps.
    K = 10  # max number of output objects.
    nms = True  # run nms in testing.
    cat_spec_wh = False  # category specific bounding box size.
    pad = 127

    # shared backbone parameters
    arch = 'hourglass2'  # hourglass2 hourglass resnet50_fpn hopenet_resnet50 vovnet57
    num_stacks = 2  # 特征提取层输出多少张特征图用于后期
    use_stacks = [[0], [1], [1], [0]]  # [box, headpose, landmarks, gaze]

    # object box detector parameters
    num_classes = 1
    reg_offset = 2
    heads = {'hm': num_classes, 'wh': 1}
    if reg_offset > 0:
        heads.update({'reg': reg_offset})
    head_conv = 256

    # head pose estimator parameters
    roi_process_headpose = 'ROIWarp'  # ROIWarp ROIPatch RROIMask
    bin_set_p = [-170, 130, 3]
    heatmap_size_headpose = [64, 64]  # ROI heatmap图尺寸
    num_middle_headpose = 512
    is_attation_headpose = False  # 模型是否加入注意力机制

    # gaze estimator parameters
    roi_process_gaze = 'ROIWarp'  # ROIWarp ROIPatch RROIMask
    bin_set_g = [-90, 90, 3]
    heatmap_size_gaze = [64, 64]  # ROI heatmap图尺寸
    num_middle_gaze = 512
    is_attation_gaze = False  # 模型是否加入注意力机制

    # landmark estimator parameters
    roi_process_landmark = 'RROIMask'  # RROIMask ROIResize
    scale_factor_landmark = 0.625  # 调节特征点在heatmap图中的大小
    heatmap_size_landmark = [64, 64]  # heatmap图尺寸
    nparts = 12  # 特征点数量
    final_conv_kernel = 1
    STAGE2 = {'NUM_MODULES': 1,
              'NUM_BRANCHES': 2,
              'BLOCK': 'BASIC',
              'NUM_BLOCKS': [4, 4],
              'NUM_CHANNELS': [18, 36],
              'FUSE_METHOD': 'SUM'}
    STAGE3 = {'NUM_MODULES': 4,
              'NUM_BRANCHES': 3,
              'BLOCK': 'BASIC',
              'NUM_BLOCKS': [4, 4, 4],
              'NUM_CHANNELS': [18, 36, 72],
              'FUSE_METHOD': 'SUM'}
    STAGE4 = {'NUM_MODULES': 3,
              'NUM_BRANCHES': 4,
              'BLOCK': 'BASIC',
              'NUM_BLOCKS': [4, 4, 4, 4],
              'NUM_CHANNELS': [18, 36, 72, 144],
              'FUSE_METHOD': 'SUM'}
    stage_num = 2  # 2 3 4


opts = Opts()
