import argparse
import os
import random
import sys

import numba
import numpy as np

sys.path.append(os.getcwd())

__all__ = ['evaluate']

def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    args, opts = parser.parse_known_args()
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    # define save path
    save_path = get_save_path(*args.configs, prefix='runs')
    os.makedirs(save_path, exist_ok=True)
    configs.train.save_path = save_path
    configs.train.best_checkpoint_path = os.path.join(save_path, 'best.pth.tar')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    configs.dataset.split = configs.evaluate.dataset.split
    if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
        if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
            configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
        else:
            configs.evaluate.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
    assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
    configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.eval.npy')

    return configs


def evaluate(configs=None):
    configs = prepare() if configs is None else configs

    import math
    import torch
    import torch.backends.cudnn as cudnn
    import torch.nn.functional as F
    from tqdm import tqdm
    import time

    from meters.shapenet import MeterShapeNet

    ###########
    # Prepare #
    ###########

    if configs.device == 'cuda':
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False
    if ('seed' not in configs) or (configs.seed is None):
        configs.seed = torch.initial_seed() % (2 ** 32 - 1)
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    import logging
    def log_string(str):
        logger.info(str)
        print(str)

    # 创建一个Logger
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(os.path.join(configs.train.save_path, 'test.log'))
    fh.setLevel(logging.INFO)
    # 定义hander的输出格式（formatter）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 给handler添加formatter
    fh.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    log_string('PARAMETER ...')
    log_string(configs)
    #################################
    # Initialize DataLoaders, Model #
    #################################

    log_string(f'\n==> loading dataset "{configs.dataset}"')
    dataset = configs.dataset()[configs.dataset.split]
    log_string("The number of evaluate data is: %d" % len(dataset))
    meter = MeterShapeNet()

    log_string(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)

    if os.path.exists(configs.evaluate.best_checkpoint_path):
        log_string(f'==> loading checkpoint "{configs.evaluate.best_checkpoint_path}"')
        checkpoint = torch.load(configs.evaluate.best_checkpoint_path)
        model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    model.eval()
    if configs.visual:
        shapenet_color_classes = { 0:	[255,127,80],
                                1:	[0,255,127],
                                2:	[147,112,219],
                                3:  [218,112,214],
                                4:  [135,206,235],
                                5:  [255,215,0],
                                6:  [119,136,153],
                                7:  [102,205,170],
                                8:  [0,0,255],
                                9:  [188,143,143],
                                10: [255,0,0]}
        from pathlib import Path
        visual_dir = Path('visual_out_shapenet\\')
        visual_dir.mkdir(exist_ok=True)
    ##############
    # Evaluation #
    ##############

    stats = np.zeros((configs.data.num_shapes, 2))
    since = time.time() # add
    for shape_index, (file_path, shape_id) in enumerate(tqdm(dataset.file_paths, desc='eval', ncols=0)):
        data = np.loadtxt(file_path).astype(np.float32)
        total_num_points_in_shape = data.shape[0]
        confidences = np.zeros(total_num_points_in_shape, dtype=np.float32)
        predictions = np.full(total_num_points_in_shape, -1, dtype=np.int64)
        coords = data[:, :3]
        if dataset.normalize:
            coords = dataset.normalize_point_cloud(coords)
        coords = coords.transpose()
        ground_truth = data[:, -1].astype(np.int64)
        if dataset.with_normal:
            normal = data[:, 3:6].transpose()
            if dataset.with_one_hot_shape_id:
                shape_one_hot = np.zeros((dataset.num_shapes, coords.shape[-1]), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, normal, shape_one_hot])
            else:
                point_set = np.concatenate([coords, normal])
        else:
            if dataset.with_one_hot_shape_id:
                shape_one_hot = np.zeros((dataset.num_shapes, coords.shape[-1]), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, shape_one_hot])
            else:
                point_set = coords
        extra_batch_size = configs.evaluate.num_votes * math.ceil(total_num_points_in_shape / dataset.num_points)
        total_num_voted_points = extra_batch_size * dataset.num_points
        num_repeats = math.ceil(total_num_voted_points / total_num_points_in_shape)
        shuffled_point_indices = np.tile(np.arange(total_num_points_in_shape), num_repeats)
        shuffled_point_indices = shuffled_point_indices[:total_num_voted_points]
        np.random.shuffle(shuffled_point_indices)
        start_class, end_class = meter.part_class_to_shape_part_classes[ground_truth[0]]

        # model inference
        inputs = torch.from_numpy(
            point_set[:, shuffled_point_indices].reshape(-1, extra_batch_size, dataset.num_points).transpose(1, 0, 2)
        ).float().to(configs.device)
        with torch.no_grad():
            vote_confidences = F.softmax(model(inputs), dim=1)

            vote_confidences, vote_predictions = vote_confidences[:, start_class:end_class, :].max(dim=1)
            vote_confidences = vote_confidences.view(total_num_voted_points).cpu().numpy()
            vote_predictions = (vote_predictions + start_class).view(total_num_voted_points).cpu().numpy()

        update_shape_predictions(vote_confidences, vote_predictions, shuffled_point_indices,
                                 confidences, predictions, total_num_voted_points)
        
        iou = update_stats(stats, ground_truth, predictions, shape_id, start_class, end_class)
        
        if configs.visual:
            filename_iou = os.path.join(visual_dir,'ccm_shapenet_iou.txt')
            with open(filename_iou, 'a') as single_iou_save:
                single_iou_save.write(str(shape_index) + ' ' + str(iou) + '\n')
                
            fout = open(os.path.join(visual_dir, str(shape_index) + '_pred.obj'), 'w')
            fout_gt = open(os.path.join(visual_dir, str(shape_index) + '_gt.obj'), 'w')
            gt = ground_truth - ground_truth.min()
            pre = predictions - predictions.min()
            xyz = data[:, :3]
            xyz = xyz - xyz.mean(axis=0)
            radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
            xyz /= (radius * 2.2) / 800
            for i in range(gt.shape[0]):
                color = shapenet_color_classes[pre[i]]
                fout.write('v %f %f %f %d %d %d\n' % (
                    data[i, 0], data[i, 1], data[i, 2], color[0], color[1],
                    color[2]))

                color_gt = shapenet_color_classes[gt[i]] 
                fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            data[i, 0], data[i, 1], data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            fout.close()
            fout_gt.close()

    ttt = time.time() - since
    log_string('Evaluate complete in {:.0f}m {:.0f}s'.format(ttt // 60, ttt % 60))
    log_string('clssIoU: {}'.format('  '.join(map('{:>8.2f}'.format, stats[:, 0] / stats[:, 1] * 100))))
    log_string('meanIoU: {:4.2f}'.format(stats[:, 0].sum() / stats[:, 1].sum() * 100))

@numba.jit()
def update_shape_predictions(vote_confidences, vote_predictions, shuffled_point_indices,
                             shape_confidences, shape_predictions, total_num_voted_points):
    for p in range(total_num_voted_points):
        point_index = shuffled_point_indices[p]
        current_confidence = vote_confidences[p]
        if current_confidence > shape_confidences[point_index]:
            shape_confidences[point_index] = current_confidence
            shape_predictions[point_index] = vote_predictions[p]


@numba.jit()
def update_stats(stats, ground_truth, predictions, shape_id, start_class, end_class):
    iou = 0.0
    for i in range(start_class, end_class):
        igt = (ground_truth == i)
        ipd = (predictions == i)
        union = np.sum(igt | ipd)
        intersection = np.sum(igt & ipd)

        if union == 0:
            iou += 1
        else:
            iou += intersection / union
    iou /= (end_class - start_class)
    stats[shape_id][0] += iou
    stats[shape_id][1] += 1
    return iou


if __name__ == '__main__':
    evaluate()